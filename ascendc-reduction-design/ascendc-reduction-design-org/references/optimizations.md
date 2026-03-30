# Reduction 类算子性能优化范式

> 9 项从 ops-math / ops-nn 实际代码中提取的性能优化技术，含代码示例和注意事项

---

## 目录

- [1. 精度提升（Type Promotion）](#1-精度提升type-promotion)
- [2. 双缓冲流水线](#2-双缓冲流水线)
- [3. V-S 同步（Vector-Scalar Pipeline Handshake）](#3-v-s-同步vector-scalar-pipeline-handshake)
- [4. Atomic 写保护](#4-atomic-写保护)
- [5. NDDMA 非连续搬运](#5-nddma-非连续搬运)
- [6. Welford 分组合并](#6-welford-分组合并)
- [7. CacheLine 对齐优化](#7-cacheline-对齐优化)
- [8. 多策略择优（Priority-based Selection）](#8-多策略择优priority-based-selection)
- [9. 空核跳过与尾核处理](#9-空核跳过与尾核处理)

---

## 1. 精度提升（Type Promotion）

**问题**: FP16/BF16 累加数千次后精度严重损失，导致归约结果不准确。

**规则**: 所有低精度输入在归约前必须 Cast 到 FP32。

### 类型提升映射表

| 输入类型 | 提升类型 | 归约后回退 | 回退 RoundMode |
|---------|---------|-----------|---------------|
| half (FP16) | float (FP32) | Cast 回 half | CAST_NONE |
| bfloat16 | float (FP32) | Cast 回 bfloat16 | CAST_RINT |
| float | float（不变） | — | — |
| int8/uint8 | half | 视需要 | CAST_NONE |
| int32/int64 | 同类型（不变）| — | — |

### 命令式实现模板

```cpp
void ComputeWithPromotion(LocalTensor<half>& xHalf, int len) {
    auto xFloat = castBuf.Get<float>();

    // 提升: FP16 → FP32
    Cast(xFloat, xHalf, RoundMode::CAST_NONE, len);

    // 在 FP32 下做归约
    ReduceSum(resultFloat, xFloat, workBuf, len);

    // 回退: FP32 → FP16
    Cast(resultHalf, resultFloat, RoundMode::CAST_NONE, 1);
}
```

### BF16 特殊处理

```cpp
// BF16 回退必须用 CAST_RINT（四舍五入到最近偶数）
if constexpr (std::is_same_v<T, bfloat16_t>) {
    Cast(outBf16, outFloat, RoundMode::CAST_RINT, len);
} else {
    Cast(outFp16, outFloat, RoundMode::CAST_NONE, len);
}
```

### Buffer 开销估算

精度提升会使计算 buffer 翻倍（FP16 2B → FP32 4B），Tiling 时需纳入 UB 预算：
```
实际 UB 占用 = 输入(sizeof(T) × len) + 计算(sizeof(float) × len) + 输出(sizeof(T) × outLen)
```

---

## 2. 双缓冲流水线

**目标**: 隐藏 MTE2（搬入）/ MTE3（搬出）延迟，与 Vector 计算重叠。

### 时间线效果

```
无双缓冲:
MTE2:  [Copy0]         [Copy1]         [Copy2]
Vector:        [Comp0]         [Comp1]         [Comp2]
MTE3:                  [Out0]          [Out1]          [Out2]

有双缓冲:
MTE2:  [Copy0][Copy1][Copy2]
Vector:       [Comp0][Comp1][Comp2]
MTE3:               [Out0] [Out1] [Out2]
```

### 实现方式

```cpp
// 声明双缓冲队列
static constexpr int BUFFER_NUM = 2;
TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

// 初始化
pipe.InitBuffer(inQueue, BUFFER_NUM, tileSize * sizeof(T));
pipe.InitBuffer(outQueue, BUFFER_NUM, outSize * sizeof(T));

// 三段式流水
void Process() {
    for (int i = 0; i < loopCount; i++) {
        CopyIn(i);    // MTE2: GM → UB (自动 ping-pong)
        Compute(i);   // Vector: 计算
        CopyOut(i);   // MTE3: UB → GM
    }
}

void CopyIn(int i) {
    auto x = inQueue.AllocTensor<T>();  // 自动从空闲 buffer 分配
    DataCopy(x, xGm[i * tileLen], curLen);
    inQueue.EnQue(x);                   // 标记为已填充
}

void Compute(int i) {
    auto x = inQueue.DeQue<T>();        // 等待 MTE2 完成，取已填充 buffer
    auto y = outQueue.AllocTensor<T>();
    // ... 计算 ...
    outQueue.EnQue(y);                  // 标记为已计算
    inQueue.FreeTensor(x);              // 释放输入 buffer 供下次 CopyIn
}
```

### 注意事项

- `BUFFER_NUM = 2` 是性能和 UB 空间的折衷（3 缓冲收益递减）
- inQueue/outQueue 通过 EnQue/DeQue 自动做流水线同步，无需手动 Event
- 如果 Compute 依赖多个输入（如 x 和 gamma），需要多个 inQueue

---

## 3. V-S 同步（Vector-Scalar Pipeline Handshake）

**问题**: Vector 流水线产生归约结果 → 需要传递给 Scalar 流水线做后续标量计算 → 再传回 Vector。

**典型场景**: Norm 类算子中 `ReduceSum → 求 rstd → 广播乘 gamma`

### 正确的同步模式

```cpp
// Phase 1: Vector 归约
ReduceSum(sumBuf, xSquare, workBuf, numCol);

// V→S 同步: 等 Vector 流水线完成，Scalar 可以读结果
SetFlag<HardEvent::V_S>();
WaitFlag<HardEvent::V_S>();

// Scalar 计算: rstd = 1 / sqrt(mean + eps)
float sum = sumBuf.GetValue(0);
float meanSq = sum / static_cast<float>(numCol);
float rstd = 1.0f / sqrtf(meanSq + epsilon);

// S→V 同步: Scalar 结果写回，Vector 可以使用
SetFlag<HardEvent::S_V>();
WaitFlag<HardEvent::S_V>();

// Phase 2: Vector 广播乘
Muls(yBuf, xBuf, rstd, numCol);       // y = x * rstd
Mul(yBuf, yBuf, gammaBuf, numCol);     // y = y * gamma
```

### 常见错误

```cpp
// 错误: 缺少 V_S 同步
ReduceSum(sumBuf, xBuf, workBuf, len);
float sum = sumBuf.GetValue(0);        // 可能读到未完成的结果!

// 错误: 只加了 V_S 没加 S_V
SetFlag<HardEvent::V_S>();
WaitFlag<HardEvent::V_S>();
float rstd = 1.0f / sqrtf(...);
Muls(yBuf, xBuf, rstd, len);          // rstd 可能还没写入寄存器!
```

### PipeBarrier 替代（简化写法）

```cpp
// 全流水线屏障（更保守但更安全）
ReduceSum(sumBuf, xBuf, workBuf, len);
PipeBarrier<PIPE_V>();                  // 等所有 Vector 操作完成
float sum = sumBuf.GetValue(0);
// ... scalar 计算 ...
PipeBarrier<PIPE_ALL>();                // 等所有流水线完成
Muls(yBuf, xBuf, rstd, len);
```

---

## 4. Atomic 写保护

**问题**: 多核向同一 GM 地址累加 partial 结果时的写冲突。

### 场景 A: Global Reduce 的 Atomic 累加

```cpp
// 各核计算 partial sum
float partialSum = ComputeLocalSum();
tmpBuf.SetValue(0, partialSum);

// Atomic 累加到 GM
SetAtomicAdd<float>();
DataCopy(resultGm, tmpBuf, 1);
SetAtomicNone();                       // 恢复非原子模式

// 等所有核完成
SyncAll();

// DataCacheCleanAndInvalid() 确保缓存一致性（如需要）
```

### 场景 B: SplitD 模式的 partial 累加

```cpp
// Norm SplitD: 多个列片段的 partial rstd 写入同一行
for (int colLoop = 0; colLoop < colLoops; colLoop++) {
    float partialSqSum = ReduceSum(xSquareChunk);

    if (colLoop == 0) {
        // 第一片直接写
        SetAtomicNone();
        DataCopy(rstdGm[rowIdx], partialBuf, 1);
    } else {
        // 后续片段原子累加
        SetAtomicAdd<float>();
        DataCopy(rstdGm[rowIdx], partialBuf, 1);
        SetAtomicNone();
    }
}
```

### Arch35 替代方案（DataCopyCustom）

Arch35（NPU_ARCH 3003/3113）不支持向量 Atomic，改用 `DataCopyCustom`：

```cpp
#ifdef NPU_ARCH_35
    DataCopyCustomParams customParams;
    customParams.isAtomicAdd = true;
    DataCopyCustom(yGm_[gmOffset], result, n, customParams);
#else
    SetAtomicAdd<float>();
    DataCopyPad(yGm_[gmOffset], result, {1, n * sizeof(float)});
    SetAtomicNone();
#endif
```

### 不需要 Atomic 的场景

各核写的 GM 地址不重叠（如标准 AR 多核切 A）不需要 Atomic。只有 SplitD / Group Reduce Phase 2 中多核写同一地址时才需要。

### SLOT_STRIDE 防 Bank Conflict

```cpp
// Group Reduce 中各核写 workspace 时，用 64B 间隔避免 cache bank conflict
const int SLOT_STRIDE = 64 / sizeof(float);  // = 16 floats

// Core i 写入位置
int wsOffset = blockIdx * SLOT_STRIDE;
DataCopy(workspaceGm[wsOffset], partialBuf, outSize);
```

---

## 5. 非连续搬运（ARA/ARARA 多轴模式）

**问题**: ARA 模式下数据在 GM 中不连续，逐行 DataCopy 启动开销大。

> **平台限制**: NDDMA（多维 DMA）仅 A5 及后续芯片支持，**A2/A3 不支持 NDDMA**。

### A5: NDDMA 多维搬运

一次配置多维 stride，硬件自动完成不连续搬运。

**启用条件**（来自 rms_norm / reduce_var）：

```
满足全部:
- 平台为 A5 (Ascend950/arch35)
- Pattern 为 ARA（有非连续维度）
- 最后两维较小
- 第一维 > 4096（足够的数据量摊薄配置开销）
- 最后一维 < ubBlockSize (256B)
```

### 配置示例

```cpp
// 多维 DataCopy 配置
DataCopyParams params;
params.blockCount = outerDimCount;        // 外层循环次数
params.blockLen = innerDimLen * sizeof(T); // 每块连续数据大小
params.srcStride = srcStrideBytes;        // GM 中行间距
params.dstStride = dstStrideBytes;        // UB 中行间距

DataCopy(dstLocal, srcGm, params);
```

### 与逐行搬运对比（A5 NDDMA）

```
逐行搬运 (100 行):
  100 次 DataCopy 调用 → 100 次 DMA 启动开销

NDDMA (100 行, 仅 A5):
  1 次 DataCopy 调用 → 1 次 DMA 启动 + 硬件自动 100 次传输
  省去 99 次 DMA 启动开销
```

### A2/A3 替代方案：DataCopyPad stride 参数

A2/A3 不支持 NDDMA，用 DataCopyPad 的 stride 参数处理规律性不连续：

```cpp
// 搬运 blockCount 个不连续片段，每片 blockLen 字节，片间跳 srcStride 字节
DataCopyExtParams copyParams;
copyParams.blockCount = numSlices;                          // 片段数
copyParams.blockLen = sliceLen * sizeof(T);                 // 每片连续字节数
copyParams.srcStride = (totalStride - sliceLen) * sizeof(T); // 源跳跃
copyParams.dstStride = 0;                                   // UB 中紧密排列
DataCopyPad(dstLocal, srcGm, copyParams);
```

对于完全无规律的不连续模式，退化为逐 slice 循环搬运：

```cpp
for (uint32_t i = 0; i < numSlices; i++) {
    DataCopyPad(dstLocal[i * sliceLen], srcGm[i * srcStride], sliceLen);
}
```

---

## 6. Welford 分组合并

**问题**: Welford 在线算法累加数万次后，`delta / count` 的除法精度下降。

**解决**: 每 WELFORD_GROUP_NUM(=8) 个 UB 片段做一次中间 Finalize。

### 原理

```
不是: update(x1) → update(x2) → ... → update(x10000) → finalize
而是: [update×8 → merge] → [update×8 → merge] → ... → final merge

中间 merge 将 8 个小组的 (mean, M2, count) 合并为一组
重置 count，使后续 delta/count 的除法精度恢复
```

### 实现要点

```cpp
const int WELFORD_GROUP_NUM = 8;
// Group cache: 9 组 × A_aligned × sizeof(float) × 2 (mean + var)
// GROUP_CACHE_BUF_SIZE = (8+1) * 512 bytes

int groupCounter = 0;
for (int rLoop = 0; rLoop < totalRLoops; rLoop++) {
    // 正常 Welford 更新
    WelfordUpdate(xLocal, curLen, groupMean[groupCounter], ...);
    groupCounter++;

    if (groupCounter == WELFORD_GROUP_NUM) {
        // 中间合并: 8 组 → 1 组
        MergeAllGroups();
        groupCounter = 1;  // 合并结果成为新的第 0 组
    }
}
```

---

## 7. CacheLine 对齐优化

**目标**: 最大化 L2/L3 Cache 命中率，减少不必要的 cache miss。

### 来自 reduce_var 的 7 步优化中的 CacheLine 步骤

```cpp
// 找到累积数据量跨越 256B cacheLine 边界的轴
void ComputeCacheLineBlock() {
    int cacheLineSize = 256;  // bytes
    int cumSize = 1;

    for (int axis = ndim - 1; axis >= 0; axis--) {
        cumSize *= shape[axis] * sizeof(T);
        if (cumSize >= cacheLineSize) {
            // 这个轴是 cacheLine 边界轴
            cacheLineAxis = axis;
            cacheLineStep = shape[axis];
            break;
        }
    }

    // 确保每次 DMA 传输大小 ≥ cacheLineSize
    // 优化内循环使得连续访问 ≥ 256B
}
```

### Tiling 对齐规则

```
1. 输入 tile 大小必须是 32B (BLOCK_SIZE) 的整数倍
   tileLen = CeilDiv(rawLen, BLOCK_SIZE / sizeof(T)) * (BLOCK_SIZE / sizeof(T))

2. 输出地址必须 32B 对齐
   outOffset = CeilDiv(outIdx, BLOCK_SIZE / sizeof(T)) * (BLOCK_SIZE / sizeof(T))

3. UB 内 buffer 分配自动对齐（pipe.InitBuffer 内部处理）

4. DMA 传输长度推荐 256B 的整数倍（CacheLine 对齐）
```

---

## 8. 多策略择优（Priority-based Selection）

**核心思想**: 同一算子根据输入 shape/dtype/UB 容量选择不同实现策略。

### layer_norm_v4 的策略优先级

```
优先级 100: Transpose      ← 小行数转置优化
优先级 150: TwoPassPerf    ← arch35 性能优化版
优先级 200: TwoPass        ← arch35 标准版
优先级 400: Welford        ← arch35 高精度版
优先级 1000: SingleRead    ← 通用标准路径
优先级 2000: Common        ← 最终回退

选择逻辑:
  for strategy in sorted_by_priority:
      if strategy.IsCapable(shape, dtype, ubSize, arch):
          return strategy
```

### softmax_v2 的策略优先级

```
优先级  50: AR_SMALL_R   (R≤16, A 大)       ← 转置优化
优先级 100: AR_FULL_LOAD (R≤16K)            ← 最快
优先级 200: AR_RECOMPUTE (R 大)             ← 重计算
优先级 300: ARA_FULL_LOAD(多维, R 可装入)    ← 多维最快
优先级 400: ARA_RECOMPUTE(多维, R 大)        ← 通用回退
```

### arg_max_v2 的策略选择

```
R=1           → COPY_ONLY (10003)    ← 无归约
A 大, R 小    → AR_GATHER (20001)    ← R 放寄存器
A 大, R 中等  → AR_CUT_A (10001)     ← 按 A 分核
ARA 标准      → ARA_CUT_A (10002)    ← 按外层 A 分核
A×nextA 大    → ARA_CUT_A_AND_NEXT_A (10012) ← 2D 分核
R 大, A 小    → GROUP_REDUCE (30001) ← 跨核分 R
```

### 通过 TilingKey 实现编译期多态

```cpp
// Host 侧设置 tilingKey
context->SetTilingKey(tilingKey);

// Kernel 侧根据 tilingKey 分发
if (TILING_KEY_IS(10001)) {
    // AR 策略
    KernelAR<T> op;
    op.Init(...);
    op.Process();
} else if (TILING_KEY_IS(10002)) {
    // ARA 策略
    KernelARA<T> op;
    op.Init(...);
    op.Process();
} else if (TILING_KEY_IS(30001)) {
    // Group Reduce
    KernelGroupReduce<T> op;
    op.Init(...);
    op.Process();
}
```

---

## 9. 空核跳过与尾核处理

### 空核跳过

当数据量小于核数时，部分核无工作：

```cpp
__global__ __aicore__ void my_reduce(...) {
    // 优先级最高: 空核直接返回
    if (GetBlockIdx() >= tilingData.realCoreNum) return;

    // 非 AICore 跳过
    if (g_coreType == AIC) return;

    // ... 正常处理 ...
}
```

### 尾核处理

最后一个核处理的元素数可能少于标准 blockFactor：

```cpp
void Init() {
    if (GetBlockIdx() < tilingData.realCoreNum - 1) {
        // 非尾核: 标准工作量
        myRowCount = tilingData.blockFactor;
    } else {
        // 尾核: 剩余工作量
        myRowCount = tilingData.totalRows -
                     (tilingData.realCoreNum - 1) * tilingData.blockFactor;
    }

    // GM 偏移
    int rowOffset = GetBlockIdx() * tilingData.blockFactor;
    xGm.SetGlobalBuffer(x + rowOffset * numCol, myRowCount * numCol);
}
```

### 两级尾处理（ArgMax ARA_CUT_A_AND_NEXT_A）

```
coreGrid = blkNumA × blkNumNextA

核 (i, j) 的工作量:
  aWork = (i < blkNumA-1) ? blkFactorA : blkTailFactorA
  nextAWork = (j < blkNumNextA-1) ? blkFactorNextA : blkTailFactorNextA

总核数 = blkNumA × blkNumNextA（可能 < 物理核数）
```
