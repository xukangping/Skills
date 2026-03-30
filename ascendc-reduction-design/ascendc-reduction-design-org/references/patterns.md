# Reduction 类算子 5 大场景切分方案

> 每个场景包含：多核切分策略、UB 切分策略、Buffer 布局、Tiling 关键字段、代码模板

---

## 目录

- [S1: AR 模式（最内维归约）](#s1-ar-模式最内维归约)
  - [S1-A: R 全载入 UB](#s1-a-r-全载入-ub)
  - [S1-B: R 需切片（CutR）](#s1-b-r-需切片cutr)
- [S2: RA 模式（最外维归约）](#s2-ra-模式最外维归约)
- [S3: ARA/ARARA 多轴模式](#s3-araarara-多轴模式)
  - [S3-A: BlockCutA（R 不大）](#s3-a-blockcutar-不大)
  - [S3-B: Welford Online（R 需 UB 切片）](#s3-b-welford-onliner-需-ub-切片)
  - [S3-C: Group Reduce（R 跨核）](#s3-c-group-reducer-跨核)
- [S4: Norm 类 Row-Reduce 两阶段](#s4-norm-类-row-reduce-两阶段)
- [S5: 全局归约](#s5-全局归约)
- [通用 Tiling 字段定义](#通用-tiling-字段定义)

---

## S1: AR 模式（最内维归约）

适用: reduce_sum(axes=-1), reduce_max(axes=-1), softmax(dim=-1) 等

### S1-A: R 全载入 UB

**条件**: `R × sizeof(PromoteT) ≤ UB可用空间 / BUFFER_NUM`

**多核切分**:
```
blockFactor = ceil(A / coreNum)
每核处理行范围: [blockIdx * blockFactor, min((blockIdx+1) * blockFactor, A))
最后一核可能少于 blockFactor 行（尾核处理）
```

**UB 切分**:
```
ubFactorA = min(blockFactor, UB可用空间 / (R × sizeof(PromoteT) × BUFFER_NUM))
每次 CopyIn ubFactorA 行 × R 列
```

**Buffer 布局**:
```
┌─────────────────────────────┐
│ inQueue[0]: ubFactorA × R   │ ← 双缓冲 ping
│ inQueue[1]: ubFactorA × R   │ ← 双缓冲 pong
├─────────────────────────────┤
│ outQueue[0]: ubFactorA × 1  │ ← 归约结果（每行一个标量）
│ outQueue[1]: ubFactorA × 1  │
├─────────────────────────────┤
│ castBuf: ubFactorA × R      │ ← 仅 FP16/BF16 时需要（Cast→FP32）
├─────────────────────────────┤
│ reduceBuf: 64 × sizeof(FP32)│ ← ReduceSum 临时空间
└─────────────────────────────┘
```

**Kernel 模板（命令式）**:
```cpp
// AR 全载 - 基础归约
void Compute(int tileIdx) {
    auto xLocal = inQueue.DeQue<T>();
    auto yLocal = outQueue.AllocTensor<float>();
    auto castLocal = castBuf.Get<float>();

    // 精度提升
    if constexpr (sizeof(T) < sizeof(float)) {
        Cast(castLocal, xLocal, RoundMode::CAST_NONE, R);
    }

    // 逐行归约
    for (int row = 0; row < curRowCount; row++) {
        // ReduceSum: 输入连续 R 个元素 → 输出 1 个标量
        ReduceSum(yLocal[row], castLocal[row * R], workLocal, R);
    }

    outQueue.EnQue(yLocal);
    inQueue.FreeTensor(xLocal);
}
```

### S1-B: R 需切片（CutR）

**条件**: `R × sizeof(PromoteT) > UB可用空间`

**多核切分**: 同 S1-A，按 A 轴均分

**UB 切分**:
```
cutRSize = UB可用空间 / (sizeof(PromoteT) × BUFFER_NUM)
cutRSize = cutRSize / VL * VL                   ← 对齐到向量长度
rLoopCount = ceil(R / cutRSize)
lastCutR = R - (rLoopCount - 1) * cutRSize      ← 尾块
```

**Kernel 模板**:
```cpp
void ComputeWithCutR(int rowIdx) {
    float partialSum = 0.0f;

    for (int rLoop = 0; rLoop < rLoopCount; rLoop++) {
        int curR = (rLoop < rLoopCount - 1) ? cutRSize : lastCutR;
        int rOffset = rLoop * cutRSize;

        // CopyIn 当前 R 片段
        DataCopy(xLocal, xGm[rowIdx * R + rOffset], curR);

        // Cast + ReduceSum
        if constexpr (sizeof(T) < sizeof(float)) {
            Cast(castLocal, xLocal, RoundMode::CAST_NONE, curR);
            ReduceSum(tmpLocal, castLocal, workLocal, curR);
        } else {
            ReduceSum(tmpLocal, xLocal, workLocal, curR);
        }

        // 累加 partial
        partialSum += tmpLocal.GetValue(0);
    }

    // 写出最终结果
    yLocal.SetValue(rowIdx, partialSum);
}
```

---

## S2: RA 模式（最外维归约）

适用: reduce_sum(axes=0), 形如 R[B] × A[C×H×W]

**多核切分**:
```
方案1: 按 A 轴均分（A 足够大时）
  blockFactorA = ceil(A / coreNum)
  每核处理 A 的连续片段，遍历全部 R

方案2: 按 R 轴均分（A 太小时，走 Group Reduce 路线）
  参见 S3-C
```

**UB 切分**:
```
每次载入: cutRSize × A_slice
cutRSize = UB / (A_aligned × sizeof(PromoteT) × BUFFER_NUM)
```

**Kernel 模板**:
```cpp
// RA 模式: R 在外，A 在内（连续）
void ProcessRA() {
    // 初始化累加器为 0
    Duplicate(accumLocal, float(0), A_aligned);

    for (int rLoop = 0; rLoop < rLoopCount; rLoop++) {
        int rOffset = rLoop * cutRSize;
        int curR = min(cutRSize, R - rOffset);

        for (int r = 0; r < curR; r++) {
            // 搬入一行 A 个元素
            DataCopy(xLocal, xGm[(rOffset + r) * A + myAStart], myALen);

            // 逐元素累加 (accumLocal += xLocal)
            Add(accumLocal, accumLocal, xLocal, myALen);
        }
    }
    // 写出
    DataCopy(yGm[myAStart], accumLocal, myALen);
}
```

**注意**: RA 模式下 A 维内存连续，单次 burst 效率高。

---

## S3: 多轴归约模式（ARA / ARAR / ARARA / ...）

适用: 归约轴不在最外或最内，或归约轴有多个（如 axes=[1,3]）

### 多轴归约的 Shape 三步变换

任意 N 维 shape + 任意 axes 组合，经过三步变换压缩为 A/R 交替序列：

```
Step 1: EliminateOne()
  消除 size=1 且内存连续的维度（不影响布局的冗余维）
  例: [1, 8, 1, 4] axes=[2] → [8, 4] axes=[0]（size=1 的 dim0/dim2 消除）
  注意: 非连续的 size=1 维度保留（stride 不匹配说明有实际内存含义）

Step 2: PadDimForNonContiguous()
  检测非连续轴（stride[i] ≠ stride[i+1] × shape[i+1]），在其后插入 size=1 填充维度
  目的: 让后续 MergeAxis 能正确识别非连续边界

Step 3: MergeAxis()
  合并相邻同类型（同为 A 或同为 R）的连续轴
  关键: 类型不同时立即停止合并 → 保留 A/R 交替结构
```

**实际算子示例**：

| 算子 | Shape | Axes | 变换过程 | 最终 Pattern |
|------|-------|------|---------|-------------|
| reduce_max | [2,100,4] | [1,2] | R 轴相邻合并: R=100×4=400 → [2, 400] | **AR** (2维) |
| reduce_sum | [2,3,4,5] | [1,3] | 不相邻不合并: A[2],R[3],A[4],R[5] | **ARAR** (4维) |
| reduce_sum | [2,3,4,5] | [0,2] | 不相邻不合并: R[2],A[3],R[4],A[5] → pad→9维 | **ARARARARA** (9维) |
| reduce_sum | [2048,2,48,2,2,2] | [1,3,5] | 交替不合并: A,R,A,R,A,R → pad→8维 | **ARARARAR** (8维) |
| reduce_log_sum_exp | [1,2,3,4,5] | [2,4] | 消1→合并A[0,1]→ A[2],R[3],A[4],R[5] → pad | **ARARARARA** (9维) |
| **bn_training_reduce** | [N,C,H,W] | [0,2,3] | R[N],A[C],R[H×W]（相邻R合并）→ 前置1 → ARAR | **ARAR** (4维) |

**典型场景：BatchNorm 的 "保留 C、归约 N/H/W"**：

```
bn_training_reduce: 对 NCHW 张量沿 N,H,W 归约，保留 C 通道维
  输入: [N, C, H, W]    axes=[0, 2, 3]
         R  A  R  R

  MergeAxis: axes 2,3 相邻且同为 R → 合并 → R[N], A[C], R[H×W]
  EliminateOne 前置 shape[0]=1 → A[1], R[N], A[C], R[H×W]
  最终: shapeSize=4 → ARAR

  输出: sum[C] 和 squareSum[C]（双输出，每个通道一个值）
  特点: 双输出归约 — 同一遍扫描同时算 Σx 和 Σx²，避免二次遍历
```

**具体变换 trace（axes=[1,3] on [2,3,4,5]）**：

```
原始: shape=[2, 3, 4, 5], axes=[1, 3]
       A    R    A    R    ← 类型标记

EliminateOne: 无 size=1 维，不变
PadDimForNonContiguous: 全连续，不变
MergeAxis:
  i=0: dim0 是 A，dim1 是 R → 类型不同，停止 → viewShape[0]=2 (A)
  i=1: dim1 是 R，dim2 是 A → 类型不同，停止 → viewShape[1]=3 (R)
  i=2: dim2 是 A，dim3 是 R → 类型不同，停止 → viewShape[2]=4 (A)
  i=3: dim3 是 R，结束 → viewShape[3]=5 (R)

结果: shapeSize=4 → Pattern = ARAR
      viewShape = [2, 3, 4, 5]
      4 层嵌套循环: A[2] × R[3] × A[4] × R[5]
```

**具体变换 trace（axes=[1,2] on [2,100,4]，相邻合并）**：

```
原始: shape=[2, 100, 4], axes=[1, 2]
       A     R     R    ← 类型标记

MergeAxis:
  i=0: dim0 是 A，dim1 是 R → 不同，停止 → viewShape[0]=2 (A)
  i=1: dim1 是 R，dim2 是 R → 相同且连续，合并 → viewShape[1]=100×4=400 (R)

结果: shapeSize=2 → Pattern = AR
      viewShape = [2, 400]
      等价于对最后一维做 R=400 的 AR 归约
```

### 高维 Pattern（≥5 维）的统一处理

变换后维度 ≥ 5 时，框架通过 PadDimOne() 填充 size=1 维度统一到 **8 维（ARARARAR）或 9 维（ARARARARA）**：

```
5 维 ARARA → pad 到 9 维 ARARARARA
6 维 ARARAR → pad 到 8 维 ARARARAR
7 维 ARARARA → pad 到 9 维 ARARARARA
8 维 → 直接 ARARARAR
9 维 → 直接 ARARARARA
```

**Kernel 侧执行方式**（ReduceSchAuxBase）：

```
高维 Pattern 通过递归模板展开为嵌套循环:

  IterateInnerA<0, N>()   ← 遍历所有 A 轴（递归模板，编译期展开）
    for a0 in A_axis_0:
      for a1 in A_axis_1:
        ...
          LinearComputeR()  ← 处理对应 R 轴归约（支持二分累加）
            for r in R_axis:
              CopyIn → PreReduce → ReduceCompute → DoCaching → PostReduce → CopyOut

实际效果: ARAR [2,3,4,5] 展开为:
  for a0 in range(2):       ← A 轴 0
    for r0 in range(3):     ← R 轴 0
      for a1 in range(4):   ← A 轴 1
        for r1 in range(5): ← R 轴 1
          Reduce(tile)
```

**非连续多轴的数据搬运**：

多轴归约时内存往往不连续（如 axes=[0,2] 使得 R 轴散布在内存中），框架通过以下机制处理：

```
1. sliceNum/sliceShape/sliceStride 追踪合并轴内的子结构
   → MergeAxis 合并时记录非连续断点

2. A5: CopyInWithNddma() — 多维 DMA 自动处理 stride 跳跃（仅 A5 及后续芯片）
   A2/A3: 不支持 NDDMA，需用 DataCopyPad + stride 参数 或 逐 slice 循环搬运

3. Tiling 侧 IsUseNddma() 决策（仅 A5）:
   - 最内维非连续 (stride[last] ≠ 1) → NDDMA
   - CacheLine 切点跨 >2 维 → NDDMA
   - 否则逐 slice DataCopyPad

A2/A3 非连续搬运替代方案:
   - DataCopyPad 的 blockCount/blockLen/srcStride 参数配置 stride copy
   - 或外层循环逐 slice 搬运（每次搬连续片段，循环处理不连续间隔）
```

---

### S3-A: BlockCutA（R 不大）

**条件**: R 可被单核完整处理（R × sizeof(T) ≤ 单核 UB 可循环处理完）

**多核切分**:
```
outerA = A0 (第一个 A 维)
blockFactor = ceil(outerA / coreNum)
每核处理: outerA[start:end] 的所有 R × innerA 数据
```

**UB 切分**:
```
ubFactorR = min(R, UB / (innerA × sizeof(PromoteT) × BUFFER_NUM))
ubFactorA = 1  (一次处理 1 个外层 A 块)

每次 CopyIn: ubFactorR × innerA 的数据块
```

**多轴 ARA 处理流程**:
```
for outerA in [myStart, myEnd):
    partial = 初始值 (0 for sum, -inf for max)
    for rLoop in range(ceil(R / ubFactorR)):
        CopyIn(outerA, rLoop * ubFactorR, innerA)
        partial = Reduce(partial, currentTile)
    CopyOut(outerA, partial)  ← 输出 shape: [outerA, innerA]
```

### S3-B: Welford Online（R 需 UB 切片）

**适用**: reduce_var / reduce_std 的 ARA 模式

参见 `references/algorithms.md#welford` 的完整算法。

**Tiling 关键字段**:
```
ubFactorA: A 轴 UB 切片大小
ubFactorR: R 轴 UB 切片大小
factorACntPerCore: 每核 A 轴工作量
factorRCntPerCore: 每核 R 轴工作量（Normal模式 = 全部R）
```

### S3-C: Group Reduce（R 跨核）

**条件**: R 太大，单核无法完成全部 R 归约；同时 A 太小不足以充分利用多核

**多核切分**:
```
groupR = ceil(R / maxRPerCore)         ← R 分成 groupR 组
aBlocks = ceil(A / blockFactorA)       ← A 也可能分块
realCoreNum = min(coreNum, groupR × aBlocks)

Phase1（各核独立）:
  core_id → (aBlockIdx, rGroupIdx) = (id / groupR, id % groupR)
  每核处理: A[aBlockIdx 段] × R[rGroupIdx 段]
  输出 partial → workspace[core_id × outSize]

SyncAll()

Phase2（合并）:
  对每个 aBlock: 遍历 groupR 个 partial，合并为最终结果
```

**Workspace 计算**:
```cpp
// workspace 大小 = 核数 × 对齐后的输出大小 × 2（value + index for argmax）
outAAlign = CeilDiv(A, ELEMENT_PER_BLOCK) * ELEMENT_PER_BLOCK;
workspaceSize = CeilDiv(outAAlign * 2 * sizeof(int32_t) * coreNum, 256) * 256;
```

**适用算子**: arg_max_v2 (TILING_KEY=30001), reduce_var (groupR>1)

---

## 多输出归约的 Buffer 规划

许多归约算子需要**同一遍扫描输出多个结果**，比单输出多占 UB。以下给出通用 Buffer 方程。

### 常见多输出场景

| 算子 | 输出数 | 输出内容 | 累加器 dtype |
|------|--------|---------|-------------|
| bn_training_reduce | 2 | sum[C] + squareSum[C] | FP32 |
| reduce_var / reduce_std | 2 | mean[A] + var[A] | FP32 |
| reduce_std_with_mean | 2 | std[A] + mean[A] | FP32 |
| arg_max / arg_min | 2 | value[A] + index[A] | FP32 + INT32/FP32 |
| arg_max_with_value | 2 | value[A] + index[A] | 同上 |

### 通用 UB 方程

```
输入:
  T_in  = 输入元素大小 (FP16=2, FP32=4)
  T_acc = 累加器元素大小 (通常 FP32=4)
  K     = 输出个数（双输出 K=2，三输出 K=3）
  A_aligned = 保留轴对齐后大小

Buffer 清单:
  inBuf × 2              = tileSize × T_in × 2          ← 输入双缓冲
  castBuf (仅低精度)      = tileSize × T_acc             ← FP16/BF16 → FP32
  accumBuf × K           = A_aligned × T_acc × K         ← K 个累加器
  tmpBuf                 = A_aligned × T_acc             ← 中间计算（如 x²）
  outBuf × 2             = A_aligned × T_acc × 2         ← 输出双缓冲（可选）

UB 方程:
  tileSize × (T_in × 2 + T_acc)              ← 输入 + Cast
  + A_aligned × T_acc × (K + 1 + 2)          ← K 累加器 + tmp + out双缓冲
  ≤ UB_SIZE (A2/A3: 192KB)

求解 tileSize:
  fixedBuf = A_aligned × T_acc × (K + 3)
  perTileBuf = T_in × 2 + T_acc
  tileSize = (UB_SIZE - fixedBuf) / perTileBuf
```

### 双输出示例：bn_training_reduce (K=2, FP16 输入)

```
A_aligned = CeilAlign(C, 8)        ← FP32 按 32B 对齐: 32/4=8
T_in = 2 (FP16), T_acc = 4 (FP32)

fixedBuf = A_aligned × 4 × (2 + 3) = A_aligned × 20
  分解: sumBuf(A×4) + sqSumBuf(A×4) + tmpBuf(A×4) + outBuf×2(A×4×2)

perTileBuf = 2 × 2 + 4 = 8   (FP16 双缓冲 + FP32 Cast)

tileSize = (192KB - A_aligned × 20) / 8

示例: C=64 → A_aligned=64
  fixedBuf = 64 × 20 = 1280B
  tileSize = (196608 - 1280) / 8 = 24416 元素
  实际取 tileRows = tileSize / A_aligned = 381 行
```

### 双输出示例：ArgMax (K=2, value + index)

```
A_aligned = CeilAlign(A, 8)

累加器:
  maxValBuf: A_aligned × 4 (FP32)      ← 当前最大值
  maxIdxBuf: A_aligned × 4 (FP32)      ← 当前最大值的下标（存为 float）

额外:
  cmpBuf: A_aligned / 8 (uint8_t)      ← Compare mask
  注意: A2/A3 上 Select 不支持 int32 dst → 下标存为 float，最后 Cast 为 int32

fixedBuf = A_aligned × 4 × 2 + A_aligned × 4 + max(A_aligned/8, 32) + outBuf
```

### 跨核合并时 Workspace 方程

多输出时 workspace 也要乘 K：

```
workspace = coreNum × CeilAlign(A_aligned × T_acc × K, cacheLineSize)

示例: bn_training_reduce, C=256, 20核
  workspace = 20 × CeilAlign(256 × 4 × 2, 256) = 20 × 2048 = 40KB
```

### NCHW 布局的数据搬运注意事项

"保留 C 归约 NHW" 场景（bn_training_reduce）中，NCHW 布局的内存不是按通道连续的：

```
NCHW 内存布局: x[n,c,h,w] 地址 = n×C×H×W + c×H×W + h×W + w
  同一 (n,h,w) 的 C 个通道值步长 = H×W（不连续！）
  同一 (n,c) 的 H×W 个空间位置连续

两种处理方式:
  方式A（推荐，A2/A3/A5 通用）: 按通道遍历，外层 (n, c)，内层连续搬 H×W
    → 每次搬运连续内存（高效），需 C 个独立累加器
    → 适合大多数场景

  方式B（仅 A5）: 用 NDDMA 多维搬运，一次配置自动处理 stride 跳跃
    → A2/A3 不支持 NDDMA

  方式C（A2/A3）: DataCopyPad 配置 stride 参数
    → blockCount=R 行数, blockLen=连续片段字节数, srcStride=跳跃步长
    → 适合有规律的 stride 模式

实际 batch_norm_v3 采用方式B: 按通道遍历，每通道累加 N×H×W 个连续值
  参见 /ascendc-api-best-practices 获取 DataCopyPad stride 配置详情
```

---

## S4: Norm 类 Row-Reduce 两阶段

适用: rms_norm, layer_norm_v4, batch_norm

**核心特征**: 每行归约得到统计量(mean/rstd)，再广播回逐元素变换。

**多核切分**:
```
blockFactor = ceil(numRow / coreNum)
每核处理 blockFactor 行（最后一核可能更少）
```

**UB 切分 4 档**:

| 档位 | 条件 | UB 策略 | 特点 |
|------|------|---------|------|
| **MergeN** | numCol ≤ 2000 | 多行打包 rowFactor 行 | 最高效，rstd 广播乘 |
| **Normal** | numColAlign ≤ UB 容量 | 单行完整 ubFactor=numColAlign | 标准路径 |
| **SplitD** | numCol > UB | 分片遍历列 | 两遍扫描 |
| **SingleRow** | 极端压缩 | 最小 buffer | FP16 + FP32 gamma |

**两阶段流水（Normal 模式示例）**:
```
Phase1 (Former): 求 rstd
  for row in myRows:
    CopyIn(x[row])
    xFp32 = Cast(x, FP32)          ← 精度提升
    sqx = Mul(xFp32, xFp32)        ← x²
    sum = ReduceSum(sqx)            ← Σx²
    SetFlag<V_S>()                  ← Vector→Scalar 同步
    WaitFlag<V_S>()
    mean_sq = sum / numCol          ← 标量计算
    rstd = 1.0 / sqrt(mean_sq + eps)
    SetFlag<S_V>()                  ← Scalar→Vector 同步
    WaitFlag<S_V>()

Phase2 (Latter): 归一化输出
  for row in myRows:
    CopyIn(x[row]), CopyIn(gamma)
    y = x * rstd * gamma + beta
    CopyOut(y[row])
```

**SplitD 模式（numCol > UB）**:
```
ubFactor = UB可容纳列数（对齐后）
colLoops = ceil(numCol / ubFactor)

Phase1: 遍历 colLoops 片段，累加 partial_sum
Phase2: 用累积的 rstd 对每片段做 y = x * rstd * gamma
```

---

## S5: 全局归约

适用: reduce_sum(axes=所有轴), reduce_max(axes=所有轴)

**多核切分**:
```
totalElements = Πshapeᵢ
elementsPerCore = ceil(totalElements / coreNum)
每核处理: [blockIdx * elementsPerCore, min((blockIdx+1) * elementsPerCore, total))
```

**两阶段执行**:
```
Stage1: 各核独立归约 → partial[blockIdx]
  partial = ReduceSum(mySlice)
  workspace[blockIdx * SLOT_STRIDE] = partial   ← SLOT_STRIDE=64B 防 bank conflict

Stage2: 合并
  方式A: Atomic 累加
    SetAtomicAdd<float>()
    DataCopy(resultGm, partial, 1)
    SetAtomicNone()
    SyncAll()

  方式B: 两阶段显式合并
    DataCopy(workspace[blockIdx], partial, 1)
    SyncAll()
    if (blockIdx == 0):
      finalResult = 0
      for i in range(coreNum):
        finalResult += workspace[i * SLOT_STRIDE]
      DataCopy(yGm, finalResult, 1)
```

---

## 通用 Tiling 字段定义

以下是 ops-math/ops-nn 中 Reduction 算子常用的 Tiling 数据结构字段：

### 基础归约 Tiling（ReduceOpTilingData）

| 字段 | 类型 | 含义 |
|------|------|------|
| `factorACntPerCore` | uint64 | 每核 A 轴工作量 |
| `factorATotalCnt` | uint64 | A 轴总工作单元 |
| `ubFactorA` | uint64 | UB 的 A 轴切片大小 |
| `factorRCntPerCore` | uint64 | 每核 R 轴工作量 |
| `factorRTotalCnt` | uint64 | R 轴总工作单元 |
| `ubFactorR` | uint64 | UB 的 R 轴切片大小 |
| `groupR` | uint64 | R 轴分组数（>1 触发 Group Reduce）|
| `outSize` | uint64 | 输出缓冲区大小 |
| `basicBlock` | uint64 | 输入 UB 缓冲区大小 |
| `resultBlock` | uint64 | 输出/中间缓冲区大小 |
| `coreNum` | int32 | 使用核数 |
| `useNddma` | int32 | 是否使用 NDDMA |
| `shape[8]` | uint64[] | 各维度大小 |
| `stride[8]` | int64[] | 各维度步进 |

### ArgMax 系列 Tiling

| 字段 | 类型 | 含义 |
|------|------|------|
| `aSize` | uint64 | 归约轴前所有维度之积 |
| `rSize` | uint64 | 归约维度大小 |
| `nextASize` | uint64 | 归约轴后所有维度之积 |
| `cutASize` | uint16 | UB 的 A 切片 |
| `cutRSize` | uint16 | UB 的 R 切片 |
| `cutNextASize` | uint16 | UB 的 nextA 切片 |
| `realCoreNum` | uint64 | 实际使用核数 |
| `blkFactor` | uint64 | 每核主维度块大小 |
| `blkTailFactor` | uint64 | 尾核主维度块大小 |
| `tilingKey` | uint64 | 策略选择键 |
| `aRaMode` | uint64 | ARA 子模式 (1-6) |
| `workSpaceSize` | uint64 | Group Reduce workspace |

### Norm 类 Tiling（RmsNorm/LayerNorm）

| 字段 | 类型 | 含义 |
|------|------|------|
| `num_row` | uint64 | 输入行数 (M) |
| `num_col` | uint64 | 输入列数 (N) |
| `num_col_align` | uint64 | 对齐后列数 |
| `block_factor` | uint64 | 每核行数 |
| `row_factor` | uint32 | 每次迭代处理行数 |
| `ub_factor` | uint32 | 每次迭代处理列数 |
| `reduce_mask` | uint32 | 归约 mask 配置 |
| `epsilon` | float | 数值稳定常数 |
| `avg_factor` | float | 1.0/num_col |

### Softmax 系列 Tiling

| 字段 | 类型 | 含义 |
|------|------|------|
| `a` (或 `totalA0Len`/`totalA1Len`) | uint64 | A 维大小 |
| `r` (或 `totalRLen`) | uint64 | R 维大小 |
| `rAligned` | uint64 | R 对齐后大小 |
| `ubFactor` | uint64 | UB 处理大小 |
| `aBlockFactor` | uint64 | 每核 A 行数 |
| `tilesPerCore` | uint64 | 每核 tile 数 |
| `rLoopCount` | uint64 | R / VL_FP32 |
