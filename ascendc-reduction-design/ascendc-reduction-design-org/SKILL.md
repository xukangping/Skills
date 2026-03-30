---
name: ascendc-reduction-design
description: >
  Ascend C Reduction / Norm 类算子设计指南。当用户需要以下操作时使用：
  (1) 识别归约模式（AR/RA/ARA/ARAR 等 Pattern）并设计切分方案
  (2) 选择归约算法（TwoPass / Welford / Group Reduce / Recompute）
  (3) 规划多输出归约的 Buffer（sum+sqSum、mean+var、value+index）
  (4) 处理多轴归约的 Shape 变换（不相邻 axes 合并规则、NCHW 归约方向）
  (5) 解决归约精度或性能问题（累加精度、大数吃小数、核间同步）
---

# Ascend C Reduction 类算子设计指南

> 基于 ops-math (21 算子) + ops-nn (47+ 算子) 的 Kernel/Tiling 源码深度分析

---

## 算子分类体系

| 子族 | 代表算子 | 归约操作 | 特殊处理 |
|------|---------|---------|---------|
| **基础归约** | reduce_sum/max/min/prod | 单一归约函数 | 类型提升防溢出 |
| **统计归约** | reduce_mean/var/std | 多步归约(sum→mean→var) | Welford/TwoPass 算法 |
| **索引归约** | arg_max_v2, arg_min | 归约+索引追踪 | 双输出(value+index)，分片合并 |
| **归一化** | rms_norm, layer_norm_v4, bn_training_reduce | row-wise 或 channel-wise 归约 + 变换 | rstd 广播乘；BN 保留C归约NHW（双输出 sum+sqSum）|
| **复合归约** | softmax_v2, log_softmax | max→sub→exp→sum→div | 三遍扫描 / Recompute |
| **逻辑归约** | reduce_all, reduce_any | 布尔归约 | 继承 Max/Min 操作 |

---

## 第一步：识别归约模式（ReducePattern）

把输入 shape 和归约轴**压缩为 A/R 交替序列**（A=保留轴，R=归约轴），相邻同类维度合并：

```
shape=[B,T,H], axes=[1]  → A[B] × R[T] × A[H]     → ARA
shape=[M,N],   axes=[1]  → A[M] × R[N]             → AR（最常见）
shape=[B,C,H,W], axes=[2,3] → A[B×C] × R[H×W]     → AR（相邻合并）
shape=[B,C,H,W], axes=[0,2] → R[B] × A[C] × R[H] × A[W] → RARA
```

框架内部三步变换（EliminateOne → PadDimForNonContiguous → MergeAxis）将任意 N 维压缩为 1-9 维交替序列，映射到 Pattern ID（A=10, RA=0, AR=1, ARA=2, ARAR=3, ...ARARARARA=8）。

---

## 第二步：快速决策表

根据 Pattern 和 R/UB 关系，直接查表选方案：

| 场景 | Pattern | R 与 UB 关系 | 推荐方案 | 参考 |
|------|---------|-------------|---------|------|
| **最内维归约** | AR | R 全载入 UB | 单次 AR 归约 | `references/patterns.md#S1A` |
| **最内维归约** | AR | R > UB | CutR 分片归约 | `references/patterns.md#S1B` |
| **最外维归约** | RA | 任意 | RA 累加 | `references/patterns.md#S2` |
| **中间轴/多轴** | ARA/ARARA | R 全载，单核可完成 | TwoPass | `references/patterns.md#S3A` |
| **中间轴/多轴** | ARA/ARARA | R 需 UB 切片 | Welford Online | `references/patterns.md#S3B` |
| **中间轴/多轴** | ARA/ARARA | R 大到单核跑不完 | Group Reduce | `references/patterns.md#S3C` |
| **行内归约+广播** | AR（行） | Norm 场景 | Norm 两阶段 | `references/patterns.md#S4` |
| **全局归约** | R（全维） | — | Global Reduce | `references/patterns.md#S5` |

---

## 第三步：算法选择决策树

```
(1) R 能完整装入 UB？
    ├─ YES → [TwoPass / 直接归约]
    │        Pass1 算 mean/max，Pass2 算 var/(x-mean)*rstd
    │        适用: 基础归约、Norm-Normal、Softmax-FullLoad
    │
    └─ NO
        (2) 单核能循环遍历完所有 R？
            ├─ YES → [Welford Online / Recompute]
            │   Welford: 单趟增量更新 mean+M2（var/std 专用）
            │   Recompute: 多趟扫描不存中间结果（softmax 专用）
            │
            └─ NO → [Group Reduce]
                     Phase1: 各核处理 R 子段 → partial → workspace
                     SyncAll()
                     Phase2: 合并 partial → 最终输出
```

详细算法实现模板见 `references/algorithms.md`。

---

## 第四步：实现范式选择

### 平台约束（最重要的决策）

| 平台 | 推荐范式 | 说明 |
|------|---------|------|
| **A2 (Ascend910B) / A3 (Ascend910C)** | **命令式 Class** | TPipe + TQue + 显式 EnQue/DeQue |
| **A5 (Ascend950 / arch35)** | DAG 声明式 **或** 命令式 | DAG 更简洁，但命令式也可用 |

> **DAG（`DAGSch`/`Bind`/`MemLevel::LEVEL_2`）仅 A5 支持，A2/A3 不支持。**

### 按算子特征选范式

| 算子特征 | 推荐范式 | 说明 |
|---------|---------|------|
| 纯元素级归约（sum/max/min/mean）| DAG（仅A5）或 命令式 | A5 用 atvoss 框架几行搞定 |
| 需同时输出 mean+var（std/var 类）| 命令式 Class | 自定义 V-S 同步，双输出 |
| Norm 类（rms_norm/layer_norm）| 命令式 Class | 两阶段流水，精确控制 FP32 中间 buffer |
| 索引归约（ArgMax/ArgMin）| 命令式 Class | 双输出 + 分片合并逻辑 |
| Softmax / 复合归约 | 命令式 Class | 多遍扫描 / Recompute |
| Fused 算子（reduce 后接 elewise）| 命令式 或 DAG(A5) | DAG 天然支持 Pre/Post-Reduce Elewise |

### 命令式 Kernel 标准结构（A2/A3/A5 通用）

```cpp
class MyReduceOp {
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueue;    // 双缓冲
    TQue<QuePosition::VECOUT, 2> outQueue;
    TBuf<QuePosition::VECCALC> workBuf;

    void Process() {
        for (int i = 0; i < tileCount; i++) {
            CopyIn(i);    // GM→UB (MTE2)
            Compute(i);   // 归约计算 (Vector)
            CopyOut(i);   // UB→GM (MTE3)
        }
    }
    // CopyIn: AllocTensor → DataCopy → EnQue
    // Compute: DeQue → Cast(FP16→FP32) → ReduceSum/Max → 后处理 → EnQue + FreeTensor
    // CopyOut: DeQue → DataCopy → FreeTensor
};
```

### DAG 声明式快速模板（仅 A5）

```cpp
template <typename T, typename PromoteT>
struct MyReduceDag {
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0     = Bind<Vec::Cast<PromoteT, T, 0>, OpCopyIn0>;     // 精度提升
    using ReduceOp  = Bind<Vec::ReduceSumOp<PromoteT>, Cast0>;         // 核心归约
    using Cast1     = Bind<Vec::Cast<T, PromoteT, 1>, ReduceOp>;       // 精度回退
    using CopyOut   = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs   = Elems<CopyOut>;
    using MemCfg    = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag     = DAGSch<Outputs, void, MemCfg>;
};
// 框架自动提取 PreReduce/PostReduce Elewise，编排双缓冲和多核调度
```

---

## 第五步：5 大场景切分速览

### 场景 1：AR 模式（归约最内维，最常见）

```
多核切分：blockFactor = CeilDiv(A, coreNum)，对齐到 BLOCK_SIZE/sizeof(T)
UB 切分：R 全载 → 一次 CopyIn(A_slice × R)
         R 切片 → cutRSize 循环合并 partial（ArgMax 用 UpdateResult 合并）
```

### 场景 2：RA 模式（归约最外维）

```
多核切分：按 A 维均分（连续，利于 burst）
UB 切分：每次载入 R_slice × A_full，逐 R 行累加到 accumBuf
```

### 场景 3：多轴归约（ARA / ARAR / ARARA ...）

```
三步变换: EliminateOne → PadDimForNonContiguous → MergeAxis
  相邻 R 轴合并: axes=[1,2] on [2,100,4] → R=400 → AR (2维)
  不相邻 R 轴不合并: axes=[1,3] on [2,3,4,5] → ARAR (4维)
  ≥5 维统一 pad 到 8-9 维 ARARARAR/ARARARARA

方案A（BlockCutA）：外层 A 分核，每核完整处理 R（无核间同步）
方案B（Group Reduce）：R 也跨核分，Phase1 局部归约→workspace，Phase2 合并
  workspace = coreNum × CeilAlign(outSize × maxBytes, cacheLineSize)
非连续搬运：A5 用 NDDMA；A2/A3 用 DataCopyPad + stride 参数或逐 slice 循环搬运
```

> **多输出归约**（sum+sqSum / mean+var / value+index）的 Buffer 方程和 NCHW 搬运模式见 `references/patterns.md#多输出归约的-buffer-规划`

### 场景 4：Norm 类（Row-Reduce + 广播变换）

```
多核切分：按行均分，blockFactor = CeilDiv(numRow, coreNum)
UB 切分 4 档：MergeN(≤2000列) / Normal(整行装入) / SplitD(列切片) / SingleRow
两阶段流水：Phase1 求 rstd（V-S 同步）→ Phase2 乘 gamma/beta
```

### 场景 5：全局归约

```
每核处理 total/coreNum 元素 → partial → workspace[blockIdx × SLOT_STRIDE]
SyncAll() → 合并（Atomic 或显式 Phase2）
```

详细切分参数和代码模板见 `references/patterns.md`。

---

## 第六步：性能优化清单

详细代码示例见 `references/optimizations.md`。

### 必选项
- [ ] **精度提升**：FP16/BF16 → FP32 归约，BF16 回退用 CAST_RINT
- [ ] **双缓冲流水**：`BUFFER_NUM=2`，CopyIn(i+1) 与 Compute(i) 重叠
- [ ] **尾块处理**：最后一核/最后一 tile 的余数元素单独处理
- [ ] **对齐到硬件边界**：32B Block / 256B CacheLine / VReg 大小

### 场景相关
- [ ] **V-S 同步**：归约→标量计算时加 `SetFlag<V_S>/WaitFlag<V_S>` 成对使用
- [ ] **Atomic 写保护**：多核写同一 GM 地址用 `SetAtomicAdd`；Arch35 改用 `DataCopyCustom`
- [ ] **非连续搬运**：ARA 高维非连续模式，A5 用 NDDMA；A2/A3 用 DataCopyPad stride 或逐 slice 循环
- [ ] **Welford 分组**：rCount > 8 时每 8 块中间 Finalize 防大数吃小数
- [ ] **Recompute**：Softmax R 远超 UB 时不存 exp 中间结果，三遍扫描

### 高级优化
- [ ] **二分累加**：Sum 归约专用，相近量级先加，解决大数吃小数
- [ ] **CacheLine 对齐**：从最内维向外找 256B 边界轴，优化 DMA 传输
- [ ] **多策略择优**：按 shape/dtype/UB 用 TilingKey 编译期多态分发
- [ ] **空核跳过**：`if (GetBlockIdx() >= realCoreNum) return;`

---

## 常见致命错误

> 以下每一条都曾导致实际算子设计翻车，设计时逐条自查。

### 1. MergeAxis 不能跨不同类型合并

```
❌ 错误: [N,C,H,W] axes=[0,2,3] → "R[N]和R[H×W]合并为R[N×H×W]" → RA
✅ 正确: R[N] 和 R[H×W] 被 A[C] 隔开，不相邻，不合并 → R[N],A[C],R[H×W] → ARAR

规则: MergeAxis 遇到 A↔R 类型切换时立即停止，只合并相邻同类型轴
```

### 2. ReduceSum/ReduceMax 是覆盖写，不是累加

```
❌ 错误: 循环多次调用 ReduceSum(dst, src, count)，以为结果会自动累加
✅ 正确: 每次调用覆盖 dst，多 tile 累加须自己做:
   ReduceSum(partial, src, work, count);   // 归约到 partial
   old = accBuf.GetValue(idx);
   accBuf.SetValue(idx, old + partial.GetValue(0));  // 显式累加
```

### 3. Cast 方向与 RoundMode 对应关系

```
❌ 错误: "BF16→FP32 用 CAST_RINT"（方向搞反）
✅ 正确:
   提升方向 (低精度→FP32): 一律 CAST_NONE
     FP16 → FP32: CAST_NONE
     BF16 → FP32: CAST_NONE
   回退方向 (FP32→低精度): 看目标类型
     FP32 → FP16: CAST_NONE
     FP32 → BF16: CAST_RINT（四舍五入到最近偶数）
```

### 4. NCHW 布局中同一通道跨 batch 不连续

```
❌ 错误: gmAddr = c × N × H × W + rOffset  （假设 c 的所有空间元素连续）
✅ 正确: NCHW 中 x[n,c,h,w] = n×C×H×W + c×H×W + h×W + w
   同一 (n,c) 的 H×W 连续，但跨 n 有 (C-1)×H×W 间隔
   正确地址: gmAddr = n×C×H×W + c×H×W + spatialOffset
```

### 5. UB 方程遗漏 buffer 导致越界

```
❌ 错误: 只算 inBuf 双缓冲，忘了 castBuf/tmpBuf/accumBuf/workBuf
✅ 正确: 列出所有 buffer 再求 tileSize（参考多输出 Buffer 方程）
   tileSize = (UB_SIZE - fixedBuf) / perTileBuf
   fixedBuf = 所有不随 tile 变化的 buffer（累加器、work、output）
   perTileBuf = 所有随 tile 线性增长的 buffer（in×2 + cast + tmp）
```

---

## atvoss Reduce 框架速览（仅 A5，Tiling 思路通用）

> 位于 `opbase/pkg_inc/op_common/atvoss/reduce/`，是 A5 基础归约算子的共享框架。

**核心组件**：ReduceSch（Kernel 调度器）+ Tiling4ReduceOp（Host Tiling）+ OpDag（声明式计算图）

**5 步 Tiling 算法**（`ComputeTiling<Pattern>`）：
1. CalcBasicBlock → 根据 DAG buffer 需求 + CACHE_BUF(16KB) 计算 UB 块大小
2. ComputeCacheLineBlock → 从最内维向外找 256B 边界轴
3. ComputeUnitA → 贪心累积 A 轴（约束: innerA×rSize ≤ basicBlock 且核利用率 ≥ 95%）
4. ComputeUnitR → 贪心累积 R 轴（受 basicBlock 剩余空间约束）
5. ComputeProgressUnitA → R 全载时用剩余 UB 扩大 A（可选）

**ReduceSch 调度**：IsEmpty→ProcessEmpty / IsTensorMove→ProcessMove / groupR==1→ProcessNormal / groupR>1→ProcessGroup

**TilingKey 编码**：`GEN_REDUCE_TILING_KEY(isContiguous, patternID, loopARCount=A{a}R{r}, loopInnerARCount)`

详细框架分析见 `references/atvoss-framework.md`。

---

## 参考文件索引

| 文件 | 内容 | 何时读取 |
|------|------|---------|
| `references/patterns.md` | 5 大场景切分参数、多输出 Buffer 方程、多轴变换详解 | 写 Tiling 代码时 |
| `references/algorithms.md` | TwoPass/Welford/GroupReduce/Recompute/二分累加/ArgMax合并 | 写 kernel 计算逻辑时 |
| `references/optimizations.md` | 9 项性能优化的完整代码示例和注意事项 | 性能调优或精度排查时 |
| `references/atvoss-framework.md` | atvoss 共享框架分析（ReduceSch/Tiling/DAG，仅 A5） | 理解 A5 框架时 |
| `references/ops-codebase-index.md` | Reduction 算子路径索引 | 查阅参考实现时 |
