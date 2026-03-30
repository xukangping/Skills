# atvoss Reduce 共享框架完整分析

> 源码位置: `opbase/pkg_inc/op_common/atvoss/reduce/` (头文件) + `opbase/src/op_common/atvoss/reduce/` (实现)
> 适用平台: A5 (Ascend950/arch35) — DAG 模式专用，但 Tiling 算法思路对 A2/A3 命令式实现有参考价值

---

## 目录

- [1. 框架文件清单](#1-框架文件清单)
- [2. 核心数据结构](#2-核心数据结构)
- [3. Pattern 检测与 Shape 变换](#3-pattern-检测与-shape-变换)
- [4. 5 步 Tiling 算法详解](#4-5-步-tiling-算法详解)
- [5. ReduceSch 调度器](#5-reducesch-调度器)
- [6. Reduce + Elewise DAG 组合](#6-reduce--elewise-dag-组合)
- [7. 二分归约（Dichotomy）](#7-二分归约dichotomy)
- [8. 非连续内存处理](#8-非连续内存处理)
- [9. Tiling Key 编码体系](#9-tiling-key-编码体系)
- [10. 归约操作类型](#10-归约操作类型)

---

## 1. 框架文件清单

```
pkg_inc/op_common/atvoss/reduce/
├── reduce_sch.h                        ← ReduceSch 主调度器模板类
├── reduce_sch_aux_base.h               ← CRTP 基类（循环控制/PreReduce/PostReduce）
├── reduce_sch_aux.h                    ← 连续模式辅助（CopyIn/CopyOut）
├── reduce_sch_aux_non_contiguous.h     ← 非连续模式辅助
├── reduce_sch_aux_util.h              ← SFINAE 检测/类型萃取/辅助工具
├── reduce_operator.h                   ← 归约操作定义（Sum/Max/Min/Prod/All/Any）
├── reduce_tiling.h                     ← Tiling 接口 + ReduceOpTiling 类
├── reduce_tiling_data.h                ← ReduceOpTilingData 结构体
├── reduce_util.h                       ← Pattern 枚举/常量/GetPromoteType/LoopInfo
├── reduce_tiling_key_decl.h            ← 连续模式 TilingKey 声明
├── reduce_tiling_key_sel.h             ← 连续模式 TilingKey 选择器
├── reduce_tiling_key_decl_non_contiguous.h  ← 非连续模式 Key 声明
├── reduce_tiling_key_sel_non_contiguous.h   ← 非连续模式 Key 选择器
├── reduce_tensor_empty.h               ← 空张量处理
└── reduce_tensor_move.h                ← 纯搬运处理

src/op_common/atvoss/reduce/
└── reduce_tiling.cpp                   ← Tiling 完整实现（~1200 行）
```

---

## 2. 核心数据结构

### ReduceOpTilingData（Host→Kernel 传递的切分参数）

```cpp
struct ReduceOpTilingData {
    // === 多核切分参数 ===
    uint64_t factorACntPerCore;      // 每核 A 轴迭代次数
    uint64_t factorATotalCnt;        // A 轴总迭代次数
    uint64_t factorRCntPerCore;      // 每核 R 轴迭代次数
    uint64_t factorRTotalCnt;        // R 轴总迭代次数
    int32_t  coreNum;                // 实际使用核数

    // === UB 切分参数 ===
    uint64_t ubFactorA;              // UB 内 A 轴切片大小
    uint64_t ubFactorR;              // UB 内 R 轴切片大小
    uint64_t basicBlock;             // 输入 UB 缓冲大小（元素数）
    uint64_t resultBlock;            // 输出 UB 缓冲大小（元素数）

    // === Group Reduce ===
    uint64_t groupR;                 // R 轴分组数（>1 触发两阶段归约）
    uint64_t outSize;                // 输出元素数

    // === 数据搬运 ===
    int32_t  useNddma;              // 是否用多维 DMA
    float    meanVar;               // 方差/均值因子（如 1/N）

    // === Shape 信息 ===
    uint64_t shape[9];              // 各维度大小
    int64_t  stride[9];             // 各维度步进
    int64_t  dstStride[9];          // 输出步进

    // === 非连续模式 ===
    uint64_t sliceNum[9];           // 每轴 slice 个数
    uint64_t sliceShape[9];         // 每轴 slice 形状
    uint64_t sliceStride[9];        // 每轴 slice 步长
};
```

### ReduceSchLoopInfo（Kernel 内部循环控制）

```cpp
struct ReduceSchLoopInfo {
    int32_t patternID;                    // Pattern 标识
    int32_t reduceDichotomy;              // 是否启用二分归约
    int32_t loopACount;                   // 外层 A 轴循环次数
    int32_t loopAAxis[5];                 // A 轴维度索引
    int32_t loopRCount;                   // 外层 R 轴循环次数（>0 则两阶段）
    int32_t loopRAxis[9];                 // R 轴维度索引
    int32_t loopInnerACount;              // 内层 A 轴循环次数
    int32_t loopInnerAAxis[9];            // 内层 A 轴索引
    int32_t loopInnerRCount;              // 内层 R 轴循环次数
    int32_t loopInnerRAxis[9];            // 内层 R 轴索引
    int32_t innerPatternID;               // 内层 Pattern
};
```

### ReduceTilingKey（Tiling 键结构）

```cpp
struct ReduceTilingKey {
    uint32_t patternID;             // 模式 ID
    uint32_t loopARCount;           // 外层循环编码 (A{a}R{r} = a×10+r)
    uint32_t loopInnerARCount;      // 内层循环编码
    bool     isContiguous;          // 是否连续内存
};
```

---

## 3. Pattern 检测与 Shape 变换

### 三步 Shape 变换

输入的 N 维 shape + axes 经过三步变换压缩为最简 A/R 交替序列：

```
Step 1: EliminateOne()
  移除 size=1 且连续的维度
  例: [1, 64, 1, 128] axes=[1] → [64, 128] axes=[0]

Step 2: PadDimForNonContiguous()
  在非连续轴后插入 size=1 的填充维度
  标记填充维度为 R 轴（确保非连续访问被正确切分）

Step 3: MergeAxis()
  合并相邻同类型（同为 A 或同为 R）的连续轴
  输出: viewShape[] / sliceNum[] / sliceShape[] / sliceStride[]
```

### Pattern 匹配

变换后的维度数直接映射到 Pattern：

| 变换后维度数 | Pattern | ID | 含义 |
|------------|---------|-----|------|
| 1 | A | 10 | 仅保留轴（纯搬运） |
| 2 | AR | 1 | 标准二维归约 |
| 3 | ARA | 2 | 三维交替 |
| 4 | ARAR | 3 | 四维交替 |
| 5 | ARARA | 4 | 五维交替 |
| 6-9 | ARARAR...A | 5-8 | 填充 dummy=1 维至 8-9 维 |

---

## 4. 5 步 Tiling 算法详解

### Step 1: CalcBasicBlock — 计算 UB 块大小

```
ubAvailSize = ubSize - CACHE_BUF_SIZE(16KB) - reservedSize

对于 Pattern A（无归约）:
  basicBlock = ubAvailSize / (PRE_MTE2×2 + (PRE_MTE3+PRE_TEMP) × ratio)

对于其他 Pattern:
  resultBlock = TryGetReduceBlock()
    = CeilAlign(cBlock_.aSize × maxBytes, cacheLineSize)
    范围: [MAX_INNER_A, CACHE_BUF_SIZE]

  basicBlock = (ubAvailSize - resultBlock × POST_BUF_NUM)
             / (PRE_MTE2×2 + (PRE_MTE3+PRE_TEMP) × ratio)

  FloorAlign(basicBlock, vRegSize)

常量:
  CACHE_BUF_SIZE  = 16 × 1024     (16KB 二分归约缓存)
  BASIC_BLOCK     = 64 × 1024     (64KB 默认值)
  POST_BUF_SIZE   = 8 × 1024      (8KB 后处理缓冲)
```

### Step 2: ComputeCacheLineBlock — CacheLine 对齐切点

```
从最内维向外扫描:
  cumSize = 1
  for axis = Dim-1 downto 0:
    cumSize *= shape[axis]
    if cumSize > cacheLineSize / sizeof(T):
      → 这个轴是 CacheLine 切点
      cacheLineStep = CeilDiv(cacheLineSize/sizeof(T), 之前累积)
      对齐到 sliceShape[axis]
      cacheLineOuter = CeilDiv(shape[axis], cacheLineStep)
      break

输出: cBlock_ = {axis, cacheLineStep, cacheLineOuter, aSize, rSize}
  aSize = CacheLine 切点之后所有 A 维的乘积
  rSize = CacheLine 切点之后所有 R 维的乘积
```

### Step 3: ComputeUnitA — A 轴 UB 切片（贪心累积）

```
从 CacheLine 切点向外扫描每个 A 轴:
  for each A-axis (from inner to outer):
    for step = 1 to axisLen:
      tmpInnerA = innerA × step
      aSize = tmpInnerA × cBlock_.aSize

      检查约束:
        (1) aSize ≤ maxInnerA
        (2) aSize × cBlock_.rSize ≤ basicBlock
        (3) 核利用率 ≥ 95% (THRESHOLD)

      如果全部满足: 记录最佳 step
      否则: break

    innerA *= bestStep

输出: unitA_ = {idx, inner, outer, step}
```

### Step 4: ComputeUnitR — R 轴 UB 切片（贪心累积）

```
从 CacheLine 切点向外扫描每个 R 轴:
  剩余空间 = basicBlock / (innerA × innerR × cBlock_.aSize × cBlock_.rSize)

  for each R-axis:
    step = min(剩余空间, axisLen)
    对齐到 sliceShape 和 ubBlockSize

    检查核利用率约束
    innerR *= step

输出: unitR_ = {idx, inner, outer, step}
```

### Step 5: ComputeProgressUnitA — 可选优化

```
条件: unitR_.idx == -1（R 轴完全装入 UB，仍有空间）
  → 用剩余 UB 扩大 unitA
  → 上限: resultBlock（输出缓冲大小）
  → 目的: 减少 A 轴循环次数
```

### 最终参数设置（SetTilingData）

```cpp
numBlocks = min(coreNum, unitA_.outer × unitR_.outer);
factorACntPerCore = CeilDiv(unitA_.outer, numBlocks);
factorRCntPerCore = CeilDiv(unitR_.outer, numBlocks / factorACntPerCore);
groupR = CeilDiv(unitR_.outer, factorRCntPerCore);

ubFactorA = unitA_.step;
ubFactorR = unitR_.step;

if (groupR > 1) {
    SetScheduleMode(1);  // 启用两阶段 Group Reduce
    workspaceSize = coreNum × CeilAlign(outSize × maxBytes, cacheLineSize);
}
```

---

## 5. ReduceSch 调度器

### 模板参数

```cpp
template <bool isContiguous, uint32_t PatternID, uint32_t LoopARCount,
          uint32_t LoopInnerARCount, class OpDag>
class ReduceSch
```

### 调度逻辑

```
Init(pipe, input, output, workspace):
  1. 解析 TilingData
  2. 分配 preBufPool / postBufPool / resBuf / cache
  3. 计算循环范围 SetLoopRange()

Process(dumpValue):
  分支选择:
  ├─ IsEmpty()      → ProcessEmpty()      // 用 dumpValue 填充输出
  ├─ IsTensorMove() → ProcessMove()       // 纯数据搬运
  ├─ groupR == 1    → ProcessNormal()     // 单阶段归约
  └─ groupR > 1     → ProcessGroup()      // 两阶段 Group Reduce

ProcessNormal():
  for aLoop in [aStart, aEnd):        // A 轴外循环（分核）
    for rLoop in [0, rTotal):          // R 轴循环
      CopyInAux(tile)                  // GM → UB（含 NDDMA 选择）
      PreReduce(tile)                  // Pre-Reduce Elewise
      ReduceComputeMerge(tile)         // 核心归约 + 二分缓存
      PostReduce(tile)                 // Post-Reduce Elewise
      CopyOutAux(tile)                // UB → GM

ProcessGroup():
  Phase1: 同 ProcessNormal，但 partial 写入 workspace
  SyncAll()
  Phase2: 读 workspace 所有 partial，合并为最终结果
```

### Buffer 分配策略

```
UB 总布局:
┌─────────────────────────────────────────┐
│ preBufPool (Pre-Reduce Elewise)         │
│   = basicBlock × PRE_MTE2_NUM(=2)      │  双缓冲输入
│   + basicBlock × ratio × PRE_REST_NUM  │  Cast/临时
├─────────────────────────────────────────┤
│ postBufPool (Post-Reduce Elewise)       │
│   = resultBlock × POST_BUF_NUM         │  Post 处理
├─────────────────────────────────────────┤
│ resBuf (归约中间结果)                    │
│   = resultBlock                         │
├─────────────────────────────────────────┤
│ cache (二分归约缓存)                     │
│   = CACHE_BUF_SIZE (16KB)              │
└─────────────────────────────────────────┘
```

---

## 6. Reduce + Elewise DAG 组合

DAG 模板将 Elewise 操作和 Reduce 操作声明式组合：

```cpp
// reduce_mean 的 DAG 示例
struct ReduceMeanDag {
    // Pre-Reduce: CopyIn → Cast(FP16→FP32)
    using OpCopyIn0 = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
    using Cast0     = Bind<Vec::Cast<PromoteT, T, 0>, OpCopyIn0>;

    // Reduce 核心
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromoteT>, Cast0>;

    // Post-Reduce: Muls(×1/N) → Cast(FP32→FP16) → CopyOut
    using Mul0      = Bind<Vec::Muls<PromoteT>, ReduceOp0, Placeholder::Var<PromoteT, 0>>;
    using Cast1     = Bind<Vec::Cast<T, PromoteT, 1>, Mul0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg  = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag   = DAGSch<Outputs, void, MemCfg>;
};
```

框架自动从 DAG 中提取：
- **PreReduceNodeInfo**: CopyIn + Cast (在 ReduceOp 之前的所有节点)
- **ReduceOpPos**: ReduceSumOp 的位置
- **PostReduceNodeInfo**: Muls + Cast + CopyOut (在 ReduceOp 之后的所有节点)
- **Buffer 需求**: PRE_MTE2_NUM / PRE_MTE3_NUM / PRE_TEMP_CALC_NUM / POST_* 计数

---

## 7. 二分累加（Dichotomy）— Sum 归约专用

ReduceSch 内建二分累加机制，**专用于 Sum 归约**，解决顺序累加中**大数吃小数**的精度问题：
当 sum 已经很大而后续元素很小时，小数因浮点精度被截断丢失。二分累加使相近量级的数先相加，避免大小悬殊的数直接累加。

```
触发条件: reduceDichotomy = 1 (由 Tiling Key 的 LoopARCount 控制)

参数计算:
  bisectionPos = FindNearestPower2(rCount)   // 最大 2^k ≤ rCount
  cacheCount = log2(bisectionPos) + 1        // 缓存层数
  bisectionTail = rCount - bisectionPos      // 尾部元素数

执行流程 (LinearComputeR):
  for rLoop in [0, rCount):
    PreReduce(tile)
    ReduceComputeMerge(tile)         // 归约当前 tile
    DoCaching(rLoop)                 // 二分缓存更新
      → cache[level] = Merge(cache[level], currentResult)
      → 层级由 rLoop 的二进制表示决定

最终: cache[0] 包含完整归约结果（类似二叉树从叶到根合并）
```

---

## 8. 非连续内存处理

当输入 stride 不连续时（如转置后的张量），框架切换到非连续模式：

```
检测: CheckIsContiguous()
  → stride[i] ≠ stride[i+1] × shape[i+1] 则非连续

Shape 变换: PadDimForNonContiguous()
  → 在非连续轴后插入 size=1 填充维度
  → 构建 sliceNum/sliceShape/sliceStride 追踪子结构

Kernel 侧:
  ReduceSchAuxNonContiguous (替代 ReduceSchAux)
  → ComputeIterAddress() 用 sliceStart/sliceStride 计算真实地址
  → CopyInWithMoveAlignNonContiguous() / CopyInWithNddmaNonContiguous()

TilingKey 选择:
  reduce_tiling_key_decl_non_contiguous.h 定义非连续模式的 key
  reduce_tiling_key_sel_non_contiguous.h 提供对应选择器
```

---

## 9. Tiling Key 编码体系

### LoopARCount 编码

```
A{a}R{r} = a × 10 + r

A1R0 = 10  → 只有 1 层 A 外循环，无 R 外循环
A1R2 = 12  → 1 层 A 外循环 + 2 层 R 外循环
A2R0 = 20  → 2 层 A 外循环
A2R3 = 23  → 2 层 A + 3 层 R
A3R0 = 30  → 3 层 A
...
```

### Pattern 与 Kernel 类型映射

| Pattern | LoopARCount | Inner | Kernel 类型 | 说明 |
|---------|-------------|-------|------------|------|
| AR_NORMAL | 10 | 0 或 1 | AIV_ONLY | 单阶段 AR |
| AR_GROUP | 12 或 13 | 0 | MIX_AIV_1_0 | 两阶段 AR + Group |
| ARA_NORMAL | 10 或 20 | 0 或 1 | AIV_ONLY | 单阶段 ARA |
| ARA_GROUP | 12/13/23 | 0 | MIX_AIV_1_0 | 两阶段 ARA + Group |
| ARAR_NORMAL | 10 或 20 | 0/1/2 | AIV_ONLY | 单阶段 ARAR |
| ARAR_GROUP | 12/13/23/24 | 0 | MIX_AIV_1_0 | 两阶段 ARAR + Group |

### GEN_REDUCE_TILING_KEY 宏

```cpp
#define GEN_REDUCE_TILING_KEY(result, key, ...)  \
    result = GET_TPL_TILING_KEY(                 \
        key.isContiguous,                        \
        key.patternID,                           \
        key.loopARCount,                         \
        key.loopInnerARCount,                    \
        __VA_ARGS__)                             \
```

---

## 10. 归约操作类型

### 定义（reduce_operator.h）

| 类 | 归约操作 | 初始值(Padding) | 对应 AscendC API |
|---|---------|----------------|-----------------|
| ReduceSumOp | 加法归约 | 0 | AscendC::ReduceSum |
| ReduceMaxOp | 最大值归约 | 类型最小值 | AscendC::ReduceMax |
| ReduceMinOp | 最小值归约 | 类型最大值 | AscendC::ReduceMin |
| ReduceProdOp | 乘法归约 | 1 | 手工 fold-multiply |
| ReduceAllOp | 逻辑 AND | 1 (true) | 继承 ReduceMinOp |
| ReduceAnyOp | 逻辑 OR | 0 (false) | 继承 ReduceMaxOp |

### DumpValue（Padding 初始值表）

| 数据类型 | Sum | Max | Min | Prod |
|---------|-----|-----|-----|------|
| float16 | 0 | HALF_MIN_VALUE | HALF_MAX_VALUE | 1 |
| float32 | 0 | FLOAT_MIN_VALUE | FLOAT_MAX_VALUE | 1 |
| bfloat16 | 0 | BFLOAT16_MIN_VALUE | BFLOAT16_MAX_VALUE | 1 |
| int32 | 0 | INT32_MIN | INT32_MAX | 1 |
| int64 | 0 | INT64_MIN | INT64_MAX | 1 |
| int8 | 0 | INT8_MIN | INT8_MAX | 1 |
| uint8 | 0 | UINT8_MIN | UINT8_MAX | 1 |

这些值用于 padding 区域填充，确保 pad 后的归约结果不被 padding 元素影响。
