# S3: 多轴归约 Shape 变换与高维处理

> 本文档描述任意 N 维 shape + 任意 axes 组合如何变换为 A/R 交替序列，以及高维 Pattern 的统一处理方式。
> 场景策略选择见 [patterns.md S3](patterns.md#s3-araarara-多轴模式)

---

## 多轴归约的 Shape 三步变换

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

## 实际算子示例

| 算子 | Shape | Axes | 变换过程 | 最终 Pattern |
|------|-------|------|---------|-------------|
| reduce_max | [2,100,4] | [1,2] | R 轴相邻合并: R=100×4=400 → [2, 400] | **AR** (2维) |
| reduce_sum | [2,3,4,5] | [1,3] | 不相邻不合并: A[2],R[3],A[4],R[5] | **ARAR** (4维) |
| reduce_sum | [2,3,4,5] | [0,2] | 不相邻不合并: R[2],A[3],R[4],A[5] → pad→9维 | **ARARARARA** (9维) |
| reduce_sum | [2048,2,48,2,2,2] | [1,3,5] | 交替不合并: A,R,A,R,A,R → pad→8维 | **ARARARAR** (8维) |
| reduce_log_sum_exp | [1,2,3,4,5] | [2,4] | 消1→合并A[0,1]→ A[2],R[3],A[4],R[5] → pad | **ARARARARA** (9维) |
| **bn_training_reduce** | [N,C,H,W] | [0,2,3] | R[N],A[C],R[H×W]（相邻R合并）→ 前置1 → ARAR | **ARAR** (4维) |

## 典型场景：BatchNorm "保留 C、归约 N/H/W"

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

## 具体变换 trace

**axes=[1,3] on [2,3,4,5]**：

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

**axes=[1,2] on [2,100,4]（相邻合并）**：

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

---

## 高维 Pattern（≥5 维）的统一处理

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

---

## 非连续多轴的数据搬运

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
