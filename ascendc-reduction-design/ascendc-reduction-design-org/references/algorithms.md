# Reduction 类算子核心算法实现

> 包含 TwoPass / Welford Online / Group Reduce / Recompute / 二分加法树 的详细算法模板

---

## 目录

- [1. TwoPass 算法（两遍扫描）](#1-twopass-算法两遍扫描)
- [2. Welford Online 算法（在线单遍）](#2-welford-online-算法在线单遍)
- [3. Group Reduce（跨核归约）](#3-group-reduce跨核归约)
- [4. Recompute 策略（重计算换存储）](#4-recompute-策略重计算换存储)
- [5. 二分加法树（Dichotomy Addition）](#5-二分加法树dichotomy-addition)
- [6. Half-Interval 归约](#6-half-interval-归约)
- [7. 算法选择对照表](#7-算法选择对照表)

---

## 1. TwoPass 算法（两遍扫描）

**适用场景**: R 完全装入 UB，需要计算 mean+var（reduce_var/std/layer_norm）

**原理**: 第一遍求 mean，第二遍用 mean 求 var，数值稳定性依赖 mean 准确度。

### 1.1 基础 TwoPass（AR 模式，R ≤ VL）

```cpp
// R 很小（≤ 向量长度），可直接 ReduceSum
void TwoPassSmallR(LocalTensor<float>& x, int R, float correction) {
    // Pass 1: mean = sum(x) / R
    float sum;
    ReduceSum(tmpBuf, x, workBuf, R);
    PipeBarrier<PIPE_V>();
    sum = tmpBuf.GetValue(0);
    float mean = sum / static_cast<float>(R);

    // Pass 2: var = sum((x - mean)²) / (R - correction)
    Adds(tmpBuf, x, -mean, R);              // tmp = x - mean
    Mul(tmpBuf, tmpBuf, tmpBuf, R);          // tmp = (x - mean)²
    ReduceSum(varBuf, tmpBuf, workBuf, R);
    PipeBarrier<PIPE_V>();
    float varSum = varBuf.GetValue(0);
    float var = varSum / static_cast<float>(R - correction);
}
```

### 1.2 TwoPass + 二分加法（AR 模式，R > VL）

```cpp
// R 较大但完全装入 UB，用二分折叠求 ReduceSum
void TwoPassLargeR(LocalTensor<float>& x, int R, float correction) {
    // Pass 1: 二分加法求 sum
    float sum = DichotomyReduceSum(x, R);   // 见第 5 节
    float mean = sum / R;

    // Pass 2: var
    Adds(tmpBuf, x, -mean, R);
    Mul(tmpBuf, tmpBuf, tmpBuf, R);
    float varSum = DichotomyReduceSum(tmpBuf, R);
    float var = varSum / (R - correction);
}
```

### 1.3 TwoPass（RA 模式）

```cpp
// RA 模式: R 在外，A 在内（连续）
void TwoPassRA(int R, int A) {
    // Pass 1: 逐 R 行累加到 sumBuf（A 长度）
    Duplicate(sumBuf, 0.0f, A_aligned);
    for (int r = 0; r < R; r++) {
        CopyIn(xLocal, xGm[r * A], A);
        Add(sumBuf, sumBuf, xLocal, A);
    }
    // sumBuf[a] = Σ_r x[r,a]
    Muls(meanBuf, sumBuf, 1.0f / R, A);    // mean = sum / R

    // Pass 2: 逐 R 行求方差
    Duplicate(varBuf, 0.0f, A_aligned);
    for (int r = 0; r < R; r++) {
        CopyIn(xLocal, xGm[r * A], A);
        Sub(diffBuf, xLocal, meanBuf, A);   // diff = x - mean
        Mul(diffBuf, diffBuf, diffBuf, A);  // diff²
        Add(varBuf, varBuf, diffBuf, A);    // 累加
    }
    Muls(varBuf, varBuf, 1.0f / (R - correction), A);
}
```

---

## 2. Welford Online 算法（在线单遍）

**适用场景**: R 不能完全装入 UB，需要流式遍历（reduce_var/std 的核心算法）

**优势**: 单遍扫描、数值稳定性好、支持 Group 化并行合并

### 2.1 核心更新公式

```
初始: mean = 0, M2 = 0, count = 0

对每个新元素 x:
    count += 1
    delta1 = x - mean           ← 旧偏差
    mean = mean + delta1/count  ← 增量更新均值
    delta3 = x - mean           ← 新偏差
    M2 = M2 + delta1 * delta3   ← 增量更新方差

最终: var = M2 / (count - correction)
```

### 2.2 向量化 Welford（处理 UB 片段）

```cpp
// 每个 UB 片段包含 cutRSize 个元素
void WelfordUpdate(LocalTensor<float>& x, int curLen,
                   LocalTensor<float>& mean, LocalTensor<float>& M2,
                   int& count) {
    for (int i = 0; i < curLen; i++) {
        count++;
        float scale = 1.0f / static_cast<float>(count);

        // delta1 = x - mean (向量化: 对整个 A 维)
        Sub(delta1Buf, x[i * A_aligned], mean, A_aligned);

        // mean = mean + delta1 * scale
        Muls(tmpBuf, delta1Buf, scale, A_aligned);
        Add(mean, mean, tmpBuf, A_aligned);

        // delta3 = x - mean_new
        Sub(delta3Buf, x[i * A_aligned], mean, A_aligned);

        // M2 = M2 + delta1 * delta3
        Mul(tmpBuf, delta1Buf, delta3Buf, A_aligned);
        Add(M2, M2, tmpBuf, A_aligned);
    }
}
```

### 2.3 Welford 初始化迭代（第一个片段）

```cpp
// 第一个元素直接赋值，避免除零
void WelfordInit(LocalTensor<float>& x, int curLen,
                 LocalTensor<float>& mean, LocalTensor<float>& M2,
                 int& count) {
    // mean = x[0], M2 = 0, count = 1
    DataCopy(mean, x, A_aligned);
    Duplicate(M2, 0.0f, A_aligned);
    count = 1;

    // 从 x[1] 开始正常更新
    WelfordUpdate(x[A_aligned], curLen - 1, mean, M2, count);
}
```

### 2.4 Group Welford（分组合并）

当 rLoopCount 很大时，每 WELFORD_GROUP_NUM(=8) 个片段做一次中间 Finalize，防止浮点误差累积。

```cpp
const int WELFORD_GROUP_NUM = 8;

void ProcessWithGroupWelford(int totalRLoops) {
    int groupCount = 0;
    // 9 个 group 缓存: [0..7] 为活跃组, [8] 为合并结果
    float groupMean[9][A_aligned], groupM2[9][A_aligned], groupCount[9];

    for (int rLoop = 0; rLoop < totalRLoops; rLoop++) {
        CopyIn(xLocal, offset, curLen);

        // 正常 Welford 更新到当前组
        WelfordUpdate(xLocal, curLen, groupMean[groupCount], groupM2[groupCount], ...);
        groupCount++;

        if (groupCount == WELFORD_GROUP_NUM) {
            // 合并 8 个组到 groupMean[8]
            MergeWelfordGroups(groupMean, groupM2, groupCount, 8);
            // 重置活跃组
            groupMean[0] = groupMean[8];
            groupM2[0] = groupM2[8];
            groupCount = 1;
        }
    }

    // 合并剩余的 groupCount 个组
    if (groupCount > 1) {
        MergeWelfordGroups(groupMean, groupM2, groupCount, groupCount);
    }
}
```

### 2.5 两组 Welford 合并公式

```
合并 (mean_a, M2_a, count_a) 和 (mean_b, M2_b, count_b):

count_total = count_a + count_b
delta = mean_b - mean_a
mean_total = mean_a + delta * count_b / count_total
M2_total = M2_a + M2_b + delta² * count_a * count_b / count_total
```

---

## 3. Group Reduce（跨核归约）

**适用场景**: R 太大，单核无法遍历完；同时 A 轴太小不能充分利用多核

### 3.1 两阶段执行模型

```
Phase 1（各核独立）:
  ┌──────┐  ┌──────┐  ┌──────┐
  │Core 0│  │Core 1│  │Core 2│
  │R[0:K]│  │R[K:2K]│ │R[2K:N]│
  └──┬───┘  └──┬───┘  └──┬───┘
     │          │          │
     ↓          ↓          ↓
  workspace[0] workspace[1] workspace[2]
     ↓          ↓          ↓
  ┌────────────────────────────┐
  │        SyncAll()           │
  └────────────────────────────┘
     ↓
Phase 2（合并核）:
  read workspace[0..coreNum]
  merge all partials → final output
```

### 3.2 Phase 1 实现模板

```cpp
void GroupReducePhase1() {
    int myRStart = rGroupIdx * rPerGroup;
    int myREnd = min(myRStart + rPerGroup, totalR);

    // 初始化 partial
    Duplicate(partialBuf, initValue, outSize);  // 0 for sum, -inf for max

    for (int r = myRStart; r < myREnd; r += cutRSize) {
        int curR = min(cutRSize, myREnd - r);
        CopyIn(xLocal, r, curR);
        // 局部归约
        ReduceOp(partialBuf, partialBuf, xLocal, curR);
    }

    // 写 partial 到 workspace
    int wsOffset = blockIdx * SLOT_STRIDE;  // 64B 对齐防 bank conflict
    DataCopy(workspaceGm[wsOffset], partialBuf, outSize);
}
```

### 3.3 Phase 2 实现模板

```cpp
void GroupReducePhase2() {
    SyncAll();  // 等所有核完成 Phase1

    // 合并所有 partial
    Duplicate(finalBuf, initValue, outSize);

    for (int g = 0; g < groupR; g++) {
        int wsOffset = (aBlockIdx * groupR + g) * SLOT_STRIDE;
        CopyIn(partialLocal, workspaceGm[wsOffset], outSize);
        ReduceOp(finalBuf, finalBuf, partialLocal, outSize);
    }

    // 写出最终结果
    CopyOut(yGm[myAStart], finalBuf, outSize);
}
```

### 3.4 Welford Group Reduce（统计归约专用）

对于 reduce_var，Phase 1 输出的是 (partial_mean, partial_M2, partial_count) 三元组，
Phase 2 用 Welford 合并公式合并：

```cpp
void WelfordGroupReducePhase2() {
    SyncAll();

    // 读第一组作为初始值
    float totalMean = workspace_mean[0];
    float totalM2 = workspace_M2[0];
    int totalCount = workspace_count[0];

    // 逐组合并
    for (int g = 1; g < groupR; g++) {
        float gMean = workspace_mean[g];
        float gM2 = workspace_M2[g];
        int gCount = workspace_count[g];

        float delta = gMean - totalMean;
        int newCount = totalCount + gCount;
        totalMean += delta * gCount / newCount;
        totalM2 += gM2 + delta * delta * totalCount * gCount / newCount;
        totalCount = newCount;
    }

    float var = totalM2 / (totalCount - correction);
}
```

---

## 4. Recompute 策略（重计算换存储）

**适用场景**: Softmax 中 R 远超 UB，不存储中间 exp 结果

### 4.1 Softmax Recompute 三遍扫描

```
Pass 1: 求 max（只需 O(1) 额外空间）
  max_val = -inf
  for each R chunk:
    CopyIn(x_chunk)
    max_val = max(max_val, ReduceMax(x_chunk))

Pass 2: 求 sum = Σexp(x-max)（流式累加，O(VL) 空间）
  sum_val = 0
  for each R chunk:
    CopyIn(x_chunk)
    exp_chunk = Exp(x_chunk - max_val)
    sum_val += ReduceSum(exp_chunk)

Pass 3: 输出 = exp(x-max)/sum（重新计算 exp，O(VL) 空间）
  for each R chunk:
    CopyIn(x_chunk)
    exp_chunk = Exp(x_chunk - max_val)
    y_chunk = exp_chunk / sum_val
    CopyOut(y_chunk)
```

### 4.2 实现模板

```cpp
void SoftmaxRecompute(int R, int ubFactor) {
    int rLoops = CeilDiv(R, ubFactor);

    // Pass 1: Max
    float maxVal = -3.4028235e+38f;
    for (int i = 0; i < rLoops; i++) {
        int curLen = min(ubFactor, R - i * ubFactor);
        DataCopy(xLocal, xGm[i * ubFactor], curLen);
        ReduceMax(tmpLocal, xLocal, workLocal, curLen);
        PipeBarrier<PIPE_V>();
        float chunkMax = tmpLocal.GetValue(0);
        maxVal = (chunkMax > maxVal) ? chunkMax : maxVal;
    }

    // Pass 2: Sum of Exp
    float sumVal = 0.0f;
    for (int i = 0; i < rLoops; i++) {
        int curLen = min(ubFactor, R - i * ubFactor);
        DataCopy(xLocal, xGm[i * ubFactor], curLen);
        Adds(xLocal, xLocal, -maxVal, curLen);       // x - max
        Exp(xLocal, xLocal, curLen);                  // exp(x - max)
        ReduceSum(tmpLocal, xLocal, workLocal, curLen);
        PipeBarrier<PIPE_V>();
        sumVal += tmpLocal.GetValue(0);
    }

    float recipSum = 1.0f / sumVal;

    // Pass 3: Output
    for (int i = 0; i < rLoops; i++) {
        int curLen = min(ubFactor, R - i * ubFactor);
        DataCopy(xLocal, xGm[i * ubFactor], curLen);
        Adds(xLocal, xLocal, -maxVal, curLen);
        Exp(xLocal, xLocal, curLen);
        Muls(yLocal, xLocal, recipSum, curLen);       // exp / sum
        DataCopy(yGm[i * ubFactor], yLocal, curLen);
    }
}
```

### 4.3 Binary Fold + Cache 优化（softmax_v2 实际实现）

实际 softmax_v2 的 Pass 2 使用二分折叠 + cache buffer 优化：

```
不是简单的 sum += chunk，而是:
  mainBlock → yMain 缓存
  foldBlock → 二分折叠到 yMain
  使用 cacheBuffer[cacheID] 存储二分中间结果
  cacheID = popcount-based 交替寻址

这样最终 sum 的精度更好（类似二分加法树）
```

---

## 5. 二分累加（Dichotomy Addition）

**适用场景**: Sum 归约（ReduceSum）专用，解决顺序累加中大数吃小数的精度问题

**问题**: 顺序累加 `sum = a1 + a2 + a3 + ...` 时，当 sum 已经很大而后续元素很小，小数会因浮点精度被"吃掉"（大数吃小数）。

**原理**: 用二叉树结构折叠求和，使**相近量级的数先相加**，避免大数与小数直接相加。

### 5.1 核心算法

```cpp
float DichotomyReduceSum(LocalTensor<float>& src, int count) {
    // Step 1: 找到最大的 2^k ≤ count
    int powerTwo = FindNextPower2LessEqual(count);

    // Step 2: 尾部折叠到前 (count - powerTwo) 个位置
    int tail = count - powerTwo;
    if (tail > 0) {
        Add(src, src, src[powerTwo], tail);  // src[0..tail] += src[powerTwo..]
    }

    // Step 3: 二分折叠
    int curCount = powerTwo;
    while (curCount > 64) {  // 64 = 一个 repeat 能处理的 FP32 个数
        curCount /= 2;
        Add(src, src, src[curCount], curCount);
    }

    // Step 4: 最终 WholeReduceSum (≤64 元素)
    float result;
    WholeReduceSum(result, src, curCount);
    return result;
}
```

### 5.2 FindNextPower2

```cpp
int FindNextPower2LessEqual(int n) {
    int p = 1;
    while (p * 2 <= n) p *= 2;
    return p;
}
```

### 5.3 与直接 ReduceSum 的对比

| 方面 | 顺序累加 ReduceSum | 二分累加 |
|------|-------------------|---------|
| 精度问题 | **大数吃小数**：sum 已很大时，小元素被截断丢失 | 相近量级先加，避免大小悬殊的数直接相加 |
| 适用操作 | **仅 Sum 归约** | **仅 Sum 归约**（Max/Min/Prod 不受大数吃小数影响） |
| 时间复杂度 | O(N) | O(N)（同样遍历所有元素） |
| UB 开销 | 无额外 | 原地操作，无额外 buffer |
| 典型场景 | R ≤ VL，元素量级均匀 | R >> VL，元素量级差异大（如 FP16 累加大量小值） |

---

## 6. Half-Interval 归约

**适用场景**: rms_norm / layer_norm 的行内归约（来自 `reduce_common.h`）

### 6.1 算法流程

```cpp
void ReduceSumHalfInterval(LocalTensor<float>& src, int count) {
    // Step 1: 找最大 2^k ≤ count
    int powerTwo = FindNextPower2LessEqual(count);

    // Step 2: 尾部处理
    int tail = count - powerTwo;
    if (tail > 0) {
        // Mask 保护: 只操作 tail 个元素
        uint64_t mask = GenMask(tail);
        Add(src, src, src[powerTwo], mask);
    }

    // Step 3: 半区间折叠（与二分加法树相同）
    while (powerTwo > 64) {
        powerTwo /= 2;
        Add(src, src, src[powerTwo], powerTwo);
    }

    // Step 4: WholeReduceSum 硬件指令
    WholeReduceSum(dst, src, mask_for_final);
}
```

### 6.2 多行版本（MergeN 模式）

```cpp
// 同时对多行做归约（利用 repeat stride）
void ReduceSumMultiN(LocalTensor<float>& src, int numRows,
                     int colsPerRow, int stride) {
    // 配置 repeat 参数使硬件并行处理多行
    uint64_t rptCfg = BuildRepeatConfig(numRows, stride);
    WholeReduceSum(dst, src, rptCfg);
}
```

---

## 7. 算法选择对照表

| 条件 | 推荐算法 | 原因 | 适用算子 |
|------|---------|------|---------|
| R ≤ UB，纯 sum/max/min | 直接 ReduceSum/Max | 最简单高效 | reduce_sum/max/min |
| R ≤ UB，需 mean+var | TwoPass | 两遍内存访问，数值稳定 | reduce_var/std, layer_norm |
| R > UB，需 mean+var | Welford Online | 单遍扫描省 IO，数值稳定 | reduce_var/std |
| R >> UB，大量 R 循环 | Welford + Group(8) | 每 8 块合并防误差累积 | reduce_var/std |
| R >> UB 且 A 小 | Group Reduce | 跨核分 R，workspace 同步 | arg_max, reduce_var |
| Softmax R > UB | Recompute | 不存 exp 中间结果，3 遍扫描 | softmax_v2 |
| Softmax R ≤ UB | FullLoad | 一次载入，exp→sum→div | softmax_v2 |
| 大向量 sum 需高精度 | 二分累加 | Sum 专用，相近量级先加，解决大数吃小数 | reduce_sum/mean TwoPass, reduce_var |
| Norm 行内归约 | Half-Interval | 适配硬件 WholeReduceSum | rms_norm, layer_norm |
| ArgMax/ArgMin R > UB | 分片合并 | CutR + UpdateResult 逐片合并 | arg_max_v2 |

---

## 8. ArgMax/ArgMin 分片合并

**适用场景**: AR 模式下 R > UB，需要分片找最大值+下标（arg_max_v2, arg_min）

### 核心 API

```cpp
// ArgMaxV1: 每行找最大值及其下标（0-based，相对本片起始）
ArgMaxV1(dst_indice, dst_values, src, batchSize, R_slice);
```

### 分片合并流程

```cpp
void ProcessCutR(uint64_t batchSize) {
    // --- 第一片：初始化 partial ---
    CopyInXCutR(0, 0, cutRSize_);
    ArgMaxV1(indiceUb[0], valuesUb[0], xUb, batchSize, cutRSize_);

    // --- 后续片：与 partial 合并 ---
    for (uint64_t rLoop = 1; rLoop < loopR_; rLoop++) {
        CopyInXCutR(0, rLoop, cutRSize_);
        ArgMaxV1(indiceUb[batchSize], valuesUb[batchSize], xUb, batchSize, cutRSize_);
        // 合并：比较新旧最大值，更新全局下标（加 rOffset）
        UpdateResult(indiceUb, valuesUb, batchSize, rLoop * cutRSize_);
    }

    // --- 尾片 ---
    if (tailR_ > 0) {
        CopyInXCutR(0, loopR_, tailR_);
        ArgMaxV1(indiceUb[batchSize], valuesUb[batchSize], xUb, batchSize, tailR_);
        UpdateResult(indiceUb, valuesUb, batchSize, loopR_ * cutRSize_);
    }
}

void UpdateResult(indice, values, batchSize, rOffset) {
    // half/float: 向量 VCMax + VCsel（高效）
    // int64: 退化为标量循环（硬件不支持向量 int64 比较）
    for (uint64_t b = 0; b < batchSize; b++) {
        if (values[b + batchSize] > values[b]) {  // ArgMin 取 <
            values[b] = values[b + batchSize];
            indice[b] = indice[b + batchSize] + rOffset;  // 绝对偏移
        }
    }
}
```

### 关键实现 API 参考

| API | 用途 | 来源算子 |
|-----|------|---------|
| `VFMeanVarTwoPassAR<T, isStd>()` | AR 模式 TwoPass 求 mean+var | reduce_var |
| `VFWelfordParallelUpdate()` | Welford 在线更新 | reduce_var |
| `VFWelfordParallelFinalize*()` | Welford 最终化（多种对齐变体） | reduce_var |
| `ArgMaxV1()` | 单片 ArgMax（返回 indice+value） | arg_max_v2 |
| `ReduceSumHalfInterval()` | 半区间折叠归约 | rms_norm, layer_norm |
| `DichotomyAdd()` | 二分累加（Sum 专用，防大数吃小数） | reduce_var TwoPass |
