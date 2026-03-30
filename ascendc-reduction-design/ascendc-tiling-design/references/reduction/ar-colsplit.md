# Reduce 类算子 - AR Col-Split 分支

> **适用场景**: A0=1（尾轴）, R > threshold（分载模式）

---

## 目录

- [一、分支特征](#一分支特征)
- [二、Buffer 规划](#二buffer-规划)
- [三、Tiling 参数计算](#三tiling-参数计算)
- [四、Kernel 实现要点](#四kernel-实现要点)
- [五、测试用例](#五测试用例)
- [六、常见问题](#六常见问题)
- [七、性能优化建议](#七性能优化建议)

---

## 一、分支特征

| 特征 | 说明 |
|------|------|
| **模板类型** | AR 模板（A 轴 + R 轴） |
| **Shape 抽象** | (A1, R) |
| **载入模式** | 分载（整行放不下，分 chunk 处理） |
| **适用条件** | A0=1, R > fullLoadThreshold |
| **切分方向** | 列方向（沿 R 分 chunk） |
| **数据连续性** | 每行 R 个元素连续 |
| **Reduce 结果** | 标量（1 个值） |

---

## 二、Buffer 规划

### 2.1 FP32 场景

```cpp
// Single Buffer 模式（分 chunk 处理）
pipe->InitBuffer(inQueueX, 1, chunkCols * sizeof(float));
pipe->InitBuffer(outQueueY, 1, 32);  // Reduce结果（1个标量）
pipe->InitBuffer(chunkResultBuf, 1, 32);  // chunk中间结果
pipe->InitBuffer(tmpBuf, 1, tmpBufSize);

// 总 UB = chunkCols×4 + 32 + 32 + tmpBufSize
```

### 2.2 chunkCols 计算

```cpp
// 基于 UB 容量计算
uint32_t chunkCols = (UB_SIZE * 0.9) / (sizeof(float) + 64 + tmpBufSize);
chunkCols = std::min(chunkCols, R);

uint32_t numChunks = (R + chunkCols - 1) / chunkCols;
uint32_t lastChunkSize = R - (numChunks - 1) * chunkCols;
```

---

## 三、Tiling 参数计算

### 3.1 分载判定

```cpp
uint32_t fullLoadThreshold = (UB_SIZE - 100) / (2 * typeSize);

if (R > fullLoadThreshold) {
    // AR-Col-Split 模式
    loadMode = LOAD_SPLIT;
    
    // 计算 chunk 大小
    uint32_t chunkCols = (UB_SIZE * 0.9) / (sizeof(float) + 64 + tmpBufSize);
    chunkCols = std::min(chunkCols, R);
    
    uint32_t numChunks = (R + chunkCols - 1) / chunkCols;
}
```

### 3.2 多核切分参数

```cpp
// 按 A1（行）切分
uint32_t rowsPerCore = (A1 + blockDim - 1) / blockDim;
uint32_t usedCoreNum = (A1 + rowsPerCore - 1) / rowsPerCore;
uint32_t tailCoreRows = A1 % rowsPerCore;
if (tailCoreRows == 0 && A1 > 0) tailCoreRows = rowsPerCore;
```

---

## 四、Kernel 实现要点

### 4.1 数据流

```
GM (A1, R) → 分 chunk 处理
    ↓
Chunk 0: GM[0:chunkCols] → UB
    ↓
[ReduceMax] → chunkResult_0
    ↓
[更新 globalResult] → globalResult = merge(globalResult, chunkResult_0)
    ↓
Chunk 1, 2, ... (重复)
    ↓
UB → GM (A1)
```

### 4.2 核心 API 调用

#### ReduceMax 分载实现

```cpp
// 初始化全局最大值为 -∞
float globalMax = -INFINITY;

for (uint32_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    uint32_t chunkStart = chunkIdx * chunkCols;
    uint32_t chunkSize = (chunkIdx == numChunks - 1) ? lastChunkSize : chunkCols;
    
    // Load chunk
    DataCopy(xLocal, xGm[chunkStart], chunkSize);
    
    // ReduceMax for this chunk
    ReduceMax<float>(chunkResultLocal, xLocal, tmpLocal, chunkSize, false);
    float chunkMax = chunkResultLocal.GetValue(0);
    
    // Update globalMax (取最大值)
    globalMax = std::max(globalMax, chunkMax);
}

// 输出最终结果
resultLocal.SetValue(0, globalMax);
```

#### ReduceSum 分载实现

```cpp
// 初始化全局和为 0
float globalSum = 0.0f;

for (uint32_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    uint32_t chunkStart = chunkIdx * chunkCols;
    uint32_t chunkSize = (chunkIdx == numChunks - 1) ? lastChunkSize : chunkCols;
    
    // Load chunk
    DataCopy(xLocal, xGm[chunkStart], chunkSize);
    
    // ReduceSum for this chunk
    ReduceSum<float>(chunkResultLocal, xLocal, tmpLocal, chunkSize, false);
    float chunkSum = chunkResultLocal.GetValue(0);
    
    // Update globalSum (累加)
    globalSum += chunkSum;
}

// 输出最终结果
resultLocal.SetValue(0, globalSum);
```

### 4.3 关键注意点

1. **跨 chunk 合并**: 
   - ReduceMax: 使用 `std::max` 或 `Max<T>` API 取最大值
   - ReduceSum: 使用 `+=` 或 `Add<T>` API 累加

2. **边界处理**: 最后一个 chunk 的大小可能小于 chunkCols

3. **初始化**: 
   - ReduceMax: 初始化为 `-INFINITY` 或数据类型的最小值
   - ReduceSum: 初始化为 `0`

---

## 五、测试用例

### 5.1 功能测试矩阵

| ID | Shape | Axis | Dtype | 说明 |
|----|-------|------|-------|------|
| AR-C01 | (1, 15000) | -1 | FP32 | 单行，触发 Col-Split |
| AR-C02 | (2, 20000) | -1 | FP32 | 多行，大规模 |
| AR-C03 | (4, 30000) | -1 | FP32 | 多核，大规模 |
| AR-C04 | (1, 50000) | -1 | FP32 | 超大 R |
| AR-C05 | (2, 15000) | -1 | FP16 | FP16 大规模 |

### 5.2 边界测试用例

| ID | 场景 | 参数 | 预期结果 |
|----|------|------|---------|
| AR-CB01 | R=threshold+1 | R=12001 | 触发 Col-Split |
| AR-CB02 | R 边界 | R=2*threshold | 2 个 chunk |
| AR-CB03 | 非对齐 R | R=15001 | 3 个 chunk，最后一个不完整 |
| AR-CB04 | 大 chunk | chunkCols=10000 | 每个处理 10000 元素 |

### 5.3 精度要求

**相对误差阈值**: < 1e-5

---

## 六、常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| 精度下降 | chunk 合并逻辑错误 | 确保正确合并（Max取最大值，Sum累加） |
| 输出错误 | 最后一个 chunk 大小处理错误 | 使用 lastChunkSize 而非 chunkCols |
| 性能差 | 多次遍历数据 | 优化 chunk 策略，减少遍历次数 |
| Buffer 不足 | chunkCols 计算错误 | 基于 UB 容量正确计算 |

---

## 七、性能优化建议

1. **chunk 大小优化**: 根据 UB 容量选择最优 chunk 大小，尽量大以减少chunk数量
2. **流水线**: 不同行的 chunk 可以并行处理
3. **避免不必要的数据拷贝**: 直接在chunk结果上合并，减少中间存储
