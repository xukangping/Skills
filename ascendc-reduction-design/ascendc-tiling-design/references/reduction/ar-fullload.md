# Reduce 类算子 - AR 全载分支

> **适用场景**: A0=1（尾轴）, R ≤ threshold（全载模式）

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
| **载入模式** | 全载（整行完整放入 UB） |
| **适用条件** | A0=1, R ≤ fullLoadThreshold |
| **数据连续性** | 每行 R 个元素连续 |
| **Reduce 结果** | 标量（1 个值） |

---

## 二、Buffer 规划

### 2.1 FP32 场景

```cpp
// Double Buffer 模式
pipe->InitBuffer(inQueueX, 2, R * sizeof(float));
pipe->InitBuffer(outQueueY, 2, 32);  // Reduce结果（1个标量）
pipe->InitBuffer(tmpBuf, tmpBufSize);

// 总 UB = 2×R×4 + 2×32 + tmpBufSize
```

### 2.2 FP16 场景（混合精度）

```cpp
// FP16 输入，FP32 计算
pipe->InitBuffer(inQueueX, 2, R * sizeof(half));
pipe->InitBuffer(outQueueY, 2, 32);  // FP32 标量 buffer
pipe->InitBuffer(tmpBuf, tmpBufSize);

// 总 UB = 2×R×2 + 2×32 + tmpBufSize
```

### 2.3 tmpBufSize 计算

```cpp
uint32_t ComputeReduceBufSize(uint32_t rLengthAlign, uint32_t typeSize) {
    uint32_t perRepeat = 256 / typeSize;  // 64 for FP32
    uint32_t perBlock = 32 / typeSize;     // 8 for FP32
    uint32_t repeats = (rLengthAlign + perRepeat - 1) / perRepeat;
    uint32_t tmpBufSize = ((repeats + perBlock - 1) / perBlock) * perBlock * typeSize;
    return std::max(tmpBufSize, 4096u);  // 最小 4KB
}
```

---

## 三、Tiling 参数计算

### 3.1 全载阈值

```cpp
// 全载条件: 2×R×typeSize + 64 + tmpBufSize ≤ UB_SIZE
uint32_t fullLoadThreshold = (UB_SIZE - 100) / (2 * typeSize);
fullLoadThreshold = min(fullLoadThreshold, 12000);  // 实际测试建议值
```

### 3.2 多核切分参数

```cpp
// 按 A1（行）切分
uint32_t rowsPerCore = (A1 + blockDim - 1) / blockDim;
uint32_t usedCoreNum = (A1 + rowsPerCore - 1) / rowsPerCore;
uint32_t tailCoreRows = A1 % rowsPerCore;
if (tailCoreRows == 0 && A1 > 0) tailCoreRows = rowsPerCore;
```

### 3.3 对齐处理

```cpp
// 计算对齐后的列数
uint32_t alignedCols = ((R * sizeof(float) + 31) / 32) * 32 / sizeof(float);

// 使用 DataCopyPad 处理非对齐
DataCopyExtParams copyParams{1, static_cast<uint32_t>(R * sizeof(float)), 0, 0, 0};
DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
```

---

## 四、Kernel 实现要点

### 4.1 数据流

```
GM (A1, R) → UB (R)
    ↓
[ReduceMax/ReduceSum] → result (1)
    ↓
UB (1) → GM (A1)
```

### ⚠️ 数据搬运 API 黑名单

**禁止使用 DataCopy 进行 GM ↔ UB 数据搬运**

| API | 状态 | 原因 |
|-----|------|------|
| `DataCopy` | ❌ **禁止** | 不支持非对齐数据，易导致隐蔽 bug |
| `DataCopyPad` | ✅ **强制使用** | 统一处理对齐/非对齐场景，避免边界问题 |

**正确示例**：
```cpp
// ✅ 正确：使用 DataCopyPad
AscendC::DataCopyPadParams padParams;
AscendC::DataCopyPad(xLocal, xGm[offset], 
    {1, static_cast<uint16_t>(R * sizeof(float)), 0, 0}, padParams);
```

**错误示例**：
```cpp
// ❌ 错误：使用 DataCopy（当 R 不是32的倍数时会出错）
AscendC::DataCopy(xLocal, xGm[offset], R);  // 危险！
```

### 4.2 核心 API 调用

> **推荐**：AR 全载使用 **Level 2 接口（逐行处理）**，更简单且无对齐要求

```cpp
for (uint32_t row = 0; row < rowsThisLoop; row++) {
    uint32_t rowOffset = row * rLengthAlign;  // ⚠️ 关键：用对齐后的长度
    
    // ReduceMax - 使用有效数据个数
    ReduceMax<T>(resultLocal, xLocal[rowOffset], tmpLocal, 
                 static_cast<int32_t>(rLength), false);
    
    // 或 ReduceSum
    ReduceSum<T>(resultLocal, xLocal[rowOffset], tmpLocal, 
                 static_cast<int32_t>(rLength), false);
    
    // 输出结果
    T result = resultLocal.GetValue(0);
    // ... 后续处理
}
```

**关键要点**:
- `rowOffset` 计算：用 `rLengthAlign`（UB 中每行按 32 字节对齐存储）
- API `count` 参数：用 `rLength`（只处理有效数据，不包括 padding）
- Buffer 大小：用 `rLengthAlign`（需要容纳对齐后的数据）

### 4.3 参数使用对照表

| 参数位置 | 用 rLength | 用 rLengthAlign |
|---------|-----------|-----------------|
| DataCopyPad blockLen | ✓ | ✗ |
| Reduce API count (Level 2) | ✓ | ✗ |
| UB rowOffset | ✗ | ✓ |
| Buffer 大小计算 | ✗ | ✓ |

### 4.4 流水线设计

**Double Buffer 模式** (depth=2):
```
Tile N:   CopyIn(row0) → Compute(row0) → CopyOut(row0)
Tile N+1:              CopyIn(row1) → Compute(row1) → CopyOut(row1)
```

---

## 五、测试用例

### 5.1 功能测试矩阵

| ID | Shape | Axis | Dtype | 说明 |
|----|-------|------|-------|------|
| AR-F01 | (1, 128) | -1 | FP32 | 最小用例，单行 |
| AR-F02 | (4, 128) | -1 | FP32 | 多行，基础场景 |
| AR-F03 | (2, 256) | -1 | FP32 | 多行，更大 R |
| AR-F04 | (8, 512) | -1 | FP32 | 多核场景 |
| AR-F05 | (1, 1000) | -1 | FP32 | 较大 R |
| AR-F06 | (4, 128) | -1 | FP16 | FP16 混合精度 |

### 5.2 边界测试用例

| ID | 场景 | 参数 | 预期结果 |
|----|------|------|---------|
| AR-B01 | R 边界 | R=threshold | 全载，精度达标 |
| AR-B02 | 小 R | R=8 | 最小对齐单位 |
| AR-B03 | 非对齐 R | R=100 | DataCopyPad 处理 |
| AR-B04 | 单元素 | shape=(1, 1) | 正确输出 |
| AR-B05 | 大 A1 | A1=1000 | 多核负载均衡 |

### 5.3 精度要求

**相对误差阈值**: < 1e-5

```python
def check_precision(output, ref):
    abs_diff = np.abs(output - ref)
    rel_diff = abs_diff / (np.abs(ref) + 1e-8)
    return np.max(rel_diff) < 1e-5
```

---

## 六、常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| ReduceMax 编译错误 | Level 2 接口参数不匹配 | 确保使用：`ReduceMax(dst, src, tmp, count)` |
| 输出全 0 | Buffer 未正确初始化 | 检查 AllocTensor/FreeTensor 配对 |
| FP16 精度差 | 中间计算精度不足 | 使用 FP32 中间计算 |
| 核心超时 | Buffer 泄漏 | 检查所有路径的 FreeTensor |
| 非对齐场景精度错误 | count 用 rLengthAlign 而非 rLength | count 参数用 `rLength`（有效数据个数） |
| 多行数据错误 | rowOffset 计算错误 | rowOffset 用 `rLengthAlign`（UB 按对齐存储） |

---

## 七、性能优化建议

1. **Double Buffer**: 使用 depth=2 的队列
2. **对齐处理**: 提前计算 alignedCols，避免重复计算
3. **流水线**: CopyIn/Compute/CopyOut 并行
4. **FP16 混合精度**: FP16 输入，FP32 计算，FP16 输出
