# Index-Tracking 索引跟踪类算子设计指南

## ⚠️ 前置要求

**Index-Tracking 算子复用 Reduction 的 Tiling 方法论**（3D 抽象、AR/ARA 判定、全载/分载判定）。
各算子的详细文档已内置分支判定，可直接使用。

本指南**只补充**索引跟踪的特殊处理方法。

---

## 算子分类

| 算子 | 功能 | 详细文档 |
|------|------|---------|
| **ArgMax/ArgMin** | 找最大/最小值索引 | ✅ [argmax.md](argmax.md) |
| **Top-K** | 前K个值+索引 | 📋 规划中 |
| **Sort/Argsort** | 排序 | 📋 规划中 |
| **Where/NonZero** | 条件查找索引 | 📋 规划中 |

---

## 共同特征

### 输出包含索引

Index-Tracking 算子的输出除了值还有索引/位置，需要额外的 Buffer 和类型处理。

### 与 Reduction 的区别

| 维度 | Reduction | Index-Tracking |
|------|-----------|----------------|
| **输出** | 只返回值 | 值 + 索引/位置 |
| **Buffer** | 较少（2-3个） | 更多（5-6个，需存储索引） |
| **Tiling** | 完全相同 | 完全相同（3D 抽象、AR/ARA、FullLoad/Split） |

### 常见约束（A2/A3）

| 约束 | 说明 |
|------|------|
| **索引类型** | Select 不支持 int32 dst，索引用 float 存储，最后 Cast 为 int32 |
| **索引范围** | float32 精确表示 [0, 2^24] 整数 |
| **索引类型转换** | `reinterpret_cast<uint32_t*>(&floatIdx)` 或 `Cast(CAST_ROUND)` |
