# ArgMax/ArgMin 算子详细设计

## 3D 抽象与分支判定

```
输入 shape + axis → 3D 抽象 (A1, R, A0)
  A1 = 归约轴前所有维度之积
  R  = 归约维度
  A0 = 归约轴后所有维度之积

分支判定：
  A0 = 1 → 场景1（AR，最后一轴）
  A0 > 1 → 场景2（ARA，非最后一轴）

全载/分载判定：
  AR:  R × sizeof(T) ≤ UB / BUFFER_NUM → FullLoad，否则 ColSplit
  ARA: R × alignedCols × sizeof(T) ≤ UB 可用空间 → FullLoad，否则 RowSplit
```

---

## 场景1：最后一轴（axis=-1, A0=1）

### 特征
- 输入：`(A1, R)`，数据**连续**
- 输出：`(A1,)`，每个元素是**标量索引**
- API：ReduceMax(calIndex=true)

### API用法

```cpp
AscendC::LocalTensor<float> dstVal = outQueue.AllocTensor<float>();
AscendC::LocalTensor<float> sharedTmpBuffer = tmpQueue.AllocTensor<float>();

AscendC::ReduceMax<float>(dstVal, srcVal, sharedTmpBuffer, count, true);
// calIndex=true 同时返回值和索引

// 输出格式：dst[0]=最大值, dst[1]=索引
float maxVal = dstVal.GetValue(0);
float idxRaw = dstVal.GetValue(1);
uint32_t maxIdx = *reinterpret_cast<uint32_t*>(&idxRaw);  // 类型转换！
```

### tmpBuffer计算（calIndex=true）

```cpp
// 使用 API 计算（推荐）
uint32_t tmpSize = AscendC::GetReduceMaxMinTmpSize<T>(count, true);
```

### Buffer规划

| Buffer | 大小 | 用途 |
|--------|------|------|
| srcQueue | count×sizeof(T) | 输入数据 |
| dstQueue | 2×sizeof(T) | 输出值+索引 |
| tmpQueue | tmpBufSize×sizeof(T) | 中间计算 |

### 约束
- 只支持**连续数据**
- 索引范围：half <= 65535, float <= 2^32
- 多个最大值返回**第一个**

---

## 场景2：非最后一轴（A0>1）

### 特征
- 输入：`(A1, R, A0)`，数据**不连续**（stride=A0）
- 输出：`(A1, A0)`，每个位置是**向量索引**
- API：Compare + Select

### API 约束（A2/A3）

> 以下约束已在实际开发中验证（参考 ops/arg_max 算子），必须严格遵守。

| 约束项 | 具体限制 | 文档来源 | 影响 |
|--------|---------|---------|------|
| **Select dst 类型** | A2/A3 仅支持 half/float，**不支持 int32** | Select.md 第 115-116 行 | 索引必须用 float 存储，输出前 Cast 为 int32 |
| **Compare count 对齐** | count 个元素所占空间必须 **256 字节对齐**（float: 64 元素倍数） | Compare.md 第 114 行 | a0Aligned = ceil(A0/64)*64，非 32 字节对齐 |
| **DataCopyPad rightPadding** | rightPadding 不超过 **32 字节**（最多 8 个 float） | DataCopyPad.md 第 167 行 | 大 padding 量不能用 rightPadding 处理 |
| **Select 8K 预留** | 模式 1/2 需预留 8K UB 空间 | Select.md 第 176-177 行 | 框架可能自动管理，UB 充裕时通常不需手动预留 |
| **float 索引精度** | float32 精确表示 [0, 2^24] 整数 | IEEE 754 | R <= 16M 时索引无精度损失 |

### 推荐方案：LE 反转 + TENSOR_SCALAR

> 经实测验证，比基础方案性能提升 **15-20%**。循环内 3 条指令/轮（vs 基础方案 4 条/轮），且少用 1 个 buffer。

**核心技巧**：Compare 用 LE（而非 GT）反转 mask 极性，使 bit=1 表示"保留旧值"，从而用 `VSEL_TENSOR_SCALAR_MODE` 将当前行索引作为 scalar 传入 Select，省掉每轮的 `Duplicate` 操作。

**算法逻辑**：

```
Compare(LE): xLocal[r] <= maxLocal
  → bit=1: 当前行不大于旧最大值 → 保留旧值
  → bit=0: 当前行大于旧最大值   → 更新新值

Select(maxLocal, cmpLocal, maxLocal, xLocal[r], TENSOR_TENSOR):
  → bit=1: 保留 maxLocal（旧最大值）
  → bit=0: 取 xLocal[r]（新最大值）

Select(idxLocal, cmpLocal, idxLocal, rowIdxFloat, TENSOR_SCALAR):
  → bit=1: 保留 idxLocal（旧索引 tensor）
  → bit=0: 取 rowIdxFloat（新索引 scalar）
```

**"首个最大值"语义保证**：当 `xLocal[r] == maxLocal` 时，LE 成立（bit=1），保留旧索引——与 numpy.argmax 行为一致。

**参考实现**（来自 ops/arg_max/arg_max.asc，已实测验证）：

```cpp
__aicore__ inline void Compute()
{
    AscendC::LocalTensor<float>   xLocal   = inQueueX.DeQue<float>();
    AscendC::LocalTensor<int32_t> yLocal   = outQueueY.AllocTensor<int32_t>();
    AscendC::LocalTensor<float>   maxLocal = maxBuf.Get<float>();
    AscendC::LocalTensor<float>   idxLocal = idxBuf.Get<float>();      // 索引用 float 存储！
    AscendC::LocalTensor<uint8_t> cmpLocal = cmpBuf.Get<uint8_t>();

    // 初始化：第一行作为初始最大值，索引为 0.0f
    AscendC::DataCopy(maxLocal, xLocal, a0Aligned);
    AscendC::Duplicate<float>(idxLocal, 0.0f, a0Aligned);
    AscendC::PipeBarrier<PIPE_ALL>();  // DataCopy(MTE) + Duplicate(V) 跨 pipe

    // LE 反转 + TENSOR_SCALAR 优化循环
    float rowIdxFloat = 1.0f;  // 用 float 累加器避免 aicore 中 uint→float cast
    for (uint32_t r = 1; r < R; r++) {
        // bit=1: row[r] <= maxLocal (保留旧值); bit=0: row[r] > maxLocal (更新)
        AscendC::Compare(cmpLocal, xLocal[r * a0Aligned], maxLocal,
                         AscendC::CMPMODE::LE, a0Aligned);
        // bit=1 → 保留 maxLocal; bit=0 → 取 xLocal[r]
        AscendC::Select(maxLocal, cmpLocal, maxLocal, xLocal[r * a0Aligned],
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, a0Aligned);
        // bit=1 → 保留 idxLocal(tensor); bit=0 → 取 rowIdxFloat(scalar)
        AscendC::Select(idxLocal, cmpLocal, idxLocal, rowIdxFloat,
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, a0Aligned);
        rowIdxFloat = rowIdxFloat + 1.0f;
    }

    // float 索引 → int32 输出
    AscendC::Cast(yLocal, idxLocal, AscendC::RoundMode::CAST_ROUND, a0Aligned);
    outQueueY.EnQue<int32_t>(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

**与基础方案的对比**：

| 指标 | 基础方案（GT + Duplicate） | 推荐方案（LE + TENSOR_SCALAR） |
|------|:------------------------:|:-----------------------------:|
| 循环内指令 | 4 条/轮 | **3 条/轮** |
| Buffer 数量 | 6 个 | **5 个** |
| idxTempBuf | 需要（a0Aligned×4 字节） | **不需要** |
| 实测耗时 (128,64,32) | 30.66 us | **24.32 us (-20.7%)** |
| 实测耗时 (128,64,36) | 36.20 us | **29.72 us (-17.9%)** |

### 基础方案（参考）

> 性能较差，仅作为理解原理的参考。如需使用，注意修正索引类型为 float（不是 int32）。

```cpp
// 注意：索引必须用 float 类型，Select 在 A2/A3 不支持 int32
float rowIdxFloat = 1.0f;
for (uint32_t r = 1; r < R; r++) {
    AscendC::Compare(cmpLocal, xLocal[r * a0Aligned], maxLocal,
                     AscendC::CMPMODE::GT, a0Aligned);
    AscendC::Duplicate<float>(idxTempLocal, rowIdxFloat, a0Aligned);  // 每轮生成索引向量
    AscendC::Select(maxLocal, cmpLocal, xLocal[r * a0Aligned], maxLocal,
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, a0Aligned);
    AscendC::Select(idxLocal, cmpLocal, idxTempLocal, idxLocal,
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, a0Aligned);
    rowIdxFloat = rowIdxFloat + 1.0f;
}
```

### Buffer 规划（推荐方案：5 个）

| Buffer | 类型 | 大小 | 用途 | Queue |
|--------|------|------|------|-------|
| inQueueX | float | R×a0Aligned×4 | 输入数据 | TQue VECIN, depth=2 |
| outQueueY | int32 | a0Aligned×4 | 输出索引 | TQue VECOUT, depth=2 |
| maxBuf | float | a0Aligned×4 | 当前最大值 | TBuf VECCALC |
| idxBuf | float | a0Aligned×4 | 当前索引（float 存储） | TBuf VECCALC |
| cmpBuf | uint8_t | max(a0Aligned/8, 32) | 比较结果 mask | TBuf VECCALC |

> 基础方案需额外增加 idxTempBuf（float, a0Aligned×4 字节），共 6 个 buffer。

### UB 计算

```
UB_USED = 2 × (R × a0Aligned × 4)       // inQueueX (depth=2)
        + 2 × (a0Aligned × 4)           // outQueueY (depth=2)
        + a0Aligned × 4                  // maxBuf
        + a0Aligned × 4                  // idxBuf
        + max(a0Aligned / 8, 32)         // cmpBuf（32 字节对齐）
```

**示例**（FP32, R=64, A0=32, a0Aligned=64）：
```
= 2×(64×64×4) + 2×(64×4) + 64×4 + 64×4 + 32
= 32768 + 512 + 256 + 256 + 32
= 33,824 字节 ≈ 33 KB << 192 KB  ✅ 全载
```

### Compare 256 字节对齐策略

**对齐计算**（非 32 字节对齐）：

```cpp
// float 类型：向上取整到 64 的倍数（256 字节 / 4 字节 = 64 元素）
uint32_t a0Aligned = ((A0 + 63) / 64) * 64;
```

| 原始 A0 | a0Aligned | Pad 量 |
|---------|-----------|--------|
| 32 | 64 | 32 |
| 36 | 64 | 28 |
| 64 | 64 | 0 |
| 65 | 128 | 63 |

**Pad 区域处理**：

Pad 区域（a0Aligned - A0 个元素）包含未初始化数据，但**不影响最终输出**：
- Compare/Select 操作覆盖整个 a0Aligned，pad 区域参与计算
- CopyOut 只输出前 curA0Len 个有效元素，pad 区域结果不写回 GM

因此**不需要**用 `Duplicate(-FLT_MAX)` 预填充 pad 区域。省掉预填充可减少一次大范围矢量写操作和一次 `PipeBarrier<PIPE_ALL>()`。

---

## ArgMin 实现

与 ArgMax 的唯一区别是 Compare 模式反转：

| | ArgMax | ArgMin |
|--|--------|--------|
| 推荐方案 | Compare(**LE**) | Compare(**GE**) |
| 基础方案 | Compare(**GT**) | Compare(**LT**) |

其余逻辑（Select、Buffer、对齐、索引类型）完全相同。

---

## 性能优化技巧

### 技巧1：LE/GE 反转 + TENSOR_SCALAR 模式

**原理**：通过反转 Compare 的比较方向（GT→LE / LT→GE），使 mask 中 bit=1 表示"保留旧值"（多数位置），bit=0 表示"更新新值"（少数位置）。这样索引更新的 Select 可以用 `VSEL_TENSOR_SCALAR_MODE`：
- bit=1 → 从 tensor（idxLocal）取值（保留旧索引）
- bit=0 → 从 scalar（rowIdxFloat）取值（更新新索引）

**收益**：
- 省掉每轮循环的 `Duplicate(idxTempLocal, rowIdxFloat, count)` 操作
- 省掉 idxTempBuf（一个 TBuf<VECCALC>）
- 循环内从 4 条指令减为 3 条（-25%）

**适用条件**：
- ArgMax/ArgMin 的非最后一轴场景（Compare + Select 逐行迭代）
- 需要 Select 的 TENSOR_SCALAR 模式在目标芯片上支持 float 类型

### 技巧2：Pad 区域无需预填充

**原理**：CopyOut 只输出 curA0Len 个有效元素到 GM，pad 区域的计算结果不会影响最终输出。因此不需要 `Duplicate(-FLT_MAX)` 预填充整个 buffer。

**收益**：
- 省掉一次 R×a0Aligned 个元素的 Duplicate 操作
- 省掉一次 `PipeBarrier<PIPE_ALL>()`（Duplicate 和 DataCopyPad 的跨 pipe 同步）

**注意**：仅当 CopyOut 能精确控制输出长度时适用。使用 DataCopyPad 的 blockLen 参数控制。

### 技巧3：float 累加器替代 Cast

aicore 中不支持 `static_cast<float>(uint32_t)`。用 float 变量逐步 +1.0f 替代：

```cpp
float rowIdxFloat = 1.0f;
for (uint32_t r = 1; r < R; r++) {
    // 使用 rowIdxFloat 而非 static_cast<float>(r)
    ...
    rowIdxFloat = rowIdxFloat + 1.0f;
}
```

---

## 约束汇总

| 约束 | 说明 |
|------|------|
| **索引类型** | A2/A3 上 Select 不支持 int32，索引必须用 float 存储，最后 Cast 为 int32 |
| **索引范围** | float32 精确表示 [0, 2^24] 整数；half <= 65535 |
| **多个最大值** | 返回**第一个**最大值的索引（LE/GE 方案自动保证） |
| **Compare 256 字节对齐** | count 个元素占 256 字节对齐（float: 64 元素倍数） |
| **DataCopyPad rightPadding** | rightPadding <= 32 字节（最多 8 个 float），大 padding 量不能用此参数 |
| **Pad 区域** | 不需要预填充，CopyOut 只输出有效元素 |
| **Select 8K 预留** | A2/A3 上模式 1/2 需预留 8K UB，框架通常自动管理 |
| **Buffer 数量** | 推荐方案 5 个，基础方案 6 个 |

---

## 设计检查清单

### 场景识别
- [ ] 3D 抽象计算 A1, R, A0
- [ ] 判断模板类型（AR vs ARA）
- [ ] 判断载入模式（全载 vs 分载）

### API 选择
- [ ] 最后一轴：ReduceMax(calIndex=true)
- [ ] 非最后一轴：Compare(LE) + Select(TT) + Select(TS)（推荐方案）
- [ ] 确认 Select 类型约束：A2/A3 不支持 int32，索引用 float

### Buffer 规划
- [ ] 最后一轴：3 个 buffer
- [ ] 非最后一轴（推荐）：5 个 buffer（inQueueX, outQueueY, maxBuf, idxBuf, cmpBuf）
- [ ] 非最后一轴（基础）：6 个 buffer（+idxTempBuf）

### 对齐与同步
- [ ] a0Aligned = ceil(A0/64)*64（256 字节对齐，非 32 字节）
- [ ] cmpBuf >= 32 字节（LocalTensor 32 字节对齐要求）
- [ ] PipeBarrier<PIPE_ALL>：仅用于跨 pipe 操作（如 DataCopy + Duplicate 之间）
- [ ] 循环内同 pipe (V) 操作不需要 PipeBarrier

### 输出
- [ ] Cast<int32_t, float>(CAST_ROUND) 将 float 索引转为 int32
- [ ] CopyOut 只输出 curA0Len 个有效元素

---

## 实测性能参考

> 数据来源：ops/arg_max 算子，Ascend910B (A2/A3), CANN 9.0.0

| Shape | DType | Axis | 方案 | Task Duration | vec_fops/core |
|-------|-------|------|------|:------------:|:-------------:|
| (128,64,32) | float32 | 1 | 基础（GT+Duplicate） | 30.66 us | 163,584 |
| (128,64,32) | float32 | 1 | **推荐（LE+TS）** | **24.32 us** | **138,752** |
| (128,64,36) | float32 | 1 | 基础（GT+Duplicate） | 36.20 us | 163,584 |
| (128,64,36) | float32 | 1 | **推荐（LE+TS）** | **29.72 us** | **138,752** |

---

## AR-ColSplit 分片合并（R > UB）

当 R 太大无法一次放入 UB 时，将 R 分为多个 chunk，每片独立求局部最大值+索引，跨片合并得到全局结果。

### 核心 API

```cpp
// ArgMaxV1: 对连续的 R_slice 个元素，找最大值及其下标
// dst_indice: 输出索引（0-based，相对本片起始位置）
// dst_values: 输出最大值
// src: 输入数据
// batchSize: 批量处理的行数（A1 维度）
// R_slice: 本片包含的元素数
ArgMaxV1(dst_indice, dst_values, src, batchSize, R_slice);
```

> ArgMaxV1 返回的索引是**片内偏移**（从 0 开始），合并时需加上片的全局起始位置转为绝对索引。

### 分片合并逻辑

1. **第一片**：调用 ArgMaxV1 得到初始的 (maxValue, maxIndex)
2. **后续每片**：调用 ArgMaxV1 得到本片的 (chunkValue, chunkIndex)，与全局 (maxValue, maxIndex) 比较：
   - 如果 chunkValue > maxValue → 更新 maxValue，maxIndex = chunkIndex + 片起始偏移
   - 否则保留原值
3. **尾片**：大小可能小于 cutRSize，同样处理
4. **ArgMin 差异**：比较条件从 `>` 改为 `<`

> **ARA-RowSplit 场景**：分片合并逻辑相同，区别仅在数据搬运（DataCopyPad 的 blockCount=R_chunk 行，带 srcStride）。每片归约后的跨片比较更新方式不变。
