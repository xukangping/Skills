# 算子类别 Tiling 设计知识提取 — 提示词模板

> **用法**：将 `{算子类型}` 替换为目标类别（如 Elementwise / Broadcast / MatMul 等），
> 将 `{代表算子列表}` 替换为该类别下的典型算子目录名（建议 6-10 个），按三步依次执行。
>
> **产出目标**：符合 `operator-category-template.md` 定义的目录结构，可直接放入 `references/{算子类型}/`。
>
> **参照实例**：`references/reduction/` 目录（从 ops-math/ops-nn 代码仓提取而来）。

---

## 第一步：场景识别与分类

```
读取以下 {算子类型} 类算子的 op_host/*tiling*.h 文件：
  {代表算子列表，逗号分隔}

任务：识别所有 Tiling 分支（TilingKey / mode / pattern / branch），按以下维度分类：

1. 数据访问模式：该类算子在不同 shape/参数下有哪几种数据访问模式？
2. 载入模式：全载（FullLoad）vs 分载（Split）— 判定条件是什么？
3. 特殊策略：该类算子是否有特有的计算策略或优化路径？

输出 — 场景判定表：
| 场景 | 数据访问模式 | 载入模式 | 触发条件（shape/dtype/参数） | 代表算子 | TilingKey/分支标识 |

同时回答：
- 该类算子的数据布局抽象方式是什么？（如 Reduction 的 3D 抽象 (A1, R, A0)）
- 多核切分通常沿哪个维度？
- 是否存在需要跨核协作的场景（workspace / atomic / 两阶段合并）？
- FullLoad 和 Split 的判定公式是什么？
```

## 第二步：逐场景提取实现细节

```
对第一步识别出的每个场景，读取对应算子的 op_kernel 和 op_host tiling 代码，
提取以下 7 项内容（每项都必须有，缺一不可）：

1. **数据流图**：GM → UB → 计算 → UB → GM
   - 标注每步使用的具体 API 名称
   - 如果有多个计算阶段，逐阶段画出

2. **核心 API 调用**：完整的函数签名和参数值
   - 不能用 ... 省略参数
   - 如果同一位置有多个 API 可选（如 Level 2 vs Pattern 版本），说明代码实际用的是哪个
   - 不同算子在同一场景下 API 调用的差异，对比列出

3. **DataCopyPad 配置**：blockCount / blockLen / srcStride / dstStride 的具体计算公式
   - 标注 stride 单位（GM=字节，UB=dataBlock=32B）
   - CopyIn 和 CopyOut 分别列出（参数数量不同：CopyIn 4 参数，CopyOut 3 参数）

4. **Buffer 规划**：列出所有 Buffer 名称、用途、大小计算公式
   - 标注是 TQue（有 depth）还是 TBuf（无 depth）
   - 标注 TPosition（VECIN / VECOUT / VECCALC）

5. **全载/分载判定**：
   - FullLoad 阈值怎么算？（列出完整公式）
   - 分载时怎么分 chunk？跨 chunk 怎么合并？

6. **多核切分**：
   - 按哪个维度切？约束条件有几个？
   - 动态核数获取用什么 API？

7. **归约/计算后的后续操作**：
   - 计算结果（如归约向量、中间标量）需要广播回原矩阵时，用什么 API？
   - 是标量广播（Adds/Muls）还是向量广播（BinaryRepeatParams src1RepStride=0）？
   - 是否有其他常见的后续操作模式？

⚠️ 重点关注：
- 代码中数据布局变换（Transpose/Reshape/Pad）是在哪一侧完成的（Host/Kernel/UB 内）？
- 不同算子在同一场景下，代码结构相同但参数不同的地方有哪些？提取为"通用模板 + 差异点"
- Buffer 是否有复用？哪些 Buffer 生命周期不重叠可以共享？
```

## 第三步：提取通用规则和反模式

```
再次通读所有已分析算子的 op_kernel 和 op_host 代码，回答以下问题：

**数据搬运规范**：
1. GM↔UB 搬运用的是 DataCopy 还是 DataCopyPad？
2. CopyIn 和 CopyOut 的参数数量是否不同？（CopyIn 4 参数有 padParams，CopyOut 3 参数无 padParams）

**API 使用规范**：
3. 是否有底层硬件指令 API（WholeReduce* / BlockReduce*）？生产代码推荐用哪层？
4. 是否在 Kernel 中使用 std:: 函数或动态内存分配？

**架构规范**：
5. 有没有在 Host 侧做计算型数据变换（Transpose/Pad/Reshape 循环）？
6. 核数是动态获取还是写死的？
7. Tiling 参数计算在 Host 侧还是 Kernel 侧？

**精度与数值**：
8. FP16/BF16 输入的算子，中间计算用什么精度？在哪里做 Cast？

**性能**：
9. 双缓冲（depth=2）在哪些场景启用？
10. 有没有 V-S（Vector→Scalar）同步？

输出格式：
- **通用规则清单**：适用于所有 {算子类型} 算子的强制规范 → 放入 patterns.md 通用规则段
- **反模式清单**：| 反模式 | 为什么错 | 正确做法 | 代码出处 |
```

---

## 第四步：组织为渐进式披露文档

```
基于前三步的提取结果，按以下结构组织输出文档：

目标结构（参照 operator-category-template.md）：

references/{算子类型小写}/
├── patterns.md           # 路由文档（≤300行）
│   ├── 场景判定决策树（从第一步的场景表提炼）
│   ├── 数据布局抽象（该类型特有的抽象方式）
│   ├── 通用规则（从第三步提取）
│   └── 各分支摘要 + 决策树 + 广播指引 + 链接
│
├── {branch-N}.md         # 每个分支一个文件（≤300行）
│   └── 从第二步提取的 7 项内容
│
├── algorithms.md         # 算法路由（≤80行）
│   └── 算法选择表 + 摘要 + 链接
│
├── alg-{algo-N}.md       # 每个算法一个文件（≤150行）
│
├── tiling-fields.md      # Tiling 设计原则 + 字段定义
├── optimizations.md      # 性能优化模式
└── multi-output-buffer.md # 多输出 Buffer 方程（如适用）

文件规范：
- patterns.md ≤ 300 行（路由 + 通用规则 + 摘要）
- 单个分支文件 ≤ 300 行
- algorithms.md ≤ 80 行（纯路由）
- 单个算法文件 ≤ 150 行
- 所有 API 调用必须有完整参数，不能用 ... 省略
- 所有 Buffer 必须标注 TQue/TBuf 和 TPosition
- 场景/分支/算法的分类条件用数据布局和 UB 容量描述，算子名仅作为"典型算子"示例
- 归约/计算后需要广播的场景，在分支摘要中给出 API 指引
```

---

## 提取经验总结（来自 Reduction 迭代）

以下是从 Reduction 类多轮提取和改进中积累的经验，适用于所有类别：

### 必须做的

1. **场景判定用决策树**，不用文字描述。Architect 看决策树两步就能定位分支
2. **FullLoad vs Split 的判定条件写完整公式**，不写"R 远超 UB"这种模糊描述
3. **每个分支文件包含完整的 API 代码**（函数签名 + 参数值），Developer 能直接参考
4. **广播操作的 API 指引放在分支摘要中**，不要让 Agent 自己去找
5. **算法和分支解耦**：算法是正交维度（执行策略），不是分支判定条件

### 不要做的

1. **不要用算子名称作为分类条件**（如"Softmax 用 Recompute"），用抽象模式（如"分载模式下多个归约操作需要多轮扫描"）
2. **不要在路由文档中内联大段代码**（patterns.md/algorithms.md 只放摘要和链接）
3. **不要留已删除文件的引用**（物理删除旧文件，grep 确认无残留引用）
4. **不要在通用模板（如 design-template.md）中写类别特有内容**
5. **不要假设 Agent 会自动查阅 best-practices**，在需要的位置（如广播操作）直接给出 API 指引和链接
