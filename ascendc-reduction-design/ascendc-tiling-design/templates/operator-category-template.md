# 算子类别 Tiling 设计文档模板

> 本模板基于 Reduction 类的最终结构，用于为新算子类别创建 tiling-design 参考文档。
> 参照实例：`references/reduction/` 目录

---

## 目标产出目录结构

```
references/{category}/
│
├── patterns.md                    # 路由文档（~200-300行）
│   ├── 场景判定决策树              #   核心：输入 → 判定 → 分支
│   ├── 3D/ND 抽象映射              #   该类型的数据布局抽象
│   ├── 通用规则                    #   所有分支通用的强制规范
│   ├── 各分支摘要 + 链接           #   每个分支 3-5 行摘要 → 链接到详细文件
│   │   ├── 分支决策树              #   FullLoad vs Split 判定
│   │   └── 归约/计算后的广播指引   #   后续操作的 API 推荐
│   └── 跨场景参考链接表            #   算法、优化、Buffer 等文档链接
│
├── {branch-1}.md                  # 分支详细文档（~100-300行/个）
│   ├── 分支特征表
│   ├── Buffer 规划（公式 + 代码）
│   ├── Tiling 参数计算
│   ├── 数据流图
│   ├── 核心 API 调用（完整代码，不省略参数）
│   ├── DataCopyPad 配置
│   ├── 测试用例
│   └── 常见问题
│
├── {branch-2}.md                  # 其他分支...
│
├── algorithms.md                  # 算法路由文档（~50-80行）
│   ├── 算法选择对照表              #   条件 → 推荐算法 → 链接
│   └── 各算法摘要 + 链接
│
├── alg-{algo-1}.md                # 算法详细文档（~50-150行/个）
│   ├── 适用场景
│   ├── 核心公式/伪代码
│   └── 实现模板
│
├── multi-output-buffer.md         # 多输出 Buffer 方程（如适用）
├── tiling-fields.md               # Tiling 设计原则 + 字段定义
└── optimizations.md               # 性能优化模式
```

---

## 关键设计原则

### 渐进式披露

| 层级 | 文件 | 行数 | 读取时机 |
|------|------|------|---------|
| L1 | SKILL.md 分类表 | ~1 行/类别 | 始终在上下文 |
| L2 | patterns.md | ≤300 行 | 确定类别后读 |
| L3 | {branch}.md | ≤300 行 | 确定分支后读 |
| L3 | alg-{algo}.md | ≤150 行 | 确定算法后读 |

Architect 的阅读路径：SKILL.md → patterns.md（判定分支）→ 对应 branch.md（获取 API + Buffer）→ 如需选算法 → algorithms.md（选择表）→ 对应 alg-{algo}.md

### 抽象化

- 场景/分支/算法的分类条件用**数据布局和 UB 容量**描述，不用算子名称
- 算子名称仅作为"典型算子"示例出现，不作为分类依据
- 具体算子的计算逻辑（如"需要 mean+var"）描述为抽象模式（如"需要两次顺序归约"）

### 归约/计算后的广播指引

在分支摘要中，归约/计算结果需要回到原维度参与后续运算时，给出 API 指引：
- 标量广播：`Adds/Muls(dst, src, scalar, count)`
- 向量广播：`Sub/Div/Mul + BinaryRepeatParams{src1RepStride=0}`
- 链接到 `/ascendc-api-best-practices` 的详细文档

---

## patterns.md 模板

```markdown
# {Category} 类算子场景路由

> 本文档用于场景判定和策略选择。确定场景后，按链接进入对应详细文档。

## 场景判定决策树

（基于该类型的数据布局抽象，给出从输入到具体分支的判定流程）

## {类型特有的}抽象映射

（如 Reduction 的 3D 抽象 (A1, R, A0)，Broadcast 的 shape 对齐规则等）

## 通用规则

（适用于所有分支的强制规范，如 DataCopyPad 黑名单、对齐规则等）

## {分支 1}

（3-5 行摘要 + 分支决策树 + 广播指引 + 链接到详细文件）

## {分支 2}

...

## 跨场景参考

| 主题 | 文档 |
|------|------|
| 算法选择 | [algorithms.md](algorithms.md) |
| 性能优化 | [optimizations.md](optimizations.md) |
| ... | ... |
```

---

## {branch}.md 分支文档模板

```markdown
# {分支名称}

> 适用场景：{条件}

## 分支特征

| 特征 | 说明 |
|------|------|
| 数据布局 | |
| 载入模式 | FullLoad / Split |
| 适用条件 | |

## Buffer 规划

（Buffer 列表、大小公式、TQue vs TBuf 标注）

## Tiling 参数计算

（阈值公式、约束条件）

## 数据流

（GM → UB → 计算 → UB → GM，标注每步 API）

> 全载模式说明：（如适用）数据完整驻留 UB，CopyIn 一次，中间结果直接复用。

## 核心 API 调用

（完整函数签名和参数，不省略）

## DataCopyPad 配置

（CopyIn/CopyOut 的参数配置代码）

## 测试用例

（功能测试 + 边界测试矩阵）

## 常见问题

（问题 → 原因 → 解决方案表）
```

---

## algorithms.md 算法路由模板

```markdown
# {Category} 类算子核心算法路由

> ⚠️ 先看算法选择对照表确定适用算法，再按链接读取对应详细文档。

## 算法选择对照表

| 条件 | 推荐算法 | 原因 | 典型算子 | 详细文档 |
|------|---------|------|---------|---------|
| {条件1} | {算法1} | {原因} | {算子} | [alg-xxx.md](alg-xxx.md) |

## 算法摘要

### {算法 1}
（2-3 行摘要 + 链接）

### {算法 2}
...
```

---

## SKILL.md 分类表行模板

```markdown
| **{Category} {类别中文名}** | {特征描述} | {典型算子} | ✅ [场景路由](references/{category}/patterns.md)（⚠️ 必须先读） / [算法实现](references/{category}/algorithms.md) / [性能优化](references/{category}/optimizations.md) |
```

---

## Reduction 实例参照

最终形成的目录结构（22 个文件）：

| 文件 | 行数 | 定位 |
|------|------|------|
| patterns.md | 272 | 路由：决策树 + 3D 抽象 + 通用规则 + 分支摘要 |
| algorithms.md | 65 | 路由：算法选择表 + 摘要 |
| ar-fullload.md | 239 | 分支：AR FullLoad |
| ar-colsplit.md | 217 | 分支：AR ColSplit |
| ara-fullload.md | 261 | 分支：ARA 全载 |
| ara-rowsplit.md | 273 | 分支：ARA 分载 |
| multi-axis-transform.md | 145 | 多轴归约 Shape 变换 |
| alg-welford.md | 40 | 算法：Welford Online |
| alg-group-reduce.md | 101 | 算法：跨核归约 |
| alg-dichotomy.md | 50 | 算法：二分累加 / Half-Interval |
| multi-output-buffer.md | 115 | 多输出 Buffer 方程 |
| tiling-fields.md | 94 | Tiling 设计原则 + 字段 |
| optimizations.md | 537 | 性能优化模式 |
