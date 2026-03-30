# ops-math / ops-nn Reduction 算子代码索引

> 查阅参考实现时的快速导航。列出所有 Reduction 类算子的目录、kernel 文件、tiling 文件路径。

---

## ops-math 基础归约算子

### 有 op_kernel + op_host 的算子（可参考完整实现）

| 算子 | 目录 | Kernel 入口 | Tiling 文件 | 算法特点 |
|------|------|------------|------------|---------|
| **reduce_sum** | `ops-math/math/reduce_sum/` | `op_kernel/reduce_sum_apt.cpp` | `op_host/arch35/reduce_sum_tiling_arch35.cpp` | DAG(A5) + 类型提升 |
| **reduce_max** | `ops-math/math/reduce_max/` | `op_kernel/reduce_max_apt.cpp` | `op_host/arch35/reduce_max_tiling_arch35.cpp` | DAG(A5) + dumpValue初始化 |
| **reduce_min** | `ops-math/math/reduce_min/` | `op_kernel/reduce_min_apt.cpp` | `op_host/arch35/reduce_min_tiling_arch35.cpp` | DAG(A5) + dumpValue初始化 |
| **reduce_mean** | `ops-math/math/reduce_mean/` | `op_kernel/reduce_mean_apt.cpp` | `op_host/arch35/reduce_mean_tiling_arch35.cpp` | DAG(A5) + Muls(1/N) |
| **reduce_prod** | `ops-math/math/reduce_prod/` | `op_kernel/reduce_prod_apt.cpp` | `op_host/arch35/reduce_prod_tiling_arch35.cpp` | DAG(A5) + ReduceProdOp |
| **reduce_var** | `ops-math/math/reduce_var/` | `op_kernel/reduce_var_apt.cpp` | `op_host/arch35/reduce_var_tiling.cpp` | **命令式 Welford/TwoPass** |
| **reduce_std_v2** | `ops-math/math/reduce_std_v2/` | `op_kernel/` | `op_host/arch35/` | 类似 reduce_var + sqrt |
| **reduce_all** | `ops-math/math/reduce_all/` | `op_kernel/` | `op_host/` | 布尔 AND 归约 |
| **reduce_any** | `ops-math/math/reduce_any/` | `op_kernel/` | `op_host/` | 布尔 OR 归约 |
| **reduce_log_sum_exp** | `ops-math/math/reduce_log_sum_exp/` | `op_kernel/` | `op_host/` | log(sum(exp(x))) |
| **arg_max_v2** | `ops-math/math/arg_max_v2/` | `op_kernel/arg_max_v2_apt.cpp` | `op_host/arch35/arg_max_v2_tiling.cpp` | **命令式 8种策略** |
| **arg_max_with_value** | `ops-math/math/arg_max_with_value/` | `op_kernel/` | `op_host/arch35/arg_common_base_tiling.cpp` | 双输出(value+index) |
| **arg_min** | `ops-math/math/arg_min/` | `op_kernel/` | `op_host/arch35/` | 类似 arg_max |
| **arg_min_with_value** | `ops-math/math/arg_min_with_value/` | `op_kernel/` | `op_host/arch35/` | 双输出 |

### 仅 op_host 的算子（Host 侧调度，复用其他 kernel）

| 算子 | 目录 | 说明 |
|------|------|------|
| reduce_nansum | `ops-math/math/reduce_nansum/` | NaN-safe sum |
| reduce_log_sum | `ops-math/math/reduce_log_sum/` | log(sum(x)) |
| reduce_std_with_mean | `ops-math/math/reduce_std_with_mean/` | 同时输出 std+mean |
| reduce_std_v2_update | `ops-math/math/reduce_std_v2_update/` | std 增量更新 |
| reduce_mean_with_count | `ops-math/math/reduce_mean_with_count/` | mean + count |

### 实验版 v2（探索新范式）

| 算子 | 目录 | 特点 |
|------|------|------|
| **reduce_sum_v2** | `ops-math/experimental/math/reduce_sum_v2/` | **命令式 + Atomic，适合 A2/A3 参考** |
| reduce_max_v2 | `ops-math/experimental/math/reduce_max_v2/` | 命令式 v2 |
| reduce_min_v2 | `ops-math/experimental/math/reduce_min_v2/` | 命令式 v2 |
| reduce_mean_v2 | `ops-math/experimental/math/reduce_mean_v2/` | 命令式 v2 |
| reduce_prod_v2 | `ops-math/experimental/math/reduce_prod_v2/` | 命令式 v2 |

---

## ops-nn 归一化算子

| 算子 | 目录 | Kernel 入口 | Tiling 文件 | 算法特点 |
|------|------|------------|------------|---------|
| **rms_norm** | `ops-nn/norm/rms_norm/` | `op_kernel/rms_norm.cpp` | `op_host/rms_norm_tiling.cpp` | **5 种模式 + Half-Interval 归约** |
| **layer_norm_v4** | `ops-nn/norm/layer_norm_v4/` | `op_kernel/layer_norm_v4.cpp` | `op_host/layer_norm_v4_*_tiling.cpp` | **9 种策略路径** |
| add_rms_norm | `ops-nn/norm/add_rms_norm/` | `op_kernel/` | `op_host/` | x+residual → RmsNorm |
| batch_norm_v3 | `ops-nn/norm/batch_norm_v3/` | `op_kernel/` | `op_host/` | 批归一化 |
| group_norm_v2 | `ops-nn/norm/group_norm_v2/` | `op_kernel/` | `op_host/` | 分组归一化 |
| instance_norm_v3 | `ops-nn/norm/instance_norm_v3/` | `op_kernel/` | `op_host/` | 实例归一化 |
| ada_layer_norm | `ops-nn/norm/ada_layer_norm/` | `op_kernel/` | `op_host/` | 自适应 LayerNorm |

---

## ops-nn 激活/Softmax 算子

| 算子 | 目录 | Kernel 入口 | Tiling 文件 | 算法特点 |
|------|------|------------|------------|---------|
| **softmax_v2** | `ops-nn/activation/softmax_v2/` | `op_kernel/softmax_v2_apt.cpp` | `op_host/arch35/softmax_v2_*_tiling.cpp` | **5 种策略 + Recompute** |
| log_softmax | `ops-nn/activation/log_softmax/` | `op_kernel/` | `op_host/` | log(softmax(x)) |

---

## 重点参考文件

### reduce_var（最复杂的统计归约，Welford + TwoPass + Group Reduce 全套）

```
op_kernel/
├── reduce_var_apt.cpp                    ← 入口分发
├── arch35/reduce_var_sch.h               ← 主调度器 (1256行)
├── arch35/reduce_var_welford.h           ← Welford 算法 (1698行)
├── arch35/reduce_var_twopass.h           ← TwoPass 算法 (615行)
├── arch35/reduce_var_vf_common.h         ← 公共工具(Cast/DichotomyAdd)
├── arch35/reduce_var_empty.h             ← 空张量处理
└── arch35/reduce_var_pure_move.h         ← 纯搬运模式

op_host/
├── arch35/reduce_var_tiling.h            ← Tiling 数据结构
└── arch35/reduce_var_tiling.cpp          ← Tiling 策略计算 (1159行)
```

### arg_max_v2（最复杂的索引归约，8种切分策略）

```
op_kernel/
└── arg_max_v2_apt.cpp                    ← 入口 + 8种 TILING_KEY 分发

op_host/
├── arch35/arg_max_v2_tiling.cpp          ← Tiling 注册
├── arch35/arg_max_v2_tiling_arch35.cpp   ← Tiling 实现类
└── (共享) arg_max_with_value/op_host/arch35/
    ├── arg_common_base_tiling.h          ← 公共数据结构
    ├── arg_common_base_tiling.cpp        ← 策略选择
    └── arg_common_base_tiling_arch35.cpp ← 完整切分算法
```

### rms_norm（归一化范式代表，5种模式）

```
op_kernel/
├── rms_norm.cpp                          ← 入口
├── rms_norm.h                            ← Mode 0: Normal
├── rms_norm_split_d.h                    ← Mode 1: SplitD (列切片)
├── rms_norm_merge_n.h                    ← Mode 2: MergeN (多行打包)
├── rms_norm_single_row.h                 ← Mode 3: SingleRow
├── rms_norm_whole_reduce_sum.h           ← Mode 4: WholeReduceSum
├── rms_norm_base.h                       ← 基类 + 常量
├── reduce_common.h                       ← ReduceSumHalfInterval 算法
└── arch35/rms_norm_regbase*.h            ← Arch35 寄存器优化

op_host/
├── rms_norm_tiling.h                     ← Tiling 数据结构
├── rms_norm_tiling.cpp                   ← 模式选择逻辑
└── rms_norm_tiling_arch35.cpp            ← Arch35 tiling
```

### softmax_v2（复合归约范式代表，5种策略 + Recompute）

```
op_kernel/arch35/
├── softmax_v2_base.h                     ← 公共工具(Dichotomy/Cast)
├── softmax_v2_ar_small_r.h              ← AR 小R (转置优化)
├── softmax_v2_ar_full_load.h            ← AR 全载 (最快)
├── softmax_v2_ar_recompute.h            ← AR 重计算 (省空间)
├── softmax_v2_ara_full_load.h           ← ARA 全载
└── softmax_v2_ara_recompute.h           ← ARA 重计算

op_host/arch35/
├── softmax_v2_tiling.h                   ← Tiling 数据结构 + 优先级
├── softmax_v2_base_tiling.cpp            ← 公共准备
├── softmax_v2_ar_small_r_tiling.cpp
├── softmax_v2_ar_full_load_tiling.cpp
├── softmax_v2_ar_recompute_tiling.cpp
├── softmax_v2_ara_full_load_tiling.cpp
└── softmax_v2_ara_recompute_tiling.cpp
```

### reduce_sum_v2（命令式 + Atomic 参考，适合 A2/A3）

```
op_kernel/
├── reduce_sum_v2.h                       ← 完整命令式实现
├── reduce_sum_v2.cpp                     ← 入口
├── reduce_sum_v2_tiling_data.h           ← Tiling 数据结构
└── reduce_sum_v2_tiling_key.h            ← Key 定义

op_host/
├── reduce_sum_v2_tiling.cpp              ← Tiling 计算
├── reduce_sum_v2_def.cpp                 ← 算子定义
└── reduce_sum_v2_infershape.cpp          ← Shape 推导
```

---

## 共享框架（A5 DAG 专用）

```
opbase/pkg_inc/op_common/atvoss/reduce/
├── reduce_sch.h              ← ReduceSch 调度器
├── reduce_operator.h         ← 归约操作定义 (SUM/MAX/MIN/PROD)
├── reduce_tiling.h           ← Tiling4ReduceOp + GEN_REDUCE_TILING_KEY
├── reduce_tiling_data.h      ← ReduceOpTilingData 结构体
├── reduce_tiling_key_decl.h  ← Pattern ID 声明
├── reduce_tiling_key_sel.h   ← 模板选择器
└── reduce_util.h             ← ReduceOpTmpl (Pattern枚举/常量/GetPromoteType)
```
