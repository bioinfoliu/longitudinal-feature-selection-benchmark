# CTSNN框架改进说明

## 📋 改进概览

本次改进全面优化了SNN算法框架，修复了多个bug，并添加了多种提升准确度的方法。

## 🐛 Bug修复

### 1. K值参数化问题
- **问题**：K值在不同文件中实现不一致，有些硬编码，有些自动计算但逻辑不统一
- **修复**：
  - 统一使用`SNNConfig`配置类管理所有参数
  - K值可通过`k_snn`参数直接设置，或通过`k_ratio`自动计算
  - 添加了K值范围限制（`k_min`, `k_max`）防止异常值

### 2. SNN计算不一致
- **问题**：不同notebook中SNN计算方式略有差异，归一化方法不统一
- **修复**：
  - 统一使用`compute_snn_weights_improved()`函数
  - 修复了SNN矩阵计算中的bug（确保只计算K近邻范围内的SNN值）
  - 添加了多种归一化方法选项（minmax, zscore, robust）

### 3. 权重计算错误
- **问题**：部分代码在计算权重时除以N而不是K
- **修复**：确保权重计算时除以正确的K值

## ✨ 新功能

### 1. K值自动选择
```python
# 通过交叉验证自动选择最优K值
best_k, k_scores = select_optimal_k(
    df, 
    target_col='shannon_diversity',
    cv_folds=5
)
```

### 2. 混合权重方法
- **SNN用于患者一致性**：检测同一患者样本之间的相似性
- **密度聚类用于代表性样本**：使用DBSCAN/OPTICS识别数据密集区域的核心样本

```python
# 使用混合方法
weights, info = compute_hybrid_weights(
    df,
    use_snn_for_consistency=True,      # SNN用于患者一致性
    use_density_for_representative=True,  # 密度方法用于代表性样本
    alpha=0.6  # 混合权重
)
```

### 3. 密度聚类方法
- **DBSCAN**：基于密度的聚类，识别核心样本和噪声样本
- **OPTICS**：改进的密度聚类，适合不同密度的簇

### 4. 改进的归一化方法
- `minmax`：标准最小-最大归一化
- `zscore`：Z-score标准化
- `robust`：基于分位数的鲁棒归一化

## 🎯 提升准确度的方法

### 1. 可调参数
所有关键参数都可以轻松调整：

```python
SNN_CONFIG = SNNConfig(
    k_snn=None,  # 手动设置K值，或None自动计算
    k_min=5,     # K值最小值
    k_max=100,   # K值最大值
    k_ratio=0.3, # 自动K值计算比例
    use_density_for_representative=True,
    density_method='dbscan',  # 'dbscan' 或 'optics'
    normalize_method='minmax',
    add_small_epsilon=True,
    epsilon=1e-5
)
```

### 2. 混合方法策略
- **策略1**：SNN用于患者一致性（temporal consistency）
- **策略2**：密度方法用于代表性样本选择（spatial representativeness）
- **融合**：通过加权平均或乘法融合两种权重

### 3. 患者一致性权重（改进的STAR方法）
专门用于检测同一患者样本之间的时间一致性：

```python
weights = compute_patient_consistency_weights(df, k_neighbors=None)
```

### 4. 特征选择改进
- 权重在特征选择阶段使用（Random Forest, Lasso）
- 模型训练阶段不使用权重（保证泛化性）

## 📝 使用指南

### 快速开始

1. **使用默认配置（推荐）**）：
```python
from snn_utils_improved import SNNConfig, compute_snn_weights_improved

config = SNNConfig(k_snn=None)  # 自动选择K值
weights, k_used = compute_snn_weights_improved(df, config=config)
```

2. **手动设置K值**：
```python
config = SNNConfig(k_snn=30)  # 使用K=30
weights, k_used = compute_snn_weights_improved(df, config=config)
```

3. **使用混合方法**：
```python
from snn_utils_improved import compute_hybrid_weights

weights, info = compute_hybrid_weights(
    df,
    use_snn_for_consistency=True,
    use_density_for_representative=True,
    alpha=0.6
)
```

### 参数调优建议

1. **K值选择**：
   - 小数据集（<100样本）：K=5-15
   - 中等数据集（100-1000样本）：K=15-50
   - 大数据集（>1000样本）：K=30-100
   - 建议先使用`AUTO_SELECT_K=True`自动选择，然后固定最优值

2. **混合权重alpha**：
   - `alpha=0.8-1.0`：更依赖SNN（适合患者一致性重要的情况）
   - `alpha=0.5-0.7`：平衡（推荐）
   - `alpha=0.0-0.4`：更依赖密度方法（适合代表性样本重要的情况）

3. **归一化方法**：
   - `minmax`：标准方法，适合大多数情况
   - `zscore`：适合数据分布接近正态分布
   - `robust`：适合有异常值的情况

## 🔧 代码结构

```
src/
├── snn_utils_improved.py    # 改进的工具模块（核心）
├── COVID_SNN.ipynb          # 已更新使用新模块
├── PE_SNN.ipynb             # 已更新使用新模块
└── ...
```

## 📊 实验建议

1. **基线实验**：使用原始SNN方法（`USE_HYBRID_METHOD=False`）
2. **混合方法实验**：使用混合方法（`USE_HYBRID_METHOD=True`）
3. **K值调优**：先运行`AUTO_SELECT_K=True`找到最优K，然后固定
4. **参数网格搜索**：尝试不同的`alpha`值（0.3, 0.5, 0.7, 0.9）

## 🚀 性能优化

- 使用稀疏矩阵优化内存使用
- 支持并行计算（n_jobs参数）
- 缓存K值选择结果避免重复计算

## ⚠️ 注意事项

1. **内存使用**：大数据集时，密度聚类可能消耗较多内存
2. **计算时间**：自动K值选择需要交叉验证，可能较慢
3. **参数敏感性**：不同数据集可能需要不同的参数设置

## 📈 预期改进

- **准确度提升**：通过混合方法和参数优化，预期RMSE/MAE可降低5-15%
- **稳定性提升**：统一的实现和bug修复提高结果稳定性
- **可解释性提升**：清晰的参数配置和文档

## 🔄 迁移指南

### 从旧代码迁移

1. 导入新模块：
```python
from snn_utils_improved import SNNConfig, compute_snn_weights_improved
```

2. 替换旧的`compute_snn_weights`调用：
```python
# 旧代码
weights = compute_snn_weights(df, k_auto=30)

# 新代码
config = SNNConfig(k_snn=30)
weights, k_used = compute_snn_weights_improved(df, config=config)
```

3. 更新配置参数：根据数据集特点调整`SNN_CONFIG`

## 📞 问题反馈

如遇到问题，请检查：
1. 是否正确导入了`snn_utils_improved`模块
2. 数据格式是否正确（必须包含'ID'列用于患者一致性）
3. K值是否在合理范围内（k_min <= k <= k_max）
