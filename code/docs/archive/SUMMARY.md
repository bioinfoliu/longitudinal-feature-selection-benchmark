# CTSNN框架全面改进总结

## 🎯 改进目标

2. ✅ 将K值改为可调参数
3. ✅ 改进SNN算法实现
4. ✅ 实现混合方法（SNN + 密度聚类）
5. ✅ 添加提升准确度的方法
6. ✅ 创建统一的配置和工具

## 📁 新增文件

### 核心模块
- **`src/snn_utils_improved.py`**: 改进的SNN工具模块
  - 改进的SNN权重计算
  - 密度聚类方法（DBSCAN, OPTICS）
  - 混合权重方法
  - K值自动选择
  - 患者一致性权重计算
  - 改进的特征选择方法

### 配置文件
- **`src/config_snn.py`**: 统一的配置管理
  - 全局配置
  - 数据集特定配置
  - 预设配置（5种预设）

### 文档
- **`IMPROVEMENTS.md`**: 详细改进说明
- **`SUMMARY.md`**: 本文件，改进总结

## 🔧 已更新的文件

### Notebooks
- **`src/COVID_SNN.ipynb`**: 已更新使用新模块
- **`src/PE_SNN.ipynb`**: 已更新使用新模块

## 🐛 Bug修复详情

### 1. K值参数化问题
**问题**：
- K值在不同文件中实现不一致
- 有些硬编码（如`n_neighbors=10`）
- 自动计算逻辑不统一

**修复**：
- 统一使用`SNNConfig`类管理参数
- K值可通过`k_snn`直接设置，或通过`k_ratio`自动计算
- 添加了`k_min`和`k_max`限制

### 2. SNN计算不一致
**问题**：
- 不同notebook中SNN计算方式有差异
- 归一化方法不统一
- 部分代码除以N而不是K

**修复**：
- 统一使用`compute_snn_weights_improved()`
- 修复了SNN矩阵计算bug
- 确保权重计算时除以正确的K值
- 添加多种归一化方法选项

### 3. 权重计算错误
**问题**：
- 部分代码在计算权重时除以N而不是K
- 归一化方法不一致

**修复**：
- 确保所有权重计算除以K值
- 统一的归一化流程

## ✨ 新功能

### 1. K值自动选择
```python
best_k, k_scores = select_optimal_k(
    df, 
    target_col='shannon_diversity',
    cv_folds=5
)
```
通过交叉验证自动选择最优K值，避免手动调参。

### 2. 混合权重方法
结合两种策略：
- **SNN**：用于患者一致性（temporal consistency）
- **密度聚类**：用于代表性样本选择（spatial representativeness）

```python
weights, info = compute_hybrid_weights(
    df,
    use_snn_for_consistency=True,
    use_density_for_representative=True,
    alpha=0.6  # 混合权重
)
```

### 3. 密度聚类方法
- **DBSCAN**：基于密度的聚类
- **OPTICS**：改进的密度聚类，适合不同密度的簇

核心样本（密度高的区域）权重高，噪声样本权重低。

### 4. 改进的归一化方法
- `minmax`：标准最小-最大归一化
- `zscore`：Z-score标准化
- `robust`：基于分位数的鲁棒归一化（适合有异常值）

### 5. 集成特征选择
结合多种特征选择方法：
- `union`：取并集
- `intersection`：取交集
- `weighted`：加权投票

## 📊 使用方法

### 快速开始

#### 方法1：使用默认配置
```python
from snn_utils_improved import SNNConfig, compute_snn_weights_improved

config = SNNConfig(k_snn=None)  # 自动选择K值
weights, k_used = compute_snn_weights_improved(df, config=config)
```

#### 方法2：使用配置文件
```python
from config_snn import get_config_for_dataset, USE_HYBRID_METHOD, HYBRID_ALPHA

config = get_config_for_dataset('covid')
weights, k_used = compute_snn_weights_improved(df, config=config)
```

#### 方法3：使用混合方法
```python
from snn_utils_improved import compute_hybrid_weights

weights, info = compute_hybrid_weights(
    df,
    use_snn_for_consistency=True,
    use_density_for_representative=True,
    alpha=0.6
)
```

#### 方法4：使用预设配置
```python
from config_snn import PRESET_BALANCED

config = PRESET_BALANCED['config']
use_hybrid = PRESET_BALANCED['use_hybrid']
alpha = PRESET_BALANCED['alpha']
```

## 🎛️ 参数调优指南

### K值选择
| 数据集大小 | 推荐K值范围 | 说明 |
|-----------|-----------|------|
| < 100样本 | 5-15 | 小数据集，K值不宜过大 |
| 100-1000样本 | 15-50 | 中等数据集，平衡局部和全局 |
| > 1000样本 | 30-100 | 大数据集，可以捕捉更大范围的流形 |

**建议**：先使用`AUTO_SELECT_K=True`自动选择，然后固定最优值。

### 混合权重alpha
| alpha值 | 策略 | 适用场景 |
|---------|------|---------|
| 0.8-1.0 | SNN主导 | 患者一致性非常重要 |
| 0.5-0.7 | 平衡 | **推荐**，大多数情况 |
| 0.0-0.4 | 密度主导 | 代表性样本选择更重要 |

### 归一化方法
- **minmax**：标准方法，适合大多数情况 ✅推荐
- **zscore**：适合数据分布接近正态分布
- **robust**：适合有异常值的情况

## 📈 预期改进效果

### 准确度提升
- **RMSE/MAE降低**：5-15%（通过混合方法和参数优化）
- **稳定性提升**：统一的实现减少结果波动
- **泛化能力**：更好的特征选择提升模型泛化

### 代码质量
- **可维护性**：统一的模块化设计
- **可扩展性**：易于添加新方法
- **可配置性**：所有参数可调

## 🔄 迁移指南

### 从旧代码迁移

1. **导入新模块**：
```python
# 旧代码
from sklearn.neighbors import NearestNeighbors
# ... 自己实现SNN

# 新代码
from snn_utils_improved import SNNConfig, compute_snn_weights_improved
```

2. **替换函数调用**：
```python
# 旧代码
weights = compute_snn_weights(df, k_auto=30)

# 新代码
config = SNNConfig(k_snn=30)
weights, k_used = compute_snn_weights_improved(df, config=config)
```

3. **更新配置**：
```python
# 在notebook开头添加
from config_snn import SNN_CONFIG_COVID, USE_HYBRID_METHOD, HYBRID_ALPHA
```

## ⚠️ 注意事项

1. **内存使用**：大数据集时，密度聚类可能消耗较多内存
2. **计算时间**：自动K值选择需要交叉验证，可能较慢（建议先运行一次找到最优K，然后关闭）
3. **参数敏感性**：不同数据集可能需要不同的参数设置
4. **数据格式**：必须包含'ID'列用于患者一致性计算

## 🚀 下一步建议

1. **实验不同配置**：
   - 尝试不同的alpha值（0.3, 0.5, 0.7, 0.9）
   - 比较纯SNN vs 混合方法
   - 测试不同的归一化方法

2. **K值优化**：
   - 使用`AUTO_SELECT_K=True`找到最优K
   - 对不同数据集分别优化

3. **特征选择改进**：
   - 尝试集成特征选择方法
   - 调整特征数量（top_n）

4. **模型集成**：
   - 结合不同权重方法的结果
   - 使用模型投票或加权平均

## 📞 问题排查

### 常见问题

1. **导入错误**：
   - 确保`src/snn_utils_improved.py`文件存在
   - 检查Python路径设置

2. **K值异常**：
   - 检查`k_min`和`k_max`设置
   - 确保K值不超过样本数-1

3. **内存不足**：
   - 减小数据集大小
   - 使用`use_density_for_representative=False`关闭密度方法

4. **结果不稳定**：
   - 设置随机种子
   - 增加交叉验证折数

## 📚 相关文档

- `IMPROVEMENTS.md`：详细的改进说明
- `src/snn_utils_improved.py`：代码注释和文档字符串
- `src/config_snn.py`：配置说明

## ✅ 完成清单

- [x] 修复K值参数化问题
- [x] 统一SNN实现
- [x] 添加K值自动选择
- [x] 实现混合权重方法
- [x] 添加密度聚类方法
- [x] 改进归一化方法
- [x] 创建统一配置
- [x] 更新主要notebook
- [x] 编写文档
- [x] 添加集成特征选择

## 🎉 总结

1. **统一和修复**：统一了SNN实现，修复了多个bug
2. **参数化**：K值和其他参数都可以轻松调整
3. **混合方法**：结合SNN和密度聚类，提升效果
4. **自动化**：K值自动选择，减少手动调参
5. **可配置**：统一的配置管理，易于实验

