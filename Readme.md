# Battery-Cycle-Life Prediction
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 使用指南

⸻

## 目录结构

```bash
final_code/model
├── dataloader_and_utils.py   # 数据加载、评估指标
├── featurgeneration.m        # MATLAB 特征工程脚本（可选）
├── model.py                  # ⭐ 主训练脚本（支持 CLI 手动调参）
├── model_autotrain.py        # 旧版一键训练脚本（保留）
├── plot_error_pic.py         # 绘制 Train/Test MPE 图
├── coeff_plotter.py        # 绘制特征重要度热图
└── results/                  # 训练完成后生成的模型与指标文件
```

注意：results/ 会在首次运行时自动创建，同步保存各模型的参数、评估指标以及旧版脚本需要的 _data.pkl。

⸻

## 环境依赖
	•	Python ≥ 3.8
	•	主要库：numpy, scikit-learn, pandas, matplotlib

# 建议使用 venv / conda
```bash
conda create -n battery python=3.10
conda activate battery
```

⸻

## 数据准备

项目假设已经存在如下 CSV：

```bash
training/cycles_2TO{N}_log.csv
testing/ cycles_2TO{N}_log.csv   # N ∈ {20,30…100}
```

每个文件包含：
	1.	battery_id（未使用）
	2.	cycle_lives（标签）
	3.	其它特征列（≥ 65 列）。

如需重新生成特征，可参考 featuregeneration.m。

⸻

## 快速开始

1. 默认全模型训练

```bash
python model.py            # 训练 ENet + RF + AB，采用默认参数网格
```

完成后，results/ 内将出现：

AB_trained_models.pkl
AB_training_percenterror.pkl
AB_crossvalid_percenterror.pkl
AB_data.pkl           # 兼容绘图脚本
RF_*.pkl  同理
enet_*.pkl 同理
___
2. 手动调参 & 选择模型

model.py 已加入 CLI：

```bash
python model.py \
       --model AB \
       --param n_estimators=500,1000 learning_rate=0.05
```

	•	--model：AB / RF / enet / all（默认）。
	•	--param：用 KEY=VALUE 指定超参；多个值用逗号分隔，将自动进行笛卡尔积组合。

示例：

### RF: 尝试深度 10 与 50；树数 300

```bash
python model.py --model RF --param n_estimators=300 max_depth=10,50
```

### ENet: 修改 l1_ratio 范围

```bash
python model.py --model enet --param l1_ratio=0.2,0.8
```

如同时想调多模型，可多次调用或设 --model all 并分别指定对应参数。

⸻

3. 加载全部特征

某些研究场景下你可能想让模型使用 **CSV 中的所有列**，而不是默认的 13 个特征。  
训练时可携带额外标志：

```bash
python model.py --model enet --param use_all_features=True
```

> `use_all_features` 可以和其他 `--param` 一起写；若缺省则保持原来子集。

⸻

## 可视化

1. 误差曲线（Fig 1）

   ```bash
   python plot_error_pic.py
   ```

   - 生成 `plots/all_error.png` / `.svg`：各模型 Train/Test MPE vs. N_cycles
   - `doFig1=False` 时改画观测 vs. 预测散点图

2. 特征重要度热图

   ```bash
   python coeff_plotter.py
   ```

   - 生成 `AB_features.png`, `RF_features.png`, `enet_features.png`
   - 图高及左侧留白自动随特征数调节，长标签不会再被裁切

⸻

## 自定义开发
	•	新增模型：继承 BaseBatteryModelTrainer，实现 _get_param_grid 与 _build_model，必要时覆盖 _predict 与 _save_results。
	•	更多特征：修改 WHICH_FEATURES 或在 CSV 中添加列；dataloader_and_utils.load_dataset 已支持 use_all_features=True。
	亦可在命令行通过 `--param use_all_features=True` 一键启用全部列。
	•	实验记录：results/ 按日期分目录或改写保存逻辑即可。

⸻

## 结果文件说明

| 文件 | 作用 |
|------|------|
| `AB_features_coeffs.pkl`, `RF_features_coeffs.pkl` | n_features × N_cycles 的特征重要度矩阵，供 `coeff_plotter.py` 读取 |

## 许可证

本项目采用 MIT 许可证，详情参见 [LICENSE](LICENSE) 文件。
