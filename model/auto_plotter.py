#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dataloader_and_utils import load_dataset
from model import BaseBatteryModelTrainer
# Default features: use Severson et al. feature subset when not using all features
DEFAULT_WHICH_FEATURES = BaseBatteryModelTrainer.WHICH_FEATURES

def find_available_models(results_dir):
    """
    扫描 results_dir 下所有 *_data.pkl 或 *_trained_models.pkl
    自动提取模型名称列表（如 enet, RF, AB, pt 等）。
    """
    models = set()
    for fname in os.listdir(results_dir):
        if fname.endswith("_data.pkl") or fname.endswith("_trained_models.pkl") or fname.endswith("_training_percenterror.pkl"):
            # 取下划线前的模型名
            model = fname.split("_", 1)[0]
            models.add(model)
    # Enforce preferred plotting order: ElasticNet, RandomForest, AdaBoost, PyTorch
    preferred = ["enet", "RF", "AB", "pt"]
    # Filter only models found, in preferred order
    ordered = [m for m in preferred if m in models]
    # Append any other models sorted alphabetically
    extras = [m for m in sorted(models) if m not in ordered]
    return ordered + extras

def load_model_data(model, results_dir, use_log_features, use_all_features, which_features):
    """
    根据模型名加载数据，返回：
      predicted_cycle_lives, train_predicted_cycle_lives, train_mpe, min_mpe, test_mpe
    对于 enet，需要特殊从 trained_models.pkl + training_percenterror.pkl 读取并转换。
    """
    if model == "enet":
        # enet 的数据结构不一样：没有 _data.pkl，需要自己合并
        models_list = pickle.load(open(os.path.join(results_dir, f"{model}_trained_models.pkl"), "rb"))
        train_mpe = pickle.load(open(os.path.join(results_dir, f"{model}_training_percenterror.pkl"), "rb"))
        # 逐 cycle 预测，并计算 test_mpe
        N_cycles = np.array([20,30,40,50,60,70,80,90,100])
        test_mpe = np.zeros_like(N_cycles, dtype=float)
        predicted_full = None
        train_full = None
        for i, n in enumerate(N_cycles):
            test_path = f"testing/cycles_2TO{n}_log.csv"
            X_test, y_test, _ = load_dataset(test_path, use_log_features, use_all_features, which_features)
            X_train, y_train, _ = load_dataset(f"training/cycles_2TO{n}_log.csv", use_log_features, use_all_features, which_features)
            m = models_list[i]
            # Predict in log10 space and invert to linear scale
            y_pred_test_log = m.predict(X_test)
            y_pred_train_log = m.predict(X_train)
            # Convert log predictions back to actual cycle lives
            y_pred_test = 10 ** y_pred_test_log
            y_pred_train = 10 ** y_pred_train_log
            # Compute mean percent error on linear scale
            test_mpe[i] = np.mean(np.abs(y_pred_test - y_test) / y_test) * 100
            if n == N_cycles[-1]:
                predicted_full = y_pred_test
                train_full = y_pred_train
        return predicted_full, train_full, train_mpe, None, test_mpe

    else:
        # 其它模型直接从 *_data.pkl 读取五元组
        data = pickle.load(open(os.path.join(results_dir, f"{model}_data.pkl"), "rb"))
        return (data[0], data[1], data[2], data[3], data[4])

def main():
    parser = argparse.ArgumentParser(
        description="绘制多个模型在各循环阶段的预测效果（散点图或误差曲线）"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        help="要绘制的模型名称，默认全部自动发现",
        default=None
    )
    parser.add_argument(
        "--mode", choices=["scatter", "error"], default="scatter",
        help="scatter: 观测值 vs 预测值；error: 训练/测试 MPE vs 循环数"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="结果文件所在目录（默认为 ./results）"
    )
    parser.add_argument(
        "--log-features", action="store_true",
        help="Use log-transformed features when loading datasets"
    )
    parser.add_argument(
        "--all-features", action="store_true",
        help="Use all available features when loading datasets"
    )
    parser.add_argument(
        "--features", nargs="*", default=None,
        help="List of specific features to load (overrides --all-features)"
    )
    args = parser.parse_args()

    # Determine which features to use: default to Severson's subset if not --all-features and no --features specified
    if not args.all_features and not args.features:
        which_features = DEFAULT_WHICH_FEATURES
    else:
        which_features = args.features

    # 1. 确定要绘制的模型列表
    if args.models:
        models = args.models
    else:
        models = find_available_models(args.results_dir)
    if not models:
        print(f"未在 {args.results_dir} 中找到任何模型结果文件")
        return

    # 2. 读取每个模型的数据
    model_data = {}
    for model in models:
        try:
            model_data[model] = load_model_data(
                model, args.results_dir,
                args.log_features, args.all_features, which_features
            )
        except FileNotFoundError:
            print(f"警告：无法找到模型 {model} 对应的数据文件，跳过。")
    if not model_data:
        print("没有有效的模型数据，退出。")
        return

    # 3. 准备绘图
    n_models = len(model_data)
    plt.close("all")
    matplotlib.rcParams.update({'font.size': 18})
    # Create subplots: one row, one column per model
    f, ax_arr = plt.subplots(1, n_models, sharey=True, figsize=(20, 6))
    N_cycles = np.array([20,30,40,50,60,70,80,90,100])
    lettering_dict = {i: chr(97 + i) for i in range(n_models)}

    for idx, (model, data) in enumerate(model_data.items()):
        f.sca(ax_arr[idx])
        pred_full, train_full, train_mpe, min_mpe, test_mpe = data

        if args.mode == "scatter":
            # Load actual observed cycle lives for final cycle
            final_cycle = N_cycles[-1]
            test_file = f"testing/cycles_2TO{final_cycle}_log.csv"
            train_file = f"training/cycles_2TO{final_cycle}_log.csv"
            _, cycle_lives, _ = load_dataset(test_file, args.log_features, args.all_features, which_features)
            _, train_cycle_lives, _ = load_dataset(train_file, args.log_features, args.all_features, which_features)

            # Invert log-transform for ElasticNet results
            if model == "enet":
                pred_full = 10 ** pred_full
                train_full = 10 ** train_full

            # diagonal line plot
            plt.plot(train_cycle_lives, train_full, 'rs', label='Train')
            plt.plot(cycle_lives, pred_full, 'bo', label='Test')
            max_lim = max(max(train_cycle_lives), max(cycle_lives))
            plt.plot([0, max_lim], [0, max_lim], 'k--')
            plt.xlabel('Observed cycle life')
            if idx == 0:
                plt.ylabel('Predicted cycle life')
            plt.xticks(np.arange(0, max_lim+1, 500))
            plt.yticks(np.arange(0, max_lim+1, 500))
            plt.axis([0, max_lim, 0, max_lim])
            plt.legend()
            ax_arr[idx].set_aspect('equal', 'box')
            ax_arr[idx].set_title(lettering_dict[idx], loc='left')

        else:  # error 模式
            ax_arr[idx].plot(N_cycles, train_mpe, '-o', label="Train")
            ax_arr[idx].plot(N_cycles, test_mpe, '-o', label="Test")
            ax_arr[idx].set_xlabel("Cycle number")
            ax_arr[idx].set_ylabel("Mean percent error (%)")
            ax_arr[idx].set_title(model)
            ax_arr[idx].set_ylim(0, max(test_mpe.max(), train_mpe.max())*1.1)
            ax_arr[idx].legend()

    f.tight_layout()
    out_name = f"all_models_{args.mode}.png"
    f.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"已保存：{out_name}")

if __name__ == "__main__":
    main()