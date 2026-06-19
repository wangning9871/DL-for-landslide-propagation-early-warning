import os, random, numpy as np, pandas as pd
from typing import List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, cohen_kappa_score,
    mean_absolute_error, classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler

# ============ 1. 基本配置 (CNN-LSTM) ============
SEED = 39
CSV_PATH = "HK_final_inputs_with_class.csv"
# --- K-Fold 划分修改 ---
CV_POOL_RATIO = 0.80
TEST_RATIO = 0.20
K_FOLDS = 5
# ----------------------
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_EVAL = 256
MAX_EPOCHS = 200
EARLY_MONITOR = "qwk"
PATIENCE = 50
MIN_DELTA = 1e-4
# --- 全局固定参数 ---
MLP_DROPOUT = 0.3
CNN_DROPOUT = 0.3
FINAL_DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
# -----------------------------
USE_CLASS_WEIGHTS = True
LAMBDA_COST = 0.5
DIST_POWER = 1
# 注意：文件名已修改
SAVE_BEST_PATH = "best_cnn_lstm_kfold_model.txt"
RESULTS_CSV = "grid_results_cnn_lstm_kfold.csv"
PLOT_DATA_FILE = "plotting_data_cnn_lstm_kfold.txt"
# 新增文件: 记录所有配置在独立测试集上的详细绘图数据
ALL_TEST_PLOT_DATA_FILE = "all_test_plotting_data_cnn_lstm_kfold.txt"

# ============ 2. GRID_CONFIGS (CNN-LSTM 配置) ============
# 针对 CNN-LSTM 模型，我们只使用时序数据
GRID_CONFIGS = [

    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [2, 4], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [4, 8], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [8, 16], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [12, 24], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [16, 32], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [20, 40], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [24, 48], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [28, 56], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [32, 64], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [40, 80], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [48, 96], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [64, 128], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [84, 168], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [100, 200], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [120, 240], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [128, 256], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},
    {"widths": [64, 32], "act_mlp": "gelu", "cnn_layers": [164, 328], "kernel_size": 3, "act_cnn": "gelu",
     "lstm_hidden_size": 32, "lstm_layers": 2, "optimizer": "adamw", "lr": 1e-4},


]


# ============ 3. 设置随机种子和设备 (无需更改) ============
def set_seed(seed=SEED):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============ 4. 数据准备 (核心修改：80/20 划分，现在包含静态数据) ============

def get_data_and_test_set(csv_path, seed, cv_ratio, ts_ratio):
    df = pd.read_csv(csv_path)
    LABEL_COL = 'class_simple'
    y = df[LABEL_COL].values.astype(np.int64)

    # 读取静态数据和时序数据
    X_static_all = df.drop(LABEL_COL, axis=1).iloc[:, :3].values.astype(np.float32)
    X_temporal_flat = df.drop(LABEL_COL, axis=1).iloc[:, 3:].values.astype(np.float32)
    X_temporal_all = X_temporal_flat.reshape(-1, 3, 10)  # 假设 (N, C, L) 格式

    NUM_CLASSES = len(np.unique(y))
    CLASS_NAMES = [str(i) for i in sorted(np.unique(y))]

    # 第一次划分：分离 80% 的 CV Pool 和 20% 的独立测试集 (联合划分静态和时序)
    X_s_cv, X_s_test, X_t_cv, X_t_test, y_cv, y_test = train_test_split(
        X_static_all, X_temporal_all, y, test_size=ts_ratio, stratify=y, random_state=seed
    )

    # 数据标准化 (基于 CV Pool 拟合)
    # 静态数据标准化
    scaler_static = StandardScaler().fit(X_s_cv)
    X_s_cv, X_s_test = scaler_static.transform(X_s_cv), scaler_static.transform(X_s_test)
    NUM_STATIC_FEATS = X_s_cv.shape[1]

    # 时序数据标准化
    scaler_temporal = StandardScaler().fit(X_t_cv.reshape(X_t_cv.shape[0] * X_t_cv.shape[1], -1))

    def scale_data(X):
        X_shape = X.shape
        X_scaled = scaler_temporal.transform(X.reshape(X_shape[0] * X_shape[1], -1))
        return X_scaled.reshape(X_shape)

    X_t_cv, X_t_test = scale_data(X_t_cv), scale_data(X_t_test)

    NUM_INPUT_CHANNELS = X_t_cv.shape[1]  # C=3

    print(
        f"数据准备完成。CV Pool: {X_t_cv.shape[0]} 样本 ({cv_ratio * 100:.0f}%) | 独立测试集: {X_t_test.shape[0]} 样本 ({ts_ratio * 100:.0f}%)")
    print(f"CNN 输入通道数: {NUM_INPUT_CHANNELS} | 静态特征数: {NUM_STATIC_FEATS}")

    # 计算类别权重 (基于整个 CV Pool)
    class_weights_t = None
    if USE_CLASS_WEIGHTS:
        binc = np.bincount(y_cv, minlength=NUM_CLASSES).astype(np.float32)
        w = 1.0 / np.clip(binc, 1.0, None)
        class_weights_t = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    def to_dataset(X_static, X_temporal, y):
        # 静态数据, 时序数据, 标签
        return TensorDataset(torch.tensor(X_static), torch.tensor(X_temporal), torch.tensor(y))

    # 返回 CV Pool 数据集和独立测试集
    cv_dataset = to_dataset(X_s_cv, X_t_cv, y_cv)
    test_loader = DataLoader(to_dataset(X_s_test, X_t_test, y_test), batch_size=BATCH_SIZE_EVAL, shuffle=False)

    # 返回 CNN 输入通道数和静态特征数
    return cv_dataset, test_loader, class_weights_t, NUM_CLASSES, CLASS_NAMES, NUM_INPUT_CHANNELS, NUM_STATIC_FEATS


# 替换原始的调用，新增接收静态特征数
cv_dataset, test_loader, class_weights_t, NUM_CLASSES, CLASS_NAMES, NUM_INPUT_CHANNELS, NUM_STATIC_FEATS = get_data_and_test_set(
    CSV_PATH, SEED, CV_POOL_RATIO, TEST_RATIO
)


# ============ 5. 核心修改: CNN-LSTM 模型定义 (MLP 静态分支 + 融合) ============
def make_activation(name: str):
    name = name.lower();
    if name == "relu": return nn.ReLU();
    if name == "gelu": return nn.GELU();
    raise ValueError("未知的激活函数: " + name)


class CNN_LSTM_Model(nn.Module):
    # num_input_channels 是时序数据的通道数 (C=3)
    def __init__(self, num_input_channels, num_classes,
                 cnn_layers, kernel_size, act_cnn,
                 lstm_hidden_size, lstm_layers, final_dropout,
                 widths, act_mlp,  # <--- 接收 ANN 参数
                 mlp_out_features=32, fusion_hidden_features=64):  # <--- 接收 ANN 参数
        super().__init__()

        # --- 1. MLP 静态提取器 (模仿 Hybrid 模型) ---
        mlp_act_fn = make_activation(act_mlp)
        mlp_layers_list = [];
        in_dim = NUM_STATIC_FEATS  # 使用全局变量

        for out_dim in widths:
            mlp_layers_list.append(nn.Linear(in_dim, out_dim));
            mlp_layers_list.append(nn.BatchNorm1d(out_dim))
            mlp_layers_list.append(mlp_act_fn);
            mlp_layers_list.append(nn.Dropout(MLP_DROPOUT));
            in_dim = out_dim

        # 最后的输出层 (固定为 mlp_out_features，方便融合)
        mlp_layers_list.append(nn.Linear(in_dim, mlp_out_features));
        mlp_layers_list.append(nn.ReLU())
        self.mlp_extractor = nn.Sequential(*mlp_layers_list)

        # --- 2. CNN/LSTM 序列总结器 ---
        cnn_blocks = []
        in_channels = num_input_channels
        for out_channels in cnn_layers:
            cnn_blocks.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
            )
            cnn_blocks.append(nn.BatchNorm1d(out_channels))
            cnn_blocks.append(make_activation(act_cnn))
            cnn_blocks.append(nn.Dropout(CNN_DROPOUT))
            in_channels = out_channels
        self.cnn_extractor = nn.Sequential(*cnn_blocks)

        self.lstm = nn.LSTM(
            input_size=in_channels, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
            batch_first=True, bidirectional=False,
            dropout=CNN_DROPOUT if lstm_layers > 1 else 0
        )

        # --- 3. 融合分类器 ---
        combined_features_dim = lstm_hidden_size + mlp_out_features  # <--- 调整融合维度

        # 仿照 Hybrid 模型增加一层融合（使用 fusion_hidden_features）
        self.classifier = nn.Sequential(
            nn.Linear(combined_features_dim, fusion_hidden_features),
            nn.BatchNorm1d(fusion_hidden_features),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            nn.Linear(fusion_hidden_features, num_classes)
        )

    # 适配之前的 get_preds_and_probs(x_static, x_temporal)
    def forward(self, x_static, x_temporal):
        # 静态分支：MLP 提取
        static_summary = self.mlp_extractor(x_static)  # (N, mlp_out_features)

        # 时序分支：CNN + LSTM 提取
        cnn_features = self.cnn_extractor(x_temporal)
        lstm_input = cnn_features.permute(0, 2, 1)
        lstm_output, _ = self.lstm(lstm_input)
        temporal_summary = lstm_output[:, -1, :]  # (N, lstm_hidden_size)

        # 拼接静态特征和时序特征
        combined_features = torch.cat([static_summary, temporal_summary], dim=1)

        logits = self.classifier(combined_features)
        return logits


# ============ 6. 损失/评估/训练函数 (适配 K-Fold) ============
class CostSensitiveCE(nn.Module):
    def __init__(self, num_classes, lam, class_weights, power, device):
        super().__init__();
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        J = torch.arange(num_classes, dtype=torch.float32, device=device);
        self.C = (J[None, :] - J[:, None]).abs() ** power;
        self.lam = lam

    def forward(self, logits, y_true):
        ce = self.ce(logits, y_true);
        p = F.softmax(logits, dim=1)
        C_rows = self.C[y_true];
        ec = (p * C_rows).sum(dim=1).mean();
        return ce + self.lam * ec


def build_optimizer(model, config):
    name = config["optimizer"].lower();
    lr = config["lr"];
    wd = WEIGHT_DECAY
    if name == "adamw": return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam": return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"未知的优化器: {name}")


@torch.no_grad()
def get_preds_and_probs(model, loader, device):
    model.eval();
    logits_all, y_all = [], [];
    with torch.no_grad():
        for x_s, x_t, yb in loader:
            logits_all.append(model(x_s.to(device), x_t.to(device)).cpu());
            y_all.append(yb)
    y_true_np = torch.cat(y_all).numpy();
    logits_tensor = torch.cat(logits_all)
    probs_tensor = F.softmax(logits_tensor, dim=1);
    return y_true_np, probs_tensor.numpy(), logits_tensor.numpy()


# --- 修改 calculate_metrics 以支持前缀 ---
def calculate_metrics(y_true_np, probs_np, logits_np, num_classes, prefix=""):
    preds_np = np.argmax(logits_np, 1);
    y_binarized = label_binarize(y_true_np, classes=range(num_classes))

    if y_binarized.shape[1] > 1 and len(np.unique(y_true_np)) > 1:
        roc_auc = roc_auc_score(y_binarized, probs_np, average='macro', multi_class='ovr')
    elif y_binarized.shape[1] == 1 and len(np.unique(y_true_np)) > 1:
        roc_auc = roc_auc_score(y_true_np, probs_np[:, 1])
    else:
        roc_auc = 0.0

    metrics = {f"{prefix}qwk": cohen_kappa_score(y_true_np, preds_np, weights="quadratic"),
               f"{prefix}mae": mean_absolute_error(y_true_np, preds_np),
               f"{prefix}macro_f1": f1_score(y_true_np, preds_np, average="macro", zero_division=0),
               f"{prefix}bacc": balanced_accuracy_score(y_true_np, preds_np),
               f"{prefix}roc_auc": roc_auc}

    # 只为最终报告准备完整指标
    if not prefix:
        metrics["cm"] = confusion_matrix(y_true_np, preds_np)
        metrics["report"] = classification_report(y_true_np, preds_np, digits=3, target_names=CLASS_NAMES,
                                                  zero_division=0)

    return metrics

# --- 核心修改 1: 修改 train_and_eval_fold 传递所有参数 ---
def train_and_eval_fold(config, train_indices, val_indices, k_fold_id):
    """(在 K-Fold 的一个折上进行训练和评估，返回训练和验证指标)"""
    set_seed(SEED + k_fold_id)

    # 1. 划分当前 Fold 的训练和验证数据
    train_subset = Subset(cv_dataset, train_indices)
    val_subset = Subset(cv_dataset, val_indices)

    # 2. 创建 DataLoader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE_EVAL, shuffle=False)

    # 3. 实例化模型 (传递所有 ANN 参数)
    model = CNN_LSTM_Model(
        num_input_channels=NUM_INPUT_CHANNELS, num_classes=NUM_CLASSES,
        cnn_layers=config["cnn_layers"], kernel_size=config["kernel_size"], act_cnn=config["act_cnn"],
        lstm_hidden_size=config["lstm_hidden_size"], lstm_layers=config["lstm_layers"],
        final_dropout=FINAL_DROPOUT,
        widths=config["widths"], act_mlp=config["act_mlp"], # <--- 新增 ANN 参数
    ).to(device)

    criterion = CostSensitiveCE(NUM_CLASSES, LAMBDA_COST, class_weights_t, DIST_POWER, device)
    optimizer = build_optimizer(model, config)

    # 4. 训练循环
    best_score = -1e9 if EARLY_MONITOR == "qwk" else 1e9
    best_state, patience_left = None, PATIENCE

    best_train_metrics, best_val_metrics = None, None

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for x_s_b, x_t_b, y_b in train_loader:
            x_s_b, x_t_b, y_b = x_s_b.to(device), x_t_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_s_b, x_t_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # 验证集评估
        # 为了高效评估，我们使用较大的 BATCH_SIZE_EVAL，但 train_loader 默认使用 BATCH_SIZE_TRAIN
        val_loader_eval = DataLoader(val_subset, batch_size=BATCH_SIZE_EVAL, shuffle=False)

        y_val_true, _, val_logits = get_preds_and_probs(model, val_loader_eval, device)
        val_metrics = calculate_metrics(y_val_true, F.softmax(torch.from_numpy(val_logits), dim=1).numpy(), val_logits,
                                        NUM_CLASSES, prefix="val_")
        monitor_val = val_metrics[f"val_{EARLY_MONITOR}"]

        is_better = (monitor_val - best_score) > MIN_DELTA if EARLY_MONITOR == "qwk" else (
                                                                                                  best_score - monitor_val) > MIN_DELTA
        if best_state is None or is_better:
            best_score, best_state, patience_left = monitor_val, {k: v.cpu().clone() for k, v in
                                                                  model.state_dict().items()}, PATIENCE

            # --- 额外: 在找到更好的验证模型时，评估训练集性能 ---
            train_loader_eval = DataLoader(train_subset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
            y_train_true, _, train_logits = get_preds_and_probs(model, train_loader_eval, device)
            train_metrics = calculate_metrics(y_train_true, F.softmax(torch.from_numpy(train_logits), dim=1).numpy(),
                                              train_logits,
                                              NUM_CLASSES, prefix="train_")
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics

        else:
            patience_left -= 1
            if patience_left == 0: break  # 早停触发

    if best_state: model.load_state_dict(best_state)

    # 确保返回非空的指标字典
    if best_train_metrics is None:
        val_loader_eval = DataLoader(val_subset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
        y_val_true, _, val_logits = get_preds_and_probs(model, val_loader_eval, device)
        best_val_metrics = calculate_metrics(y_val_true, F.softmax(torch.from_numpy(val_logits), dim=1).numpy(),
                                             val_logits,
                                             NUM_CLASSES, prefix="val_")

        train_loader_eval = DataLoader(train_subset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
        y_train_true, _, train_logits = get_preds_and_probs(model, train_loader_eval, device)
        best_train_metrics = calculate_metrics(y_train_true, F.softmax(torch.from_numpy(train_logits), dim=1).numpy(),
                                               train_logits,
                                               NUM_CLASSES, prefix="train_")

    # 返回当前 Fold 的最佳验证分数、最佳训练集指标和最佳验证集指标
    return best_score, best_train_metrics, best_val_metrics


# --- 核心修改 2: 修改 train_one_config 汇总训练集指标 ---
def train_one_config(config):
    """(主训练函数：执行 K-Fold 循环)"""

    kf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    train_metrics_list = []  # 存储每折的最佳训练指标
    val_metrics_list = []  # 存储每折的最佳验证指标

    # 获取 CV Pool 的时序数据和标签 (用于 K-Fold 划分)
    X_t_cv = cv_dataset.tensors[1].numpy()
    y_cv = cv_dataset.tensors[2].numpy()

    print(f"\n --- 开始 {K_FOLDS}-Fold 交叉验证 ---")

    # 注意：我们使用时序数据 X_t_cv 进行划分
    for fold_id, (train_index, val_index) in enumerate(kf.split(X_t_cv, y_cv)):
        best_score_fold, train_metrics_fold, val_metrics_fold = train_and_eval_fold(config, train_index, val_index,
                                                                                    fold_id + 1)
        train_metrics_list.append(train_metrics_fold)
        val_metrics_list.append(val_metrics_fold)
        print(
            f" Fold {fold_id + 1} 完成 | 最佳 训练-{EARLY_MONITOR.upper()} = {train_metrics_fold[f'train_{EARLY_MONITOR}']:.4f} | 最佳 验证-{EARLY_MONITOR.upper()} = {best_score_fold:.4f}")

    # 计算 K-Fold 平均指标
    avg_train_metrics = {}
    avg_val_metrics = {}

    for metric in ["qwk", "mae", "macro_f1", "bacc", "roc_auc"]:
        avg_train_metrics[f"train_{metric}"] = np.mean([m[f"train_{metric}"] for m in train_metrics_list])
        avg_val_metrics[f"val_{metric}"] = np.mean([m[f"val_{metric}"] for m in val_metrics_list])

    print(
        f"\n K-Fold 平均结果 | 训练集-{EARLY_MONITOR.upper()} = {avg_train_metrics[f'train_{EARLY_MONITOR}']:.4f} | 验证集-{EARLY_MONITOR.upper()} = {avg_val_metrics[f'val_{EARLY_MONITOR}']:.4f}")

    # 返回 K-Fold 平均训练指标和平均验证指标
    return None, avg_train_metrics, avg_val_metrics, None


# ============ 7. 主流程 (执行网格搜索和最终模型训练) ============

# --- 核心修改 4: 新增 append_plotting_data 函数 ---
def append_plotting_data(y_true, probs, num_classes, class_names, config, filename):
    y_binarized = label_binarize(y_true, classes=range(num_classes));
    # 转换为字符串，便于写入文件
    config_str = str(config)

    with open(filename, 'a', encoding='utf-8') as f:  # 使用 'a' for append
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"配置: {config_str}\n")
        f.write("=" * 80 + "\n")

        # 混淆矩阵
        cm = confusion_matrix(y_true, np.argmax(probs, axis=1));
        f.write("混淆矩阵\n");
        np.savetxt(f, cm, fmt='%d', delimiter=',');
        f.write("\n" + "-" * 50 + "\n")

        # 单类别 ROC 曲线
        for i in range(num_classes):
            # 确保至少有两个类别出现，否则 roc_curve 会失败
            if len(np.unique(y_binarized[:, i])) < 2: continue

            fpr, tpr, _ = roc_curve(y_binarized[:, i], probs[:, i]);
            auc_val = roc_auc_score(y_binarized[:, i], probs[:, i])
            f.write(f"类别 '{class_names[i]}' 的 ROC 曲线数据 (AUC = {auc_val:.4f})\n");
            f.write("FPR (假正例率)\n");
            np.savetxt(f, fpr, fmt='%.8f', delimiter=',');
            f.write("TPR (真正例率)\n");
            np.savetxt(f, tpr, fmt='%.8f', delimiter=',');
            f.write("\n" + "-" * 50 + "\n")

        # 微平均 ROC 曲线
        if len(np.unique(y_binarized.ravel())) >= 2:
            fpr_micro, tpr_micro, _ = roc_curve(y_binarized.ravel(), probs.ravel());
            auc_micro = roc_auc_score(y_binarized, probs, average='micro', multi_class='ovr')
            f.write(f"微平均 ROC 曲线数据 (AUC = {auc_micro:.4f})\n");
            f.write("FPR (假正例率)\n");
            np.savetxt(f, fpr_micro, fmt='%.8f', delimiter=',');
            f.write("TPR (真正例率)\n");
            np.savetxt(f, tpr_micro, fmt='%.8f', delimiter=',');
            f.write("\n" + "=" * 50 + "\n\n");

        print(f"用于绘图的数值 (配置: {config_str}) 已追加至: {filename}")


# --- 核心修改 5: 调整 save_plotting_data (仅用于最优模型) ---
def save_plotting_data(y_true, probs, num_classes, class_names, filename):
    # 此函数仅用于保存最佳配置的最终测试报告，并覆盖 PLOT_DATA_FILE
    y_binarized = label_binarize(y_true, classes=range(num_classes));
    with open(filename, 'w', encoding='utf-8') as f:
        cm = confusion_matrix(y_true, np.argmax(probs, axis=1));
        f.write("混淆矩阵\n");
        np.savetxt(f, cm, fmt='%d', delimiter=',');
        f.write("\n" + "=" * 50 + "\n\n")
        for i in range(num_classes):
            if len(np.unique(y_binarized[:, i])) < 2: continue

            fpr, tpr, _ = roc_curve(y_binarized[:, i], probs[:, i]);
            auc_val = roc_auc_score(y_binarized[:, i], probs[:, i])
            f.write(f"类别 '{class_names[i]}' 的 ROC 曲线数据 (AUC = {auc_val:.4f})\n");
            f.write("FPR (假正例率)\n");
            np.savetxt(f, fpr, fmt='%.8f', delimiter=',');
            f.write("TPR (真正例率)\n");
            np.savetxt(f, tpr, fmt='%.8f', delimiter=',');
            f.write("\n" + "=" * 50 + "\n\n")

        if len(np.unique(y_binarized.ravel())) >= 2:
            fpr_micro, tpr_micro, _ = roc_curve(y_binarized.ravel(), probs.ravel());
            auc_micro = roc_auc_score(y_binarized, probs, average='micro', multi_class='ovr')
            f.write(f"微平均 ROC 曲线数据 (AUC = {auc_micro:.4f})\n");
            f.write("FPR (假正例率)\n");
            np.savetxt(f, fpr_micro, fmt='%.8f', delimiter=',');
            np.savetxt(f, tpr_micro, fmt='%.8f', delimiter=',');
            f.write("\n" + "=" * 50 + "\n\n");

        print(f"最优模型的绘图数值已保存至: {filename}")


all_results = []
best_overall_score = -1e9 if EARLY_MONITOR == "qwk" else 1e9
best_config_raw = None
best_avg_val_metrics = None

print("--- 步骤 1: 开始网格搜索 K-Fold 交叉验证 ---")
for i, cfg_raw in enumerate(GRID_CONFIGS):
    # train_one_config 返回 (_, avg_train_metrics, avg_val_metrics, _)
    _, avg_train_metrics, avg_val_metrics, _ = train_one_config(cfg_raw)

    # --- 核心修改 3: 存储结果中包含训练集指标 ---
    combined_metrics = {**avg_train_metrics, **avg_val_metrics}
    all_results.append(
        {"config": cfg_raw, "validation": avg_val_metrics, "combined": combined_metrics, "test_metrics": None})
    current_score = avg_val_metrics[f"val_{EARLY_MONITOR}"]

    is_new_best = False
    if best_config_raw is None:
        is_new_best = True
    else:
        if EARLY_MONITOR == "qwk":
            if current_score > best_overall_score: is_new_best = True
        else:  # 如 mae，越小越好
            if current_score < best_overall_score: is_new_best = True

    if is_new_best:
        best_overall_score, best_config_raw, best_avg_val_metrics = current_score, cfg_raw, avg_val_metrics

# 打印和保存网格搜索总结 (基于 K-Fold 平均结果)
summary_rows = []
for res in all_results:
    row = {"config": str(res["config"])}
    # 记录训练集和验证集的平均指标
    for m in ["qwk", "mae", "macro_f1", "bacc", "roc_auc"]:
        row[f"cv_train_{m}"] = res["combined"][f"train_{m}"]
        row[f"cv_val_{m}"] = res["combined"][f"val_{m}"]
    summary_rows.append(row)

# 按 K-Fold 验证集的 EARLY_MONITOR 指标排序 (初始排序)
res_df = pd.DataFrame(summary_rows).sort_values(by=f"cv_val_{EARLY_MONITOR}",
                                                ascending=(EARLY_MONITOR != "qwk")).reset_index(drop=True)

# --- 核心修改 6: 循环训练所有配置的最终模型并评估 (满足用户需求) ---
print("\n" + "=" * 60 + "\n--- 步骤 2: 循环训练所有配置的最终模型 ($M_{final}$) 并在独立测试集上评估 ---")
print(f"独立测试集详细结果将写入: {ALL_TEST_PLOT_DATA_FILE}")

# 清空旧文件 (如果存在)
if os.path.exists(ALL_TEST_PLOT_DATA_FILE):
    os.remove(ALL_TEST_PLOT_DATA_FILE)

# 循环所有结果
for i, res in enumerate(all_results):
    cfg = res["config"]
    print(f"\n[{i + 1}/{len(all_results)}] 正在评估配置: {cfg}")

    # 1. 训练最终模型 (M_final)
    model_init_config = cfg.copy()
    model_init_config.pop('optimizer', None)
    model_init_config.pop('lr', None)

    # 实例化最终模型 (使用当前配置)
    final_model = CNN_LSTM_Model(
        num_input_channels=NUM_INPUT_CHANNELS, num_classes=NUM_CLASSES,
        cnn_layers=model_init_config["cnn_layers"], kernel_size=model_init_config["kernel_size"],
        act_cnn=model_init_config["act_cnn"],
        lstm_hidden_size=model_init_config["lstm_hidden_size"], lstm_layers=model_init_config["lstm_layers"],
        final_dropout=FINAL_DROPOUT,
        widths=model_init_config["widths"], act_mlp=model_init_config["act_mlp"]
    ).to(device)

    # 训练 M_final 使用整个 80% CV Pool
    final_train_loader = DataLoader(cv_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    criterion = CostSensitiveCE(NUM_CLASSES, LAMBDA_COST, class_weights_t, DIST_POWER, device)
    optimizer = build_optimizer(final_model, cfg)

    # 简化训练，只运行固定 epoch 数 (无需早停)
    for ep in range(1, MAX_EPOCHS + 1):
        final_model.train()
        for x_s_b, x_t_b, y_b in final_train_loader:
            x_s_b, x_t_b, y_b = x_s_b.to(device), x_t_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = final_model(x_s_b, x_t_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

    # 2. 独立测试集评估
    y_test_true, test_probs, test_logits = get_preds_and_probs(final_model, test_loader, device)
    # 计算测试集所有指标
    test_metrics_full = calculate_metrics(y_test_true, test_probs, test_logits, NUM_CLASSES)

    # 3. 记录详细绘图数据 (混淆矩阵和 ROC AUC)
    append_plotting_data(y_test_true, test_probs, NUM_CLASSES, CLASS_NAMES, cfg, ALL_TEST_PLOT_DATA_FILE)

    # 4. 将测试集宏观指标添加到 res_df
    test_metrics_simple = {f"test_{m}": test_metrics_full[m] for m in ["qwk", "mae", "macro_f1", "bacc", "roc_auc"]}

    # 找到 res_df 中对应的行进行更新 (基于配置字符串)
    config_str_current = str(cfg)
    match_index = res_df[res_df['config'] == config_str_current].index
    if not match_index.empty:
        for k, v in test_metrics_simple.items():
            res_df.loc[match_index, k] = v

    # 额外：如果这是最优配置，保存模型权重和 PLOT_DATA_FILE
    if cfg == best_config_raw:
        print(f"\n!!! 检测到最优配置 ({best_config_raw})，保存模型权重和 PLOT_DATA_FILE !!!")
        torch.save(final_model.state_dict(), SAVE_BEST_PATH)
        print(f"最优模型已保存至: {SAVE_BEST_PATH}")
        save_plotting_data(y_test_true, test_probs, NUM_CLASSES, CLASS_NAMES, PLOT_DATA_FILE)

        print("\n--- 最优模型的独立测试集分类报告 (最权威结果) ---")
        print(test_metrics_full['report'])
        print("\n--- 最优模型的独立测试集混淆矩阵 ---")
        print(test_metrics_full['cm'])

# --- 核心修改 7: 最终保存 CSV 文件 (包含所有测试集指标) ---
# 再次按 K-Fold 验证集指标排序 (保持一致性)
res_df = res_df.sort_values(by=f"cv_val_{EARLY_MONITOR}",
                            ascending=(EARLY_MONITOR != "qwk")).reset_index(drop=True)
res_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")

print("\n" + "=" * 80 + f"\n最终结果总结 CSV 已保存至: {RESULTS_CSV}")
print("\n" + "=" * 80 + "\nCSV 文件包含 K-Fold 平均指标和独立测试集指标")
print(res_df.head(10))  # 只显示前 10 行以供预览

print("\n" + "=" * 80 + f"\n所有配置的混淆矩阵和 ROC AUC 详细数据已保存至: {ALL_TEST_PLOT_DATA_FILE}")