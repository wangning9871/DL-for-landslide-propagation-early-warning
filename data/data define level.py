# -*- coding: utf-8 -*-
"""
增强版：在统计与分级聚合中加入 Q90（第 90 百分位）
说明：
- 与之前脚本功能相同（支持自定义阈值或使用分位阈值）
- 新增 Q90 输出（metrics summary 与 grouped stats 都包含 Q90）
- 顶部修改：USE_PERCENTILES, CUSTOM_THRESHOLDS, SAVE_AUX
"""

import os, sys
import pandas as pd, numpy as np
from datetime import datetime

# ---------------- 配置 ----------------
USE_PERCENTILES = False

CUSTOM_THRESHOLDS = {
    'interval':       {'low': 20.0, 'high': 15.0},
    'w2_max_h':       {'low': 0.4,  'high': 0.6},
    'w2_max_v':       {'low': 4,  'high': 6},
    'w2_max_q':       {'low': 15.0,  'high': 35.0},
    'w2_total_flow':  {'low': 150.0, 'high': 300.0}
}

SAVE_AUX = False   # 是否保存所有辅助文件（False 时只保存最终表 + 报告）
DEFAULT_EXTRACTED = 'HK_extracted_table_custom_order.csv'
WORKDIR = os.getcwd()

OUT_FINAL = 'HK_final_inputs_with_class.csv'
OUT_METRICS = 'HK_metrics_percentiles.csv'           # 包含 Q25,Q50,Q75,Q90
OUT_THRESH  = 'HK_simple_thresholds_used.csv'
OUT_FULL_LABELED = 'HK_simple_rule_full_with_class.csv'
OUT_METRICS_WITH_CLASS = 'HK_metrics_with_class.csv'
OUT_GROUPED = 'HK_grouped_stats.csv'                 # 每等级的聚合统计，包含 Q90
OUT_PREVIEW = 'HK_examples_per_class_preview.csv'
OUT_REPORT = 'HK_analysis_report.txt'

REQUIRED_METRICS = ['interval_w1_w2','w2_max_h','w2_max_v','w2_max_q','w2_total_flow']
# --------------------------------------

def find_extracted_file():
    default_path = os.path.join(WORKDIR, DEFAULT_EXTRACTED)
    if os.path.exists(default_path):
        return default_path
    csvs = [f for f in os.listdir(WORKDIR) if f.lower().endswith('.csv')]
    for f in csvs:
        p = os.path.join(WORKDIR, f)
        try:
            tmp = pd.read_csv(p, nrows=5)
            if all(col in tmp.columns for col in REQUIRED_METRICS):
                return p
        except Exception:
            continue
    return None

def compute_stats(df, metrics):
    """
    计算统计量：min, Q25, median(Q50), Q75, Q90, max, mean
    返回 DataFrame（每行对应一个 metric）
    """
    rows = []
    for m in metrics:
        s = df[m].dropna()
        rows.append({
            'metric': m,
            'min': float(s.min()) if not s.empty else np.nan,
            'Q25': float(s.quantile(0.25)) if not s.empty else np.nan,
            'median': float(s.quantile(0.5)) if not s.empty else np.nan,
            'Q75': float(s.quantile(0.75)) if not s.empty else np.nan,
            'Q90': float(s.quantile(0.90)) if not s.empty else np.nan,   # 新增 Q90
            'max': float(s.max()) if not s.empty else np.nan,
            'mean': float(s.mean()) if not s.empty else np.nan,
            'count': int(s.count())
        })
    return pd.DataFrame(rows)

def build_thresholds_from_stats(stats_df):
    """
    仍然保留原有的阈值构造逻辑（low 用 median，high 用 Q25 或 Q75）
    如果你想使用 Q90 作为某个指标的 high/low，可以在这里替换为 stats_df['Q90']。
    """
    thr = {
        'interval': {'low': float(stats_df.loc[stats_df['metric']=='interval_w1_w2','median'].values[0]),
                     'high': float(stats_df.loc[stats_df['metric']=='interval_w1_w2','Q25'].values[0])},
        'w2_max_h': {'low': float(stats_df.loc[stats_df['metric']=='w2_max_h','median'].values[0]),
                     'high': float(stats_df.loc[stats_df['metric']=='w2_max_h','Q75'].values[0])},
        'w2_max_v': {'low': float(stats_df.loc[stats_df['metric']=='w2_max_v','median'].values[0]),
                     'high': float(stats_df.loc[stats_df['metric']=='w2_max_v','Q75'].values[0])},
        'w2_max_q': {'low': float(stats_df.loc[stats_df['metric']=='w2_max_q','median'].values[0]),
                     'high': float(stats_df.loc[stats_df['metric']=='w2_max_q','Q75'].values[0])},
        'w2_total_flow': {'low': float(stats_df.loc[stats_df['metric']=='w2_total_flow','median'].values[0]),
                     'high': float(stats_df.loc[stats_df['metric']=='w2_total_flow','Q75'].values[0])}
    }
    return thr

def classify_row_simple(row, thr):
    # class 0: not reached
    if ('arrival_time_w2' in row.index and pd.isna(row['arrival_time_w2'])) or (pd.notna(row.get('w2_total_flow')) and row.get('w2_total_flow')==0):
        return 0
    iv = row.get('interval_w1_w2', np.nan)
    # class 3: any high
    if pd.notna(iv) and iv <= thr['interval']['high']:
        return 3
    if pd.notna(row.get('w2_max_h')) and row.get('w2_max_h') >= thr['w2_max_h']['high']:
        return 3
    if pd.notna(row.get('w2_max_v')) and row.get('w2_max_v') >= thr['w2_max_v']['high']:
        return 3
    if pd.notna(row.get('w2_max_q')) and row.get('w2_max_q') >= thr['w2_max_q']['high']:
        return 3
    if pd.notna(row.get('w2_total_flow')) and row.get('w2_total_flow') >= thr['w2_total_flow']['high']:
        return 3
    # class 1: all low
    cond_low = True
    if not (pd.notna(iv) and iv >= thr['interval']['low']):
        cond_low = False
    if not (pd.notna(row.get('w2_max_h')) and row.get('w2_max_h') <= thr['w2_max_h']['low']):
        cond_low = False
    if not (pd.notna(row.get('w2_max_v')) and row.get('w2_max_v') <= thr['w2_max_v']['low']):
        cond_low = False
    if not (pd.notna(row.get('w2_max_q')) and row.get('w2_max_q') <= thr['w2_max_q']['low']):
        cond_low = False
    if not (pd.notna(row.get('w2_total_flow')) and row.get('w2_total_flow') <= thr['w2_total_flow']['low']):
        cond_low = False
    if cond_low:
        return 1
    return 2

def save_report_text(report_lines, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("HK 分级分析报告\n")
        f.write("生成时间: " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC") + "\n\n")
        for line in report_lines:
            f.write(line + "\n")
    return

def main():
    report = []
    print("工作目录：", WORKDIR); report.append("工作目录： " + WORKDIR)

    path = find_extracted_file()
    if path is None:
        msg = "错误：未找到包含所需五个指标的特征表，请把 'HK_extracted_table_custom_order.csv' 放入此目录。"
        print(msg); report.append(msg)
        save_report_text(report, os.path.join(WORKDIR, OUT_REPORT))
        sys.exit(1)

    msg = f"读取特征表： {path}"
    print(msg); report.append(msg)
    df = pd.read_csv(path)

    # 强制把判级相关列转换成数值，避免字符串导致比较失效
    for c in REQUIRED_METRICS + (['arrival_time_w2'] if 'arrival_time_w2' in df.columns else []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    missing = [c for c in REQUIRED_METRICS if c not in df.columns]
    if missing:
        msg = f"错误：输入表缺少必要列： {missing}"
        print(msg); report.append(msg)
        save_report_text(report, os.path.join(WORKDIR, OUT_REPORT))
        sys.exit(1)

    # 计算统计（含 Q90）
    stats_df = compute_stats(df, REQUIRED_METRICS)
    stats_text = ["五个指标统计（min,Q25,median,Q75,Q90,max,mean,count）：", stats_df.to_string(index=False)]
    for t in stats_text:
        print(t); report.append(t)
    if SAVE_AUX:
        stats_df.to_csv(os.path.join(WORKDIR, OUT_METRICS), index=False, float_format='%.6f', encoding='utf-8')

    # 阈值选择
    if USE_PERCENTILES:
        thresholds = build_thresholds_from_stats(stats_df)
        note = "使用分位生成阈值（interval high 用 Q25，low 用 median）"
    else:
        if not CUSTOM_THRESHOLDS:
            msg = "错误：USE_PERCENTILES=False 但 CUSTOM_THRESHOLDS 为空，请在脚本中填入自定义阈值后重试。"
            print(msg); report.append(msg)
            save_report_text(report, os.path.join(WORKDIR, OUT_REPORT))
            sys.exit(1)
        thresholds = CUSTOM_THRESHOLDS
        note = "使用自定义阈值（来自脚本顶部）"
    print(note); report.append(note)

    thr_lines = ["参考阈值 (metric: low / high):"]
    for k,v in thresholds.items():
        line = f" - {k:15s}: low = {float(v['low']):.6f} , high = {float(v['high']):.6f}"
        print(line); thr_lines.append(line)
    report.extend(thr_lines)
    if SAVE_AUX:
        pd.DataFrame([{'metric':k,'low':v['low'],'high':v['high']} for k,v in thresholds.items()]).to_csv(os.path.join(WORKDIR, OUT_THRESH), index=False, float_format='%.6f', encoding='utf-8')

    # 打等级
    df['class_simple'] = df.apply(lambda r: classify_row_simple(r, thresholds), axis=1)

    # 计算 metrics_with_class 与 grouped_df（包括 Q90）
    metrics_with_class = df[REQUIRED_METRICS + ['class_simple']].copy()
    grouped = []
    preview_rows = []
    for cls, group in metrics_with_class.groupby('class_simple'):
        preview_rows.append(group.head(5))
        for m in REQUIRED_METRICS:
            s = group[m].dropna()
            grouped.append({
                'class': int(cls),
                'metric': m,
                'count': int(s.count()),
                'min': float(s.min()) if not s.empty else np.nan,
                'Q25': float(s.quantile(0.25)) if not s.empty else np.nan,
                'median': float(s.quantile(0.5)) if not s.empty else np.nan,
                'Q75': float(s.quantile(0.75)) if not s.empty else np.nan,
                'Q90': float(s.quantile(0.90)) if not s.empty else np.nan,   # 新增 Q90
                'max': float(s.max()) if not s.empty else np.nan,
                'mean': float(s.mean()) if not s.empty else np.nan
            })
    grouped_df = pd.DataFrame(grouped).sort_values(['class','metric']).reset_index(drop=True)

    # counts & report
    counts = df['class_simple'].value_counts().sort_index()
    counts_text = ["等级计数 (class_simple):", counts.to_string()]
    for t in counts_text:
        print(t); report.append(t)

    # Always save final inputs + class (new_df)
    drop_cols = [c for c in REQUIRED_METRICS if c in df.columns]
    if 'arrival_time_w2' in df.columns:
        drop_cols.append('arrival_time_w2')
    new_df = df.drop(columns=drop_cols)
    final_path = os.path.join(WORKDIR, OUT_FINAL)
    new_df.to_csv(final_path, index=False, float_format='%.6f', encoding='utf-8')
    line = f"已保存最终表（仅输入 + class）到: {OUT_FINAL}"
    print(line); report.append(line)

    # If SAVE_AUX, write auxiliary CSVs
    if SAVE_AUX:
        stats_df.to_csv(os.path.join(WORKDIR, OUT_METRICS), index=False, float_format='%.6f', encoding='utf-8')
        thr_df = pd.DataFrame([{'metric':k,'low':v['low'],'high':v['high']} for k,v in thresholds.items()])
        thr_df.to_csv(os.path.join(WORKDIR, OUT_THRESH), index=False, float_format='%.6f', encoding='utf-8')
        df.to_csv(os.path.join(WORKDIR, OUT_FULL_LABELED), index=False, float_format='%.6f', encoding='utf-8')
        metrics_with_class.to_csv(os.path.join(WORKDIR, OUT_METRICS_WITH_CLASS), index=False, float_format='%.6f', encoding='utf-8')
        grouped_df.to_csv(os.path.join(WORKDIR, OUT_GROUPED), index=False, float_format='%.6f', encoding='utf-8')
        preview_df = pd.concat(preview_rows) if preview_rows else pd.DataFrame()
        if not preview_df.empty:
            preview_df.to_csv(os.path.join(WORKDIR, OUT_PREVIEW), index=False, float_format='%.6f', encoding='utf-8')
        report.append("已保存辅助文件： " + ", ".join([OUT_METRICS, OUT_THRESH, OUT_FULL_LABELED, OUT_METRICS_WITH_CLASS, OUT_GROUPED, OUT_PREVIEW]))
    else:
        report.append("未保存辅助文件（SAVE_AUX=False），仅保存最终表。")

    # append grouped stats text to report (includes Q90)
    report.append("\n每等级对五指标的聚合统计（count,min,Q25,median,Q75,Q90,max,mean）：")
    report.append(grouped_df.to_string(index=False))

    # save human-readable report txt
    save_report_text(report, os.path.join(WORKDIR, OUT_REPORT))
    print("已保存完整分析报告到:", OUT_REPORT)

    # final summary
    print("\n主要输出：")
    print(" - 最终表（输入 + class）:", OUT_FINAL)
    if SAVE_AUX:
        print(" - 辅助文件已保存（见工作目录）")
    print(" - 分析报告：", OUT_REPORT)

if __name__ == '__main__':
    main()
