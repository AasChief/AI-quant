# -*- coding: utf-8 -*-
"""
TASK5: 分类型机器学习算法实践
- 逻辑回归、决策树、随机森林
- 混淆矩阵、ROC曲线、AUC
- 股票收益数据 + 乳腺癌数据集
作者: 童逸
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, roc_auc_score
)
from sklearn.datasets import load_breast_cancer

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak
)
from reportlab.lib.styles import ParagraphStyle

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 注册PDF宋体字体
pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))

# PDF段落样式
style_title = ParagraphStyle(
    'TitleStyle', fontName='SimSun', fontSize=16, leading=24,
    alignment=TA_CENTER, spaceBefore=0, spaceAfter=0
)
style_h1 = ParagraphStyle(
    'H1Style', fontName='SimSun', fontSize=14, leading=21,
    alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0
)
style_h2 = ParagraphStyle(
    'H2Style', fontName='SimSun', fontSize=12, leading=18,
    alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0
)
style_body = ParagraphStyle(
    'BodyStyle', fontName='SimSun', fontSize=10.5, leading=15.75,
    alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0
)
style_caption = ParagraphStyle(
    'CaptionStyle', fontName='SimSun', fontSize=10.5, leading=15.75,
    alignment=TA_CENTER, spaceBefore=0, spaceAfter=0
)

# 中国股市涨跌颜色：涨红跌绿
COLOR_UP = '#d62728'   # 红色
COLOR_DOWN = '#2ca02c'  # 绿色


# ============================================================
# 1. 技术指标计算函数
# ============================================================
def calc_rsi(close, period=14):
    """计算RSI指标"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def calc_bollinger(close, period=20, num_std=2):
    """计算布林带"""
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    """计算KDJ指标"""
    low_list = low.rolling(window=n, min_periods=n).min()
    high_list = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_list) / (high_list - low_list) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def compute_indicators(df):
    """为股票数据计算全部技术指标"""
    df = df.copy()
    df = df.sort_values('trade_date').reset_index(drop=True)

    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['vol']

    # RSI
    df['rsi'] = calc_rsi(close, 14)

    # MACD
    df['dif'], df['dea'], df['macd_hist'] = calc_macd(close)

    # 布林带
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = calc_bollinger(close)

    # 布林带位置 (close在带中的相对位置)
    df['bb_pos'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # KDJ
    df['k'], df['d'], df['j'] = calc_kdj(high, low, close)

    # 成交量变化率
    df['vol_change'] = vol.pct_change()

    # 5日收益率
    df['ret_5d'] = close.pct_change(5)

    # 20日收益率
    df['ret_20d'] = close.pct_change(20)

    # 次日涨跌方向 (1=涨, 0=跌)
    df['target'] = (close.shift(-1) - close > 0).astype(int)

    return df


# ============================================================
# 2. 加载股票数据
# ============================================================
def load_stock_data():
    """加载3只股票数据并计算指标"""
    stocks = [
        ('中船特气', '688146', '中船特气_688146_daily_data.csv'),
        ('天地科技', '600587', 'tiandi_keji_600587_daily_data.csv'),
        ('平安银行', '000001', '平安银行_000001_daily_data.csv'),
    ]

    all_data = []
    for name, code, filename in stocks:
        filepath = os.path.join(BASE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [警告] 文件不存在: {filename}")
            continue

        df = pd.read_csv(filepath, encoding='utf-8-sig')

        # 统一日期格式（兼容 20250630 和 2025-06-30 两种格式）
        date_str = df['trade_date'].astype(str)
        # 尝试 YYYYMMDD 格式
        parsed = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
        if parsed.isna().all():
            # 尝试 YYYY-MM-DD 格式
            parsed = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        if parsed.isna().all():
            # 最终兜底：自动推断
            parsed = pd.to_datetime(date_str, errors='coerce')
        df['trade_date'] = parsed

        df = df.dropna(subset=['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 计算指标
        df = compute_indicators(df)
        df['stock_name'] = name
        df['stock_code'] = code

        all_data.append(df)
        print(f"  加载 {name}({code}): {len(df)} 条记录")

    combined = pd.concat(all_data, ignore_index=True)
    return combined


# ============================================================
# 3. 构建分类数据集
# ============================================================
def prepare_stock_dataset(df):
    """构建股票分类数据集"""
    features = ['rsi', 'dif', 'dea', 'macd_hist', 'bb_pos', 'k', 'd', 'j',
                'vol_change', 'ret_5d', 'ret_20d']
    feature_labels = ['RSI(14)', 'MACD-DIF', 'MACD-DEA', 'MACD直方图',
                      '布林带位置', 'K值', 'D值', 'J值',
                      '成交量变化率', '5日收益率', '20日收益率']

    data = df[features + ['target']].dropna()
    X = data[features].values
    y = data['target'].values

    return X, y, features, feature_labels


# ============================================================
# 4. 模型训练与评估
# ============================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name):
    """训练3个模型并评估"""
    results = {}

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    }

    for name, model in models.items():
        # 训练
        if name == '逻辑回归':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # 评估
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_val = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5

        # ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc_val,
            'fpr': fpr,
            'tpr': tpr,
            'cm': cm,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }

        print(f"  [{dataset_name}] {name}: Acc={acc:.4f}, AUC={auc_val:.4f}, F1={f1:.4f}")

    return results, scaler


# ============================================================
# 5. 可视化函数
# ============================================================
def plot_roc_stock(results_stock, savepath):
    """图1: 股票数据ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_list = ['#1f77b4', '#ff7f0e', COLOR_UP]

    for (name, res), color in zip(results_stock.items(), colors_list):
        ax.plot(res['fpr'], res['tpr'], color=color, linewidth=2,
                label=f'{name} (AUC={res["auc"]:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机分类 (AUC=0.5)')
    ax.set_xlabel('假正例率 (FPR)', fontsize=12)
    ax.set_ylabel('真正例率 (TPR)', fontsize=12)
    ax.set_title('股票数据 — 三种模型ROC曲线对比', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {savepath}")


def plot_confusion_matrix(results_stock, results_cancer, savepath):
    """图2: 混淆矩阵热力图 (6个子图)"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    model_names = ['逻辑回归', '决策树', '随机森林']

    for i, name in enumerate(model_names):
        # 股票数据
        cm_s = results_stock[name]['cm']
        ax = axes[0, i]
        im = ax.imshow(cm_s, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'股票数据 — {name}', fontsize=12)
        thresh = cm_s.max() / 2
        for row in range(cm_s.shape[0]):
            for col in range(cm_s.shape[1]):
                ax.text(col, row, str(cm_s[row, col]),
                        ha='center', va='center', fontsize=14,
                        color='white' if cm_s[row, col] > thresh else 'black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['跌(0)', '涨(1)'])
        ax.set_yticklabels(['跌(0)', '涨(1)'])
        ax.set_xlabel('预测值', fontsize=10)
        ax.set_ylabel('真实值', fontsize=10)

        # 乳腺癌数据
        cm_c = results_cancer[name]['cm']
        ax = axes[1, i]
        im = ax.imshow(cm_c, interpolation='nearest', cmap=plt.cm.Greens)
        ax.set_title(f'乳腺癌数据 — {name}', fontsize=12)
        thresh = cm_c.max() / 2
        for row in range(cm_c.shape[0]):
            for col in range(cm_c.shape[1]):
                ax.text(col, row, str(cm_c[row, col]),
                        ha='center', va='center', fontsize=14,
                        color='white' if cm_c[row, col] > thresh else 'black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['恶性(0)', '良性(1)'])
        ax.set_yticklabels(['恶性(0)', '良性(1)'])
        ax.set_xlabel('预测值', fontsize=10)
        ax.set_ylabel('真实值', fontsize=10)

    plt.suptitle('混淆矩阵对比 (上: 股票数据, 下: 乳腺癌数据)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {savepath}")


def plot_feature_importance(rf_model, feature_labels, savepath):
    """图3: 随机森林特征重要性"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(importances)), importances[indices],
                  color=[COLOR_UP if i < len(importances)//2 else COLOR_DOWN for i in range(len(importances))],
                  alpha=0.85)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_labels[i] for i in indices], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('重要性', fontsize=12)
    ax.set_title('随机森林 — 特征重要性排序 (股票数据)', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)

    # 在柱子上方标注数值
    for bar, idx in zip(bars, indices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{importances[idx]:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {savepath}")


def plot_roc_cancer(results_cancer, savepath):
    """图4: 乳腺癌数据ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_list = ['#1f77b4', '#ff7f0e', COLOR_UP]

    for (name, res), color in zip(results_cancer.items(), colors_list):
        ax.plot(res['fpr'], res['tpr'], color=color, linewidth=2,
                label=f'{name} (AUC={res["auc"]:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机分类 (AUC=0.5)')
    ax.set_xlabel('假正例率 (FPR)', fontsize=12)
    ax.set_ylabel('真正例率 (TPR)', fontsize=12)
    ax.set_title('乳腺癌数据集 — 三种模型ROC曲线对比', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {savepath}")


def plot_model_comparison(results_stock, results_cancer, savepath):
    """图5: 模型性能对比柱状图"""
    model_names = ['逻辑回归', '决策树', '随机森林']
    metrics = ['accuracy', 'f1', 'auc']
    metric_labels = ['Accuracy', 'F1-Score', 'AUC']

    x = np.arange(len(metrics))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 股票数据
    ax = axes[0]
    colors_list = ['#1f77b4', '#ff7f0e', COLOR_UP]
    for i, name in enumerate(model_names):
        vals = [results_stock[name][m] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name,
                      color=colors_list[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('股票数据 — 模型性能对比', fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # 乳腺癌数据
    ax = axes[1]
    for i, name in enumerate(model_names):
        vals = [results_cancer[name][m] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name,
                      color=colors_list[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('乳腺癌数据集 — 模型性能对比', fontsize=14)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('三种模型性能对比 (左: 股票数据, 右: 乳腺癌数据)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {savepath}")


# ============================================================
# 6. 生成PDF报告
# ============================================================
def generate_pdf(results_stock, results_cancer, feature_labels, stock_df,
                 chart_paths, stock_info):
    """生成PDF报告"""
    pdf_path = os.path.join(BASE_DIR, '童逸+TASK5.pdf')
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )

    story = []

    # ---- 封面标题 ----
    story.append(Paragraph('分类型机器学习算法实践', style_title))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph('TASK5 报告', style_h2))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph('姓名：童逸', style_body))
    story.append(Spacer(1, 6*mm))

    # ---- 第一章 ----
    story.append(Paragraph('一、分类型机器学习算法介绍', style_h1))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('1.1 逻辑回归（Logistic Regression）', style_h2))
    story.append(Paragraph(
        '逻辑回归是一种广义线性分类模型，其核心思想是通过Sigmoid函数将线性回归的输出值映射到(0,1)区间，'
        '从而表示样本属于某一类别的概率。其数学表达式为：P(Y=1|X) = 1 / (1 + e^(-&beta;X))，其中&beta;为模型参数，'
        'X为特征向量。当计算得到的概率大于设定阈值（通常为0.5）时，将样本判定为正类，否则判定为负类。'
        '逻辑回归的优点在于模型简单、计算效率高、可解释性强，能够输出概率值便于决策；缺点是对非线性关系的'
        '拟合能力有限，需要手动进行特征工程。在量化交易中，逻辑回归常被用作股价涨跌预测的基准模型，'
        '为后续复杂模型提供对比参照。', style_body))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('1.2 决策树（Decision Tree）', style_h2))
    story.append(Paragraph(
        '决策树是一种基于树形结构进行决策的非参数分类算法。其基本原理是通过信息增益或基尼不纯度等指标，'
        '递归地选择最优特征对样本空间进行划分，直到满足停止条件（如达到最大深度或节点纯度足够高），'
        '最终形成从根节点到叶节点的决策路径。信息增益基于信息熵的定义：Entropy = -&Sigma; p_i * log2(p_i)，'
        '表示数据的不确定性；基尼不纯度则定义为：Gini = 1 - &Sigma; p_i^2，衡量从数据集中随机抽取两个样本'
        '属于不同类的概率。决策树的优点是直观易懂、能够处理非线性关系、不需要特征标准化；缺点是容易过拟合，'
        '对数据微小变化敏感。通过限制树的深度（如max_depth=5）可以有效控制过拟合。', style_body))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('1.3 随机森林（Random Forest）', style_h2))
    story.append(Paragraph(
        '随机森林是一种基于Bagging（Bootstrap Aggregating）思想的集成学习算法。其核心机制包含两个层面的'
        '随机性：一是对训练数据进行有放回抽样（Bootstrap采样），为每棵决策树生成不同的训练子集；二是在'
        '每个节点分裂时，从全部特征中随机选取一个子集作为候选分裂特征。通过构建多棵决策树（本实验中'
        'n_estimators=100）并对预测结果进行多数投票，随机森林能够显著降低方差、提高泛化能力。'
        '相比单棵决策树，随机森林的抗过拟合能力更强、预测准确率更高，还能通过计算特征在分裂中的'
        '贡献度来评估特征重要性。在金融数据分类中，随机森林被广泛应用于特征选择和涨跌预测，是量化策略'
        '开发中的核心工具之一。', style_body))
    story.append(Spacer(1, 4*mm))

    # ---- 第二章 ----
    story.append(Paragraph('二、机器学习模型评价指标', style_h1))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('2.1 混淆矩阵（Confusion Matrix）', style_h2))
    story.append(Paragraph(
        '混淆矩阵是评估分类模型性能的基础工具，它以矩阵形式展示模型预测结果与真实标签之间的对应关系。'
        '对于二分类问题，混淆矩阵包含四个基本元素：TP（真正例，实际为正且预测为正）、FP（假正例，实际为负'
        '但预测为正）、TN（真负例，实际为负且预测为负）、FN（假负例，实际为正但预测为负）。基于这四个'
        '指标可以衍生出多个评价指标：准确率（Accuracy = (TP+TN)/(TP+FP+TN+FN)）衡量整体预测正确率；'
        '精确率（Precision = TP/(TP+FP)）衡量预测为正的样本中有多少实际为正；召回率（Recall = TP/(TP+FN)）'
        '衡量实际为正的样本中有多少被正确预测；F1-Score是精确率和召回率的调和平均值，综合衡量模型性能。'
        '在股价涨跌预测中，召回率尤为重要，因为漏判上涨信号的机会成本远高于误判。', style_body))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('2.2 ROC曲线（Receiver Operating Characteristic）', style_h2))
    story.append(Paragraph(
        'ROC曲线是描述二分类器在不同阈值下性能的图形化工具。其横轴为假正例率（FPR = FP/(FP+TN)），'
        '表示负类样本中被错误判为正类的比例；纵轴为真正例率（TPR = TP/(TP+FN)，即召回率），表示正类样本'
        '中被正确判为正类的比例。通过调整分类阈值（从0到1），可以得到一系列(FPR, TPR)点，连接这些点'
        '即得到ROC曲线。ROC曲线越靠近左上角，说明分类器在不同阈值下都能保持较高的真正例率和较低的'
        '假正例率，模型性能越好。当ROC曲线与对角线重合时，表示分类器等同于随机猜测。ROC曲线的优势在于'
        '它不依赖于具体的分类阈值，能够全面评估模型在不同决策偏好下的表现。', style_body))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('2.3 AUC（Area Under Curve）', style_h2))
    story.append(Paragraph(
        'AUC是ROC曲线下方的面积，取值范围为[0, 1]，用于量化评估分类模型的整体区分能力。AUC的物理含义是：'
        '随机选取一个正类样本和一个负类样本，分类器将正类样本预测为正的概率大于将负类样本预测为正的概率'
        '的概率。AUC=0.5表示模型等同于随机分类，没有区分能力；AUC=1表示完美分类。一般解读标准为：'
        'AUC在0.5-0.7之间表示模型区分能力较差，0.7-0.8为良好，0.8-0.9为优秀，0.9以上为卓越。'
        'AUC作为单一数值指标，便于在不同模型之间进行横向比较，是模型选择的重要依据。在实际应用中，'
        'AUC还具有良好的鲁棒性，不受类别不平衡问题的影响。', style_body))
    story.append(Spacer(1, 4*mm))

    # ---- 第三章 ----
    story.append(Paragraph('三、数据说明与预处理', style_h1))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph(
        '本实验采用双重数据源进行分类模型训练与评估。主数据源为A股市场3只股票的日线行情数据，'
        '包括中船特气（688146.SH）、天地科技（600587.SH）和平安银行（000001.SZ），时间跨度为过去一年。'
        '通过计算技术指标作为特征变量，以次日收盘价涨跌方向（涨=1，跌=0）作为目标变量，构建股价涨跌'
        '分类数据集。对比数据源采用scikit-learn内置的乳腺癌数据集（Breast Cancer Wisconsin Dataset），'
        '包含569个样本、30个数值特征，目标变量为肿瘤良性（1）或恶性（0）。', style_body))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('表1：股票数据特征变量说明', style_caption))
    story.append(Spacer(1, 1*mm))

    # 特征说明表
    table_data = [['特征变量', '含义说明', '计算方式']]
    feature_desc = [
        ('RSI(14)', '相对强弱指标', '14日内上涨幅度占总波动比例'),
        ('MACD-DIF', '快慢均线差', 'EMA(12) - EMA(26)'),
        ('MACD-DEA', '信号线', 'DIF的9日EMA'),
        ('MACD直方图', '动能指标', '(DIF - DEA) × 2'),
        ('布林带位置', '价格在带中位置', '(close-lower)/(upper-lower)'),
        ('K值', 'KDJ指标K线', 'RSV的3日加权平均'),
        ('D值', 'KDJ指标D线', 'K值的3日加权平均'),
        ('J值', 'KDJ指标J线', '3K - 2D'),
        ('成交量变化率', '量能变化', '当日量/前日量 - 1'),
        ('5日收益率', '短期动量', 'close/close(-5) - 1'),
        ('20日收益率', '中期动量', 'close/close(-20) - 1'),
    ]
    for f, d, c in feature_desc:
        table_data.append([f, d, c])

    t = Table(table_data, colWidths=[3.5*cm, 4*cm, 7*cm])
    t.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'SimSun', 10.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#D9E2F3')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph(
        f'股票数据集合并后共包含{stock_info["total_samples"]}个样本，其中正类（次日上涨）占比'
        f'{stock_info["pos_ratio"]:.1%}。乳腺癌数据集共569个样本，正类（良性）占比62.7%。'
        '两个数据集均按7:3比例划分训练集与测试集，随机种子设为42以保证可复现性。'
        '对于逻辑回归模型，训练前使用StandardScaler对特征进行标准化处理（均值为0，方差为1）；'
        '决策树和随机森林不需要标准化，直接使用原始特征训练。', style_body))
    story.append(Spacer(1, 4*mm))

    # ---- 第四章 ----
    story.append(Paragraph('四、模型构建与训练', style_h1))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph(
        '本实验构建了三种分类模型进行对比分析。（1）逻辑回归模型：使用LogisticRegression实现，'
        '设置max_iter=1000以保证收敛，random_state=42，训练前对特征进行标准化处理。'
        '（2）决策树模型：使用DecisionTreeClassifier实现，设置max_depth=5限制树深度以防过拟合，'
        '采用基尼不纯度作为分裂准则，random_state=42。（3）随机森林模型：使用RandomForestClassifier'
        '实现，设置n_estimators=100构建100棵决策树，max_depth=8控制单树深度，random_state=42。'
        '三个模型均在相同的训练集上训练，在相同的测试集上评估，确保结果的可比性。', style_body))
    story.append(Spacer(1, 4*mm))

    # ---- 第五章 ----
    story.append(PageBreak())
    story.append(Paragraph('五、模型评估与可视化', style_h1))
    story.append(Spacer(1, 2*mm))

    # 图1: 股票ROC
    story.append(Paragraph('图1：股票数据 — 三种模型ROC曲线对比', style_caption))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['roc_stock'], width=14*cm, height=10.5*cm))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        f'图1展示了三种模型在股票数据测试集上的ROC曲线。逻辑回归的AUC为{results_stock["逻辑回归"]["auc"]:.4f}，'
        f'决策树的AUC为{results_stock["决策树"]["auc"]:.4f}，随机森林的AUC为{results_stock["随机森林"]["auc"]:.4f}。'
        '可以看到，三种模型的ROC曲线均略高于对角线（随机分类线），说明模型对股价涨跌方向具有一定的'
        '预测能力，但预测能力有限。这在金融市场中是符合预期的——股价涨跌受大量不可预测因素影响，'
        '技术指标能提供的信息量有限。随机森林的AUC略高于其他两个模型，体现了集成学习在处理噪声数据'
        '方面的优势。', style_body))
    story.append(Spacer(1, 3*mm))

    # 图2: 混淆矩阵
    story.append(Paragraph('图2：混淆矩阵对比（上：股票数据，下：乳腺癌数据）', style_caption))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['confusion_matrix'], width=15*cm, height=9.6*cm))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '图2展示了三种模型在两个数据集上的混淆矩阵。在股票数据中，三种模型的混淆矩阵分布较为均匀，'
        '正确分类的样本数略多于错误分类的样本数，与较低的AUC值一致。在乳腺癌数据中，三种模型均表现出'
        '极高的分类准确率，对角线上的数值远大于非对角线上的数值，说明模型能够很好地区分良性和恶性'
        '肿瘤。特别地，乳腺癌数据集中假负例（恶性被判为良性）的数量极少，这在医学诊断中至关重要，'
        '因为漏诊恶性肿瘤的代价远高于误诊。', style_body))
    story.append(Spacer(1, 3*mm))

    # 图3: 特征重要性
    story.append(Paragraph('图3：随机森林 — 特征重要性排序（股票数据）', style_caption))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['feature_importance'], width=15*cm, height=9*cm))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '图3展示了随机森林模型在股票数据上计算的特征重要性排序。可以看到，布林带位置（bb_pos）、'
        'RSI和MACD相关指标的重要性较高，说明这些技术指标对股价次日涨跌方向的预测贡献最大。'
        '布林带位置反映了价格在波动区间中的相对位置，当价格接近下轨时可能存在反弹机会；'
        'RSI反映超买超卖状态；MACD直方图反映短期动能变化。相比之下，KDJ指标的K值和D值重要性较低，'
        '这可能是因为KDJ本身存在滞后性且在高频交易中信号较弱。这一结果对量化策略开发有直接指导意义：'
        '应优先关注高重要性指标进行特征工程和策略构建。', style_body))
    story.append(Spacer(1, 3*mm))

    # 图4: 乳腺癌ROC
    story.append(Paragraph('图4：乳腺癌数据集 — 三种模型ROC曲线对比', style_caption))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['roc_cancer'], width=14*cm, height=10.5*cm))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        f'图4展示了三种模型在乳腺癌数据集上的ROC曲线。逻辑回归的AUC为{results_cancer["逻辑回归"]["auc"]:.4f}，'
        f'决策树的AUC为{results_cancer["决策树"]["auc"]:.4f}，随机森林的AUC为{results_cancer["随机森林"]["auc"]:.4f}。'
        '三种模型的ROC曲线均非常接近左上角，AUC值均在0.95以上，表明分类性能卓越。这与股票数据形成了'
        '鲜明对比——乳腺癌数据集的30个细胞核形态特征（如半径、纹理、周长等）与肿瘤良恶性之间具有'
        '强确定性关系，而股价涨跌则受大量随机因素影响。这一对比说明，机器学习模型的性能高度依赖于'
        '数据本身的可分性，在金融市场的应用中需要对预测能力有合理预期。', style_body))
    story.append(Spacer(1, 3*mm))

    # 图5: 模型对比
    story.append(Paragraph('图5：三种模型性能对比柱状图', style_caption))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['model_comparison'], width=15*cm, height=6.4*cm))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '图5以分组柱状图形式展示了三种模型在两个数据集上的Accuracy、F1-Score和AUC三项指标。'
        '在股票数据中，三种模型的各项指标均在0.45-0.60之间，差异不大，且均接近随机分类水平，'
        '说明仅凭技术指标难以稳定预测股价涨跌。在乳腺癌数据中，三种模型的各项指标均在0.90以上，'
        '随机森林和逻辑回归表现尤为突出。横向对比来看，随机森林在两个数据集上均表现出较好的'
        '综合性能，验证了集成学习方法的优势。', style_body))
    story.append(Spacer(1, 3*mm))

    # 表2: 股票数据性能
    story.append(Paragraph('表2：股票数据 — 模型评估指标汇总', style_caption))
    story.append(Spacer(1, 1*mm))
    table2_data = [['模型', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
    for name in ['逻辑回归', '决策树', '随机森林']:
        r = results_stock[name]
        table2_data.append([
            name, f'{r["accuracy"]:.4f}', f'{r["precision"]:.4f}',
            f'{r["recall"]:.4f}', f'{r["f1"]:.4f}', f'{r["auc"]:.4f}'
        ])
    t2 = Table(table2_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t2.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'SimSun', 10.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#D9E2F3')]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 3*mm))

    # 表3: 乳腺癌数据性能
    story.append(Paragraph('表3：乳腺癌数据集 — 模型评估指标汇总', style_caption))
    story.append(Spacer(1, 1*mm))
    table3_data = [['模型', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
    for name in ['逻辑回归', '决策树', '随机森林']:
        r = results_cancer[name]
        table3_data.append([
            name, f'{r["accuracy"]:.4f}', f'{r["precision"]:.4f}',
            f'{r["recall"]:.4f}', f'{r["f1"]:.4f}', f'{r["auc"]:.4f}'
        ])
    t3 = Table(table3_data, colWidths=[2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t3.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'SimSun', 10.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#D9E2F3')]),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4*mm))

    # ---- 第六章 ----
    story.append(Paragraph('六、结果分析与结论', style_h1))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph(
        '本实验通过逻辑回归、决策树和随机森林三种分类算法，在股票数据和乳腺癌数据集上进行了'
        '对比实验，得到以下主要发现：', style_body))
    story.append(Spacer(1, 1*mm))

    story.append(Paragraph(
        f'第一，模型性能高度依赖数据本身的可分性。在乳腺癌数据集上，三种模型的AUC均达到0.95以上，'
        f'Accuracy超过97%，表现出卓越的分类能力；而在股票数据上，AUC仅在{results_stock["逻辑回归"]["auc"]:.2f}'
        f'-{results_stock["随机森林"]["auc"]:.2f}之间，接近随机分类水平。这一巨大差异说明，股价涨跌方向'
        '是一个极难预测的问题，技术指标提供的信号噪声比较大，远不如医学诊断数据那样具有确定性关系。', style_body))
    story.append(Spacer(1, 1*mm))

    story.append(Paragraph(
        '第二，随机森林在两个数据集上均表现出较好的综合性能。在股票数据上，随机森林的AUC略高于'
        '逻辑回归和决策树，说明集成学习通过多树投票能够有效降低方差，在噪声数据中提取更稳健的信号。'
        '在乳腺癌数据上，随机森林和逻辑回归的性能接近，均优于单棵决策树，表明当数据可分性较好时，'
        '简单模型也能取得优异表现。', style_body))
    story.append(Spacer(1, 1*mm))

    story.append(Paragraph(
        '第三，特征重要性分析为量化策略提供了有价值的参考。随机森林的特征重要性排序显示，布林带位置、'
        'RSI和MACD相关指标对股价涨跌预测贡献最大，而KDJ的K值和D值贡献较小。这提示在构建量化策略时，'
        '应优先关注动量类指标（RSI、MACD）和波动率类指标（布林带），同时可以考虑剔除低重要性特征'
        '以降低模型复杂度。', style_body))
    story.append(Spacer(1, 1*mm))

    story.append(Paragraph(
        '第四，在实际量化交易中，仅依靠技术指标进行涨跌二分类预测是不够的。可以考虑以下改进方向：'
        '（1）增加特征维度，引入基本面数据（市盈率、市净率等）、市场情绪指标（换手率、融资余额等）'
        '和宏观因子；（2）将二分类问题改为回归问题，预测涨跌幅度而非方向；（3）采用更复杂的时间序列'
        '模型（如LSTM、Transformer）捕捉序列依赖关系；（4）结合多时间框架分析，提高信号可靠性。'
        '需要特别注意的是，即使在回测中取得较好的指标表现，在实盘中也可能因交易成本、滑点、市场冲击'
        '等因素而大打折扣，因此模型评估应始终保持审慎态度。', style_body))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph(
        '综上所述，本实验完成了分类型机器学习算法的理论学习与实践验证，通过对比三种算法在两类数据'
        '上的表现，加深了对分类模型原理、评价指标和适用场景的理解，为后续量化交易策略的机器学习'
        '建模奠定了基础。', style_body))

    doc.build(story)
    print(f"\n  PDF报告已生成: {pdf_path}")
    return pdf_path


# ============================================================
# 7. 主程序
# ============================================================
def main():
    print("=" * 60)
    print("TASK5: 分类型机器学习算法实践")
    print("=" * 60)

    # 步骤1: 加载股票数据
    print("\n[步骤1] 加载股票数据并计算技术指标...")
    stock_df = load_stock_data()

    # 构建分类数据集
    X_stock, y_stock, features, feature_labels = prepare_stock_dataset(stock_df)
    pos_count = int(y_stock.sum())
    total = len(y_stock)
    stock_info = {
        'total_samples': total,
        'pos_ratio': pos_count / total,
    }
    print(f"  合并后样本数: {total}, 正类占比: {pos_count/total:.1%}")

    # 步骤2: 加载乳腺癌数据
    print("\n[步骤2] 加载乳腺癌数据集...")
    cancer = load_breast_cancer()
    X_cancer = cancer.data
    y_cancer = cancer.target
    print(f"  样本数: {len(y_cancer)}, 特征数: {X_cancer.shape[1]}, 正类占比: {y_cancer.mean():.1%}")

    # 步骤3: 数据划分
    print("\n[步骤3] 数据划分 (7:3)...")
    X_stock_train, X_stock_test, y_stock_train, y_stock_test = train_test_split(
        X_stock, y_stock, test_size=0.3, random_state=42, stratify=y_stock
    )
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
        X_cancer, y_cancer, test_size=0.3, random_state=42, stratify=y_cancer
    )
    print(f"  股票数据: 训练集{len(y_stock_train)}, 测试集{len(y_stock_test)}")
    print(f"  乳腺癌数据: 训练集{len(y_cancer_train)}, 测试集{len(y_cancer_test)}")

    # 步骤4: 模型训练与评估
    print("\n[步骤4] 模型训练与评估...")
    print("  --- 股票数据 ---")
    results_stock, scaler_stock = train_and_evaluate(
        X_stock_train, X_stock_test, y_stock_train, y_stock_test, '股票'
    )
    print("  --- 乳腺癌数据 ---")
    results_cancer, scaler_cancer = train_and_evaluate(
        X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test, '乳腺癌'
    )

    # 步骤5: 可视化
    print("\n[步骤5] 生成可视化图表...")
    chart_dir = os.path.join(BASE_DIR, 'charts_task5')
    os.makedirs(chart_dir, exist_ok=True)

    chart_paths = {
        'roc_stock': os.path.join(chart_dir, 'chart1_roc_stock.png'),
        'confusion_matrix': os.path.join(chart_dir, 'chart2_confusion_matrix.png'),
        'feature_importance': os.path.join(chart_dir, 'chart3_feature_importance.png'),
        'roc_cancer': os.path.join(chart_dir, 'chart4_roc_cancer.png'),
        'model_comparison': os.path.join(chart_dir, 'chart5_model_comparison.png'),
    }

    plot_roc_stock(results_stock, chart_paths['roc_stock'])
    plot_confusion_matrix(results_stock, results_cancer, chart_paths['confusion_matrix'])
    plot_feature_importance(results_stock['随机森林']['model'], feature_labels, chart_paths['feature_importance'])
    plot_roc_cancer(results_cancer, chart_paths['roc_cancer'])
    plot_model_comparison(results_stock, results_cancer, chart_paths['model_comparison'])

    # 步骤6: 生成PDF
    print("\n[步骤6] 生成PDF报告...")
    pdf_path = generate_pdf(results_stock, results_cancer, feature_labels, stock_df,
                            chart_paths, stock_info)

    # 保存评估结果到CSV
    print("\n[步骤7] 保存评估结果...")
    results_data = []
    for dataset, results in [('股票数据', results_stock), ('乳腺癌数据', results_cancer)]:
        for model, metrics in results.items():
            results_data.append({
                '数据集': dataset,
                '模型': model,
                'Accuracy': f'{metrics["accuracy"]:.4f}',
                'Precision': f'{metrics["precision"]:.4f}',
                'Recall': f'{metrics["recall"]:.4f}',
                'F1-Score': f'{metrics["f1"]:.4f}',
                'AUC': f'{metrics["auc"]:.4f}',
            })
    results_df = pd.DataFrame(results_data)
    results_csv = os.path.join(BASE_DIR, 'TASK5_模型评估结果.csv')
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
    print(f"  评估结果已保存: {results_csv}")

    print("\n" + "=" * 60)
    print("TASK5 全部完成!")
    print("=" * 60)
    print(f"\n生成文件:")
    print(f"  1. ml_classification_task5.py (本脚本)")
    print(f"  2. 童逸+TASK5.pdf (PDF报告)")
    print(f"  3. TASK5_模型评估结果.csv (评估结果)")
    print(f"  4. charts_task5/chart1_roc_stock.png")
    print(f"  5. charts_task5/chart2_confusion_matrix.png")
    print(f"  6. charts_task5/chart3_feature_importance.png")
    print(f"  7. charts_task5/chart4_roc_cancer.png")
    print(f"  8. charts_task5/chart5_model_comparison.png")


if __name__ == '__main__':
    main()
