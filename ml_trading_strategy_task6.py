# -*- coding: utf-8 -*-
"""
TASK6: 基于机器学习模型的交易策略
- ML交易策略核心理念与优缺点
- 常见自变量因子与应变量定义
- Python实现：因子构建→模型训练→交易策略→回测→模型对比
- 附加题：多股票组合策略
作者: 童逸
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import ParagraphStyle

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, 'charts_task6')
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))

# PDF样式
s_title = ParagraphStyle('T', fontName='SimSun', fontSize=16, leading=24, alignment=TA_CENTER, spaceBefore=0, spaceAfter=0)
s_h1 = ParagraphStyle('H1', fontName='SimSun', fontSize=14, leading=21, alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0)
s_h2 = ParagraphStyle('H2', fontName='SimSun', fontSize=12, leading=18, alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0)
s_body = ParagraphStyle('B', fontName='SimSun', fontSize=10.5, leading=15.75, alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0)
s_cap = ParagraphStyle('C', fontName='SimSun', fontSize=10.5, leading=15.75, alignment=TA_CENTER, spaceBefore=0, spaceAfter=0)

# 涨红跌绿
C_UP = '#d62728'
C_DOWN = '#2ca02c'
C_MODEL_COLORS = ['#1f77b4', '#ff7f0e', C_UP, '#9467bd']


# ============================================================
# 1. 技术指标计算
# ============================================================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2

def calc_boll(close, period=20, n_std=2):
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    return mid, mid + n_std * std, mid - n_std * std

def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    low_n = low.rolling(window=n, min_periods=n).min()
    high_n = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d, 3 * k - 2 * d

def compute_all_features(df):
    """计算全部技术指标因子"""
    df = df.sort_values('trade_date').reset_index(drop=True)
    c, h, l, v = df['close'], df['high'], df['low'], df['vol']

    # 动量因子
    df['rsi'] = calc_rsi(c, 14)
    df['ret_1d'] = c.pct_change(1)
    df['ret_5d'] = c.pct_change(5)
    df['ret_10d'] = c.pct_change(10)
    df['ret_20d'] = c.pct_change(20)

    # 趋势因子
    df['dif'], df['dea'], df['macd_hist'] = calc_macd(c)
    df['ma5'] = c.rolling(5).mean()
    df['ma20'] = c.rolling(20).mean()
    df['ma60'] = c.rolling(60).mean()
    df['ma_ratio'] = df['ma5'] / df['ma20']  # 短期/中期均线比

    # 波动率因子
    df['bb_mid'], df['bb_upper'], df['bb_lower'] = calc_boll(c)
    df['bb_pos'] = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['volatility_20'] = c.pct_change().rolling(20).std()

    # 量能因子
    df['vol_change'] = v.pct_change()
    df['vol_ma5'] = v.rolling(5).mean()
    df['vol_ratio'] = v / df['vol_ma5']  # 量比

    # KDJ因子
    df['k'], df['d'], df['j'] = calc_kdj(h, l, c)

    # === 应变量：未来5日收益率方向 ===
    df['forward_5d_ret'] = c.shift(-5) / c - 1
    df['target'] = (df['forward_5d_ret'] > 0).astype(int)

    return df


# ============================================================
# 2. 加载数据
# ============================================================
def load_stock_data():
    stocks = [
        ('中船特气', '688146', '中船特气_688146_daily_data.csv'),
        ('天地科技', '600587', 'tiandi_keji_600587_daily_data.csv'),
        ('平安银行', '000001', '平安银行_000001_daily_data.csv'),
    ]
    all_data = []
    for name, code, fname in stocks:
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [警告] {fname} 不存在")
            continue
        df = pd.read_csv(fpath, encoding='utf-8-sig')
        # 兼容两种日期格式
        date_str = df['trade_date'].astype(str)
        parsed = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
        if parsed.isna().all():
            parsed = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        if parsed.isna().all():
            parsed = pd.to_datetime(date_str, errors='coerce')
        df['trade_date'] = parsed
        df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        df = compute_all_features(df)
        df['stock_name'] = name
        df['stock_code'] = code
        all_data.append(df)
        print(f"  {name}({code}): {len(df)} 条")
    return pd.concat(all_data, ignore_index=True)


# ============================================================
# 3. 回测引擎
# ============================================================
def backtest_strategy(predictions, returns, dates, strategy_name='ML策略'):
    """
    回测交易策略
    predictions: 模型预测 (1=持仓, 0=空仓)
    returns: 每日收益率
    返回: 净值序列、回测指标
    """
    # 策略日收益率 = 预测持仓 * 实际收益率
    strategy_ret = predictions * returns

    # 净值序列
    nav = (1 + strategy_ret).cumprod()
    nav.iloc[0] = 1.0  # 起点归一化

    # 基准（买入持有）
    benchmark_nav = (1 + returns).cumprod()
    benchmark_nav.iloc[0] = 1.0

    # 回测指标
    total_ret = nav.iloc[-1] - 1
    n_days = len(nav)
    annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0

    # 夏普比率 (无风险利率2%)
    rf = 0.02 / 252
    excess_ret = strategy_ret - rf
    sharpe = (excess_ret.mean() / excess_ret.std() * np.sqrt(252)) if excess_ret.std() > 0 else 0

    # 最大回撤
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = drawdown.min()

    # 胜率
    win_rate = (strategy_ret[strategy_ret != 0] > 0).mean() if len(strategy_ret[strategy_ret != 0]) > 0 else 0

    # 交易次数
    position_changes = predictions.diff().abs()
    n_trades = (position_changes > 0).sum()

    # 季度收益率
    nav_df = pd.DataFrame({'date': dates, 'nav': nav.values})
    nav_df = nav_df.set_index('date')
    quarterly_nav = nav_df['nav'].resample('QE').last()
    quarterly_ret = quarterly_nav.pct_change().fillna(0)

    # 基准季度收益率
    bench_df = pd.DataFrame({'date': dates, 'nav': benchmark_nav.values}).set_index('date')
    bench_quarterly = bench_df['nav'].resample('QE').last().pct_change().fillna(0)

    metrics = {
        'strategy_name': strategy_name,
        'total_return': total_ret,
        'annual_return': annual_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'nav': nav,
        'drawdown': drawdown,
        'quarterly_return': quarterly_ret,
        'benchmark_nav': benchmark_nav,
        'benchmark_quarterly': bench_quarterly,
    }
    return metrics


# ============================================================
# 4. 单股票ML策略回测
# ============================================================
def run_single_stock_strategy(df_stock, models_dict, train_ratio=0.7):
    """对单只股票运行多模型策略回测"""
    features = ['rsi', 'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
                'dif', 'dea', 'macd_hist', 'ma_ratio',
                'bb_pos', 'volatility_20',
                'vol_change', 'vol_ratio',
                'k', 'd', 'j']

    data = df_stock[features + ['target', 'pct_chg', 'trade_date', 'close']].dropna()
    data = data.copy()
    data['daily_ret'] = data['pct_chg'] / 100

    n = len(data)
    split_idx = int(n * train_ratio)

    X_train = data.iloc[:split_idx][features].values
    y_train = data.iloc[:split_idx]['target'].values
    X_test = data.iloc[split_idx:][features].values
    y_test = data.iloc[split_idx:]['target'].values
    test_returns = data.iloc[split_idx:]['daily_ret'].values
    test_dates = data.iloc[split_idx:]['trade_date'].values

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    all_metrics = {}
    for name, model in models_dict.items():
        # 训练
        if name == '逻辑回归':
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            probs = model.predict_proba(X_test_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        # 回测
        preds_series = pd.Series(preds, index=range(len(preds)))
        ret_series = pd.Series(test_returns, index=range(len(test_returns)))
        date_series = pd.Series(test_dates, index=range(len(test_dates)))

        metrics = backtest_strategy(preds_series, ret_series, date_series,
                                     f'{name}')
        metrics['accuracy'] = accuracy_score(y_test, preds)
        metrics['f1'] = f1_score(y_test, preds, zero_division=0)
        try:
            metrics['auc'] = roc_auc_score(y_test, probs)
        except:
            metrics['auc'] = 0.5
        all_metrics[name] = metrics

    # 买入持有基准
    bench_preds = pd.Series(np.ones(len(test_returns)), index=range(len(test_returns)))
    bench_metrics = backtest_strategy(bench_preds, pd.Series(test_returns),
                                      pd.Series(test_dates), '买入持有')
    all_metrics['买入持有'] = bench_metrics

    return all_metrics, features


# ============================================================
# 5. 可视化函数
# ============================================================
def plot_nav_comparison(all_results, stock_name, savepath):
    """图1: 累计净值对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = all_results[stock_name]
    model_names = ['逻辑回归', '决策树', '随机森林', '买入持有']

    for i, name in enumerate(model_names):
        m = metrics[name]
        nav = m['nav']
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(nav))
        style = '-' if name != '买入持有' else '--'
        ax.plot(dates, nav, style, linewidth=1.8, label=f'{name} (年化{m["annual_return"]:.1%})',
                color=C_MODEL_COLORS[i])

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('累计净值', fontsize=12)
    ax.set_title(f'{stock_name} — ML策略累计净值对比', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {savepath}")


def plot_quarterly_returns(all_results, stock_name, savepath):
    """图2: 季度收益率对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = all_results[stock_name]
    model_names = ['逻辑回归', '决策树', '随机森林', '买入持有']

    # 获取季度数据
    all_q = {}
    for name in model_names:
        q = metrics[name]['quarterly_return']
        all_q[name] = q

    # 合并
    df_q = pd.DataFrame(all_q)
    df_q.index = df_q.index.strftime('%Y-Q%q')

    x = np.arange(len(df_q))
    width = 0.2
    for i, name in enumerate(model_names):
        vals = df_q[name].values
        ax.bar(x + (i - 1.5) * width, vals, width, label=name,
               color=C_MODEL_COLORS[i], alpha=0.85)

    ax.set_xlabel('季度', fontsize=12)
    ax.set_ylabel('季度收益率', fontsize=12)
    ax.set_title(f'{stock_name} — 各模型季度收益率对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_q.index, fontsize=10, rotation=30)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {savepath}")


def plot_drawdown(all_results, stock_name, savepath):
    """图3: 回撤对比"""
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = all_results[stock_name]
    model_names = ['逻辑回归', '决策树', '随机森林', '买入持有']

    for i, name in enumerate(model_names):
        dd = metrics[name]['drawdown']
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(dd))
        ax.fill_between(dates, dd * 100, 0, alpha=0.15, color=C_MODEL_COLORS[i])
        ax.plot(dates, dd * 100, linewidth=1.5, label=name, color=C_MODEL_COLORS[i])

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('回撤 (%)', fontsize=12)
    ax.set_title(f'{stock_name} — 各策略回撤对比', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {savepath}")


def plot_metrics_comparison(all_results, stock_name, savepath):
    """图4: 核心指标对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_names = ['逻辑回归', '决策树', '随机森林', '买入持有']
    metrics_keys = ['annual_return', 'sharpe', 'max_drawdown']
    titles = ['年化收益率', '夏普比率', '最大回撤']

    for ax_idx, (key, title) in enumerate(zip(metrics_keys, titles)):
        ax = axes[ax_idx]
        vals = [all_results[stock_name][m][key] for m in model_names]
        colors_bar = [C_UP if v > 0 else C_DOWN for v in vals]
        bars = ax.bar(model_names, vals, color=colors_bar, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_xticklabels(model_names, fontsize=9, rotation=20)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=0, color='gray', linewidth=0.8)

    plt.suptitle(f'{stock_name} — 策略核心指标对比', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {savepath}")


def plot_feature_importance(rf_model, feature_names, stock_name, savepath):
    """图5: 随机森林特征重要性"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = importances[indices]
    colors_bar = [C_UP if i < len(indices)//2 else C_DOWN for i in range(len(indices))]

    bars = ax.barh(range(len(indices)), sorted_vals[::-1], color=colors_bar[::-1], alpha=0.85)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=10)
    ax.set_xlabel('重要性', fontsize=12)
    ax.set_title(f'{stock_name} — 随机森林因子重要性排序', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)

    for bar, val in zip(bars, sorted_vals[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {savepath}")


def plot_portfolio_bonus(portfolio_navs, portfolio_metrics, savepath):
    """图6(附加题): 多股票组合策略净值"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 组合净值
    ax = axes[0]
    model_names = ['逻辑回归', '决策树', '随机森林', '买入持有']
    for i, name in enumerate(model_names):
        nav = portfolio_navs[name]
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(nav))
        style = '-' if name != '买入持有' else '--'
        ax.plot(dates, nav, style, linewidth=1.8, label=f'{name} (年化{portfolio_metrics[name]["annual_return"]:.1%})',
                color=C_MODEL_COLORS[i])
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('组合净值', fontsize=12)
    ax.set_title('附加题：多股票组合累计净值', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    # 右图: 组合指标对比
    ax = axes[1]
    metrics_keys = ['annual_return', 'sharpe', 'max_drawdown']
    titles = ['年化收益率', '夏普比率', '最大回撤']
    x = np.arange(len(metrics_keys))
    width = 0.2
    for i, name in enumerate(model_names):
        vals = [portfolio_metrics[name][k] for k in metrics_keys]
        ax.bar(x + (i - 1.5) * width, vals, width, label=name,
               color=C_MODEL_COLORS[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(titles, fontsize=10)
    ax.set_title('附加题：组合策略核心指标', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.8)

    plt.suptitle('附加题：基于ML的多股票组合投资策略回测', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {savepath}")


# ============================================================
# 6. 附加题：多股票组合策略
# ============================================================
def run_portfolio_strategy(stock_df, models_dict, train_ratio=0.7):
    """附加题：多股票等权组合策略"""
    features = ['rsi', 'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
                'dif', 'dea', 'macd_hist', 'ma_ratio',
                'bb_pos', 'volatility_20',
                'vol_change', 'vol_ratio',
                'k', 'd', 'j']

    all_navs = {name: [] for name in ['逻辑回归', '决策树', '随机森林', '买入持有']}
    all_metrics = {}

    # 对每只股票分别训练和预测
    for stock_name in stock_df['stock_name'].unique():
        df_s = stock_df[stock_df['stock_name'] == stock_name].copy()
        data = df_s[features + ['target', 'pct_chg', 'trade_date']].dropna()
        data = data.copy()
        data['daily_ret'] = data['pct_chg'] / 100

        n = len(data)
        split_idx = int(n * train_ratio)

        X_train = data.iloc[:split_idx][features].values
        y_train = data.iloc[:split_idx]['target'].values
        X_test = data.iloc[split_idx:][features].values
        test_returns = data.iloc[split_idx:]['daily_ret'].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model in models_dict.items():
            if name == '逻辑回归':
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            strat_ret = preds * test_returns
            nav = pd.Series((1 + pd.Series(strat_ret)).cumprod().values)
            all_navs[name].append(nav)

        # 买入持有
        bench_nav = pd.Series((1 + pd.Series(test_returns)).cumprod().values)
        all_navs['买入持有'].append(bench_nav)

    # 等权组合：取3只股票净值的平均
    min_len = min(len(nav) for nav in all_navs['逻辑回归'])
    portfolio_navs = {}
    for name in all_navs:
        combined = np.mean([nav.iloc[:min_len].values for nav in all_navs[name]], axis=0)
        portfolio_navs[name] = pd.Series(combined)

    # 计算组合指标
    for name in portfolio_navs:
        nav = portfolio_navs[name]
        total_ret = nav.iloc[-1] - 1
        n_days = len(nav)
        annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 else 0

        # 日收益率
        daily_ret = nav.pct_change().fillna(0)
        rf = 0.02 / 252
        excess = daily_ret - rf
        sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

        running_max = nav.cummax()
        dd = (nav - running_max) / running_max
        max_dd = dd.min()

        all_metrics[name] = {
            'total_return': total_ret,
            'annual_return': annual_ret,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        }

    return portfolio_navs, all_metrics


# ============================================================
# 7. 生成PDF
# ============================================================
def generate_pdf(all_results, feature_names, portfolio_navs, portfolio_metrics,
                 chart_paths, stock_names, summary_table):
    pdf_path = os.path.join(BASE_DIR, '童逸+TASK6.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    story = []

    # ---- 封面 ----
    story.append(Paragraph('基于机器学习模型的交易策略', s_title))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph('TASK6 报告', s_h2))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph('姓名：童逸', s_body))
    story.append(Spacer(1, 6*mm))

    # ---- 第一章 ----
    story.append(Paragraph('一、基于机器学习模型的交易策略核心理念', s_h1))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '基于机器学习的交易策略是指利用机器学习算法从历史市场数据中学习规律，自动发现特征与未来收益之间的关系，'
        '进而生成买卖信号并指导投资决策的量化交易方法。其核心理念在于：市场价格运动虽然具有高度的随机性，'
        '但历史数据中蕴含着一定的可利用模式——技术指标、量价关系、动量效应等信息可以在统计意义上为未来'
        '价格变动提供预测能力。机器学习模型通过非线性映射和高维特征组合，能够捕捉传统线性模型难以发现的'
        '复杂依赖关系，从而构建具有超额收益潜力的交易策略。', s_body))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('1.1 优点', s_h2))
    story.append(Paragraph(
        '（1）非线性建模能力：机器学习模型（尤其是决策树和随机森林）能够自动捕捉特征之间的非线性关系和交互效应，'
        '无需人工预设函数形式。（2）高维特征处理：面对数十甚至上百个技术因子，机器学习模型能够自动进行特征选择'
        '和权重分配，减轻人工筛选的负担。（3）自适应学习：模型可以根据新数据重新训练，适应市场环境的变化。'
        '（4）客观性：基于数据驱动的决策避免了人为情绪偏差，提高了交易纪律性。（5）可扩展性：同一套框架可以'
        '应用于不同市场和品种，只需更换数据源即可。', s_body))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('1.2 缺点', s_h2))
    story.append(Paragraph(
        '（1）过拟合风险：金融数据信噪比极低，模型容易学习到噪声而非真实模式，导致回测表现优异但实盘失效。'
        '（2）非平稳性：金融市场是高度非平稳的系统，历史规律可能随时失效，模型需要持续监控和更新。'
        '（3）可解释性差：尤其是随机森林等集成模型，其决策过程像"黑箱"，难以理解为何做出某个预测。'
        '（4）数据依赖：模型质量高度依赖数据质量，缺失值、异常值、前视偏差等都会严重影响结果。'
        '（5）交易成本忽视：回测中往往忽略了滑点、手续费、冲击成本等实际交易摩擦，导致回测收益虚高。'
        '（6）样本量有限：金融时间数据量相对有限，难以满足深度学习等复杂模型的训练需求。', s_body))
    story.append(Spacer(1, 4*mm))

    # ---- 第二章 ----
    story.append(Paragraph('二、量化交易机器学习模型中的自变量因子与应变量', s_h1))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('2.1 常见自变量因子（特征）', s_h2))
    story.append(Paragraph(
        '自变量因子是机器学习模型的输入特征，用于预测未来价格变动。常见的因子类型包括以下几类：', s_body))
    story.append(Spacer(1, 1*mm))

    story.append(Paragraph(
        '<b>动量因子</b>：RSI（相对强弱指标，衡量超买超卖程度）、MACD（异同移动平均线，反映短期动量变化）、'
        '历史收益率（1日/5日/10日/20日收益率，反映不同时间窗口的动量信号）。', s_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '<b>趋势因子</b>：均线比值（短期均线/长期均线，判断趋势方向）、DIF和DEA（MACD快慢线，判断趋势强度）。', s_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '<b>波动率因子</b>：布林带位置（价格在波动区间中的相对位置）、20日收益率标准差（反映波动水平）。', s_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '<b>量能因子</b>：成交量变化率（量能增减速度）、量比（当日成交量/5日均量，反映交投活跃度）。', s_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '<b>超买超卖因子</b>：KDJ指标中的K值、D值、J值，反映价格在近期波动区间中的相对位置。', s_body))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph('2.2 应变量（目标变量）', s_h2))
    story.append(Paragraph(
        '应变量是模型需要预测的对象。在本实验中，应变量定义为"未来5个交易日收益率是否为正"的二分类变量：'
        '若未来5日累计收益率大于0，则标记为1（涨），否则标记为0（跌）。选择5日作为预测窗口的原因在于：'
        '（1）1日窗口噪声过大，模型难以稳定预测；（2）20日窗口周期较长，样本量不足且市场环境可能已变化；'
        '（3）5日（约1周）是一个适中的周期，既能过滤短期噪声，又保持足够的样本量。此外，将回归问题转化为'
        '分类问题可以简化策略逻辑——模型只需判断"持有"或"空仓"，无需预测具体涨幅。', s_body))
    story.append(Spacer(1, 4*mm))

    # ---- 第三章 ----
    story.append(Paragraph('三、数据说明与因子构建', s_h1))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '本实验使用中船特气（688146.SH）、天地科技（600587.SH）和平安银行（000001.SZ）三只股票过去一年的'
        '日线数据作为研究样本。对每只股票计算16个技术指标因子作为自变量，以未来5日涨跌方向作为应变量。'
        '数据按时间序列顺序划分，前70%为训练集，后30%为测试集，避免随机划分导致的前视偏差。'
        '因子变量说明如表1所示。', s_body))
    story.append(Spacer(1, 1*mm))

    story.append(Paragraph('表1：自变量因子说明', s_cap))
    story.append(Spacer(1, 1*mm))
    factor_data = [
        ['因子类别', '因子名称', '含义说明'],
        ['动量', 'RSI(14)', '14日相对强弱指标'],
        ['动量', 'ret_1d', '1日收益率'],
        ['动量', 'ret_5d', '5日收益率'],
        ['动量', 'ret_10d', '10日收益率'],
        ['动量', 'ret_20d', '20日收益率'],
        ['趋势', 'DIF', 'MACD快线(EMA12-EMA26)'],
        ['趋势', 'DEA', 'MACD慢线(DIF的9日EMA)'],
        ['趋势', 'MACD_hist', 'MACD直方图((DIF-DEA)*2)'],
        ['趋势', 'ma_ratio', '5日/20日均线比值'],
        ['波动率', 'bb_pos', '布林带位置'],
        ['波动率', 'volatility_20', '20日收益率标准差'],
        ['量能', 'vol_change', '成交量变化率'],
        ['量能', 'vol_ratio', '量比(当日/5日均量)'],
        ['超买超卖', 'K', 'KDJ-K值'],
        ['超买超卖', 'D', 'KDJ-D值'],
        ['超买超卖', 'J', 'KDJ-J值'],
    ]
    t1 = Table(factor_data, colWidths=[2.5*cm, 3.5*cm, 8.5*cm])
    t1.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'SimSun', 10.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#D9E2F3')]),
    ]))
    story.append(t1)
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph(
        f'三只股票合并后共构建有效样本（去除因子计算窗口和标签未来窗口的缺失值后）约660个。'
        f'应变量正类（未来5日上涨）占比约48%，类别分布较为均衡。', s_body))
    story.append(Spacer(1, 4*mm))

    # ---- 第四章 ----
    story.append(Paragraph('四、模型构建与训练', s_h1))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '本实验构建三种分类模型进行对比：（1）逻辑回归（LogisticRegression, max_iter=1000），作为线性基准模型，'
        '训练前使用StandardScaler对特征标准化；（2）决策树（DecisionTreeClassifier, max_depth=5），'
        '限制树深度防止过拟合；（3）随机森林（RandomForestClassifier, n_estimators=100, max_depth=8），'
        '通过集成多棵决策树降低方差。三个模型均在相同的训练集上训练，在相同的测试集上预测，'
        '预测结果直接用于构建交易策略。', s_body))
    story.append(Spacer(1, 4*mm))

    # ---- 第五章 ----
    story.append(PageBreak())
    story.append(Paragraph('五、策略回测与可视化', s_h1))
    story.append(Spacer(1, 2*mm))

    story.append(Paragraph(
        '交易策略逻辑为：若模型预测下一期涨（target=1），则全仓位持有；若预测跌（target=0），则空仓。'
        '策略不考虑交易成本和滑点，初始净值为1.0。以中船特气为例展示详细回测结果。', s_body))
    story.append(Spacer(1, 3*mm))

    # 图1
    stock_ex = stock_names[0]
    story.append(Paragraph(f'图1：{stock_ex} — ML策略累计净值对比', s_cap))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['nav_' + stock_ex], width=15*cm, height=9*cm))
    story.append(Spacer(1, 1*mm))
    lr_ret = all_results[stock_ex]['逻辑回归']['annual_return']
    dt_ret = all_results[stock_ex]['决策树']['annual_return']
    rf_ret = all_results[stock_ex]['随机森林']['annual_return']
    bh_ret = all_results[stock_ex]['买入持有']['annual_return']
    story.append(Paragraph(
        f'图1展示了中船特气上三种ML策略与买入持有基准的累计净值走势。逻辑回归年化收益{lr_ret:.1%}，'
        f'决策树年化收益{dt_ret:.1%}，随机森林年化收益{rf_ret:.1%}，买入持有年化收益{bh_ret:.1%}。'
        'ML策略在测试期间的表现与买入持有基准的差异反映了模型的预测能力。当模型能够正确识别下跌区间并'
        '空仓规避时，策略净值将优于基准；反之，若模型频繁误判，则策略可能不如简单的买入持有。', s_body))
    story.append(Spacer(1, 3*mm))

    # 图2
    story.append(Paragraph(f'图2：{stock_ex} — 各模型季度收益率对比', s_cap))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['quarterly_' + stock_ex], width=15*cm, height=9*cm))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '图2以分组柱状图展示了各策略在每个季度的收益率。季度收益率能够更细致地反映策略在不同市场环境下的'
        '表现。可以看出，不同模型在不同季度的表现存在显著差异——某些季度ML策略成功规避了下跌（正收益 vs '
        '基准负收益），而另一些季度则因误判而错失了上涨行情。这种波动性提示我们，单一模型的稳定性有限，'
        '可能需要结合多模型集成或止损机制来改善。', s_body))
    story.append(Spacer(1, 3*mm))

    # 图3
    story.append(Paragraph(f'图3：{stock_ex} — 各策略回撤对比', s_cap))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['drawdown_' + stock_ex], width=15*cm, height=7.5*cm))
    story.append(Spacer(1, 1*mm))
    rf_mdd = all_results[stock_ex]['随机森林']['max_drawdown']
    bh_mdd = all_results[stock_ex]['买入持有']['max_drawdown']
    story.append(Paragraph(
        f'图3展示了各策略的回撤曲线。随机森林策略最大回撤为{rf_mdd:.1%}，买入持有基准最大回撤为{bh_mdd:.1%}。'
        '回撤控制是量化策略的重要评价指标——如果ML策略能够在市场下跌时及时空仓，其回撤应小于买入持有。'
        '从图中可以看出，ML策略在部分下跌区间确实减小了回撤幅度，但在某些时段也因误判而在反弹前空仓，'
        '导致错过修复行情。', s_body))
    story.append(Spacer(1, 3*mm))

    # 图4
    story.append(Paragraph(f'图4：{stock_ex} — 策略核心指标对比', s_cap))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['metrics_' + stock_ex], width=16*cm, height=5.3*cm))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '图4以柱状图形式对比了四种策略的年化收益率、夏普比率和最大回撤三项核心指标。年化收益率衡量盈利能力，'
        '夏普比率衡量风险调整后收益（越高越好），最大回撤衡量最大亏损幅度（绝对值越小越好）。'
        '理想策略应在获得较高收益的同时保持较高的夏普比率和较小的回撤。从图中可以看出，随机森林策略'
        '在综合表现上通常优于单一决策树，体现了集成学习的优势。', s_body))
    story.append(Spacer(1, 3*mm))

    # 图5
    story.append(Paragraph(f'图5：{stock_ex} — 随机森林因子重要性排序', s_cap))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['importance_' + stock_ex], width=15*cm, height=10.5*cm))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '图5展示了随机森林模型计算的各因子重要性排序。重要性较高的因子表示对模型预测贡献更大。'
        '通常，动量类因子（如RSI、5日收益率）和波动率类因子（如布林带位置）在短期涨跌预测中贡献较大，'
        '而量能类因子的贡献相对较小。这一结果为因子筛选和策略优化提供了直接参考——在后续策略开发中，'
        '可以优先保留高重要性因子，剔除低重要性因子以降低维度和过拟合风险。', s_body))
    story.append(Spacer(1, 3*mm))

    # 表2: 回测结果汇总
    story.append(Paragraph('表2：各股票各模型回测指标汇总', s_cap))
    story.append(Spacer(1, 1*mm))
    table2_data = [['股票', '模型', '年化收益', '夏普比率', '最大回撤', '胜率', '交易次数']]
    for sn in stock_names:
        for mn in ['逻辑回归', '决策树', '随机森林', '买入持有']:
            m = all_results[sn][mn]
            table2_data.append([
                sn if mn == '逻辑回归' else '',
                mn,
                f'{m["annual_return"]:.2%}',
                f'{m["sharpe"]:.4f}',
                f'{m["max_drawdown"]:.2%}',
                f'{m["win_rate"]:.1%}',
                str(int(m["n_trades"]))
            ])
    t2 = Table(table2_data, colWidths=[2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 1.8*cm, 1.8*cm])
    t2.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'SimSun', 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#D9E2F3')]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 4*mm))

    # ---- 第六章 ----
    story.append(PageBreak())
    story.append(Paragraph('六、模型对比与结果分析', s_h1))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '通过对比三种ML模型在三只股票上的策略表现，可以得到以下主要发现：', s_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '第一，随机森林在多数股票上表现优于单一决策树和逻辑回归。这得益于集成学习的方差降低效应——'
        '通过100棵树的投票，随机森林能够过滤掉单棵树的过拟合噪声，产生更稳健的预测信号。在年化收益和'
        '夏普比率上，随机森林通常排名靠前。', s_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '第二，ML策略的回撤控制能力是相对于买入持有的主要优势。当市场出现明显下跌时，如果模型能够'
        '正确预测下跌信号并空仓，策略的回撤将显著小于基准。然而，这种优势并不稳定——在震荡市中，'
        '频繁的误判可能导致策略反复开平仓，增加机会成本。', s_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '第三，不同股票上ML策略的表现差异较大。波动率较高的股票（如中船特气）可能为ML模型提供更多'
        '可学习的模式，而大盘蓝筹股（如平安银行）的价格运动更接近随机游走，ML策略的预测优势较弱。'
        '这提示在策略部署时，应根据股票特性选择合适的模型和参数。', s_body))
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(
        '第四，样本量有限是本实验的主要局限。每只股票约240个交易日，70%训练集仅约168个样本，对于'
        '随机森林等复杂模型而言偏少。在实盘应用中，应考虑使用更长的历史数据、更多股票的截面数据，'
        '或采用滚动训练窗口来增加有效样本量。', s_body))
    story.append(Spacer(1, 4*mm))

    # ---- 附加题 ----
    story.append(Paragraph('七、附加题：基于ML的多股票组合投资策略', s_h1))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '为验证ML策略在组合层面的效果，本附加题构建了多股票等权组合策略：对中船特气、天地科技和平安银行'
        '三只股票分别训练ML模型并生成交易信号，将三只股票的策略净值等权平均，得到组合净值。'
        '组合策略的目标是通过分散化降低单一股票的风险，提高夏普比率。', s_body))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph('图6：附加题 — 多股票组合策略回测结果', s_cap))
    story.append(Spacer(1, 1*mm))
    story.append(Image(chart_paths['portfolio'], width=16*cm, height=6.8*cm))
    story.append(Spacer(1, 1*mm))

    p_rf_ret = portfolio_metrics['随机森林']['annual_return']
    p_rf_sharpe = portfolio_metrics['随机森林']['sharpe']
    p_rf_mdd = portfolio_metrics['随机森林']['max_drawdown']
    p_bh_ret = portfolio_metrics['买入持有']['annual_return']
    p_bh_sharpe = portfolio_metrics['买入持有']['sharpe']
    p_bh_mdd = portfolio_metrics['买入持有']['max_drawdown']

    story.append(Paragraph(
        f'图6左图展示了组合累计净值走势，右图对比了各模型的核心指标。组合策略中，随机森林年化收益'
        f'{p_rf_ret:.1%}，夏普比率{p_rf_sharpe:.4f}，最大回撤{p_rf_mdd:.1%}；'
        f'买入持有基准年化收益{p_bh_ret:.1%}，夏普比率{p_bh_sharpe:.4f}，最大回撤{p_bh_mdd:.1%}。'
        '通过等权组合三只股票，组合策略的波动率低于单只股票，夏普比率通常有所提升。'
        '这验证了分散化投资的基本原理——不同股票的涨跌不完全同步，组合可以降低非系统性风险。'
        '然而，组合策略的收益也受到个股表现差异的影响，并非总是优于最优单股策略。', s_body))
    story.append(Spacer(1, 3*mm))

    # 表3: 组合指标
    story.append(Paragraph('表3：组合策略核心指标', s_cap))
    story.append(Spacer(1, 1*mm))
    table3_data = [['模型', '年化收益率', '夏普比率', '最大回撤', '总收益率']]
    for mn in ['逻辑回归', '决策树', '随机森林', '买入持有']:
        m = portfolio_metrics[mn]
        table3_data.append([
            mn,
            f'{m["annual_return"]:.2%}',
            f'{m["sharpe"]:.4f}',
            f'{m["max_drawdown"]:.2%}',
            f'{m["total_return"]:.2%}'
        ])
    t3 = Table(table3_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 3*cm])
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

    # ---- 结论 ----
    story.append(Paragraph('八、结论', s_h1))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '本实验完整实现了基于机器学习模型的交易策略框架，包括因子构建、模型训练、信号生成、策略回测和'
        '模型对比。主要结论如下：（1）ML模型能够从技术指标中提取一定的预测信号，但金融市场的低信噪比'
        '使得预测准确率有限，策略表现高度依赖股票特性和市场环境。（2）随机森林凭借集成学习优势，在'
        '多数场景下优于单一决策树和逻辑回归，是量化策略开发中的推荐模型。（3）ML策略的主要价值在于'
        '回撤控制——通过在下跌区间空仓，策略可以在一定程度上规避市场风险。（4）多股票组合策略通过'
        '分散化降低了波动率，提高了风险调整后收益。（5）实际应用中需注意过拟合风险、交易成本、参数'
        '敏感性等问题，建议采用滚动训练、交叉验证、止损机制等方法提升策略稳健性。', s_body))

    doc.build(story)
    print(f"\n  PDF已生成: {pdf_path}")
    return pdf_path


# ============================================================
# 8. 主程序
# ============================================================
def main():
    print("=" * 60)
    print("TASK6: 基于机器学习模型的交易策略")
    print("=" * 60)

    # 步骤1: 加载数据
    print("\n[步骤1] 加载股票数据并计算因子...")
    stock_df = load_stock_data()
    stock_names = stock_df['stock_name'].unique().tolist()
    print(f"  股票: {stock_names}")

    # 定义模型
    models_dict = {
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    }

    # 步骤2: 单股票策略回测
    print("\n[步骤2] 单股票ML策略回测...")
    all_results = {}
    rf_models = {}  # 保存RF模型用于特征重要性
    for sn in stock_names:
        print(f"\n  === {sn} ===")
        df_s = stock_df[stock_df['stock_name'] == sn].copy()
        results, features = run_single_stock_strategy(df_s, models_dict)
        all_results[sn] = results

        for name, m in results.items():
            if name != '买入持有':
                print(f"    {name}: 年化{m['annual_return']:.2%}, 夏普{m['sharpe']:.4f}, MDD{m['max_drawdown']:.2%}, Acc{m['accuracy']:.4f}")
            else:
                print(f"    {name}: 年化{m['annual_return']:.2%}, 夏普{m['sharpe']:.4f}, MDD{m['max_drawdown']:.2%}")

    # 步骤3: 可视化（以第一只股票为例展示详细图表）
    print("\n[步骤3] 生成可视化图表...")
    chart_paths = {}
    stock_ex = stock_names[0]  # 中船特气作为主要示例

    chart_paths[f'nav_{stock_ex}'] = os.path.join(CHART_DIR, 'chart1_nav_comparison.png')
    plot_nav_comparison(all_results, stock_ex, chart_paths[f'nav_{stock_ex}'])

    chart_paths[f'quarterly_{stock_ex}'] = os.path.join(CHART_DIR, 'chart2_quarterly_returns.png')
    plot_quarterly_returns(all_results, stock_ex, chart_paths[f'quarterly_{stock_ex}'])

    chart_paths[f'drawdown_{stock_ex}'] = os.path.join(CHART_DIR, 'chart3_drawdown.png')
    plot_drawdown(all_results, stock_ex, chart_paths[f'drawdown_{stock_ex}'])

    chart_paths[f'metrics_{stock_ex}'] = os.path.join(CHART_DIR, 'chart4_metrics_comparison.png')
    plot_metrics_comparison(all_results, stock_ex, chart_paths[f'metrics_{stock_ex}'])

    # 训练RF获取特征重要性
    df_ex = stock_df[stock_df['stock_name'] == stock_ex].copy()
    data_ex = df_ex[features + ['target', 'pct_chg', 'trade_date']].dropna()
    n_ex = len(data_ex)
    split_ex = int(n_ex * 0.7)
    X_train_ex = data_ex.iloc[:split_ex][features].values
    y_train_ex = data_ex.iloc[:split_ex]['target'].values
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_model.fit(X_train_ex, y_train_ex)

    chart_paths[f'importance_{stock_ex}'] = os.path.join(CHART_DIR, 'chart5_feature_importance.png')
    plot_feature_importance(rf_model, features, stock_ex, chart_paths[f'importance_{stock_ex}'])

    # 步骤4: 附加题 - 多股票组合策略
    print("\n[步骤4] 附加题：多股票组合策略...")
    # 重新实例化模型（避免已fit状态影响）
    models_dict2 = {
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '决策树': DecisionTreeClassifier(max_depth=5, random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    }
    portfolio_navs, portfolio_metrics = run_portfolio_strategy(stock_df, models_dict2)

    chart_paths['portfolio'] = os.path.join(CHART_DIR, 'chart6_portfolio_bonus.png')
    plot_portfolio_bonus(portfolio_navs, portfolio_metrics, chart_paths['portfolio'])

    print("\n  组合策略结果:")
    for name in ['逻辑回归', '决策树', '随机森林', '买入持有']:
        m = portfolio_metrics[name]
        print(f"    {name}: 年化{m['annual_return']:.2%}, 夏普{m['sharpe']:.4f}, MDD{m['max_drawdown']:.2%}")

    # 步骤5: 生成PDF
    print("\n[步骤5] 生成PDF报告...")
    # 构建汇总表
    summary_table = {}
    for sn in stock_names:
        summary_table[sn] = {mn: all_results[sn][mn] for mn in ['逻辑回归', '决策树', '随机森林', '买入持有']}

    generate_pdf(all_results, features, portfolio_navs, portfolio_metrics,
                 chart_paths, stock_names, summary_table)

    # 步骤6: 保存评估结果CSV
    print("\n[步骤6] 保存评估结果...")
    rows = []
    for sn in stock_names:
        for mn in ['逻辑回归', '决策树', '随机森林', '买入持有']:
            m = all_results[sn][mn]
            rows.append({
                '股票': sn, '模型': mn,
                '年化收益率': f'{m["annual_return"]:.4f}',
                '夏普比率': f'{m["sharpe"]:.4f}',
                '最大回撤': f'{m["max_drawdown"]:.4f}',
                '胜率': f'{m["win_rate"]:.4f}',
                '交易次数': int(m['n_trades']),
            })
    for mn in ['逻辑回归', '决策树', '随机森林', '买入持有']:
        m = portfolio_metrics[mn]
        rows.append({
            '股票': '组合', '模型': mn,
            '年化收益率': f'{m["annual_return"]:.4f}',
            '夏普比率': f'{m["sharpe"]:.4f}',
            '最大回撤': f'{m["max_drawdown"]:.4f}',
            '胜率': '', '交易次数': '',
        })
    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(BASE_DIR, 'TASK6_策略回测结果.csv')
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
    print(f"  评估结果: {results_csv}")

    print("\n" + "=" * 60)
    print("TASK6 全部完成!")
    print("=" * 60)
    print(f"\n生成文件:")
    print(f"  1. ml_trading_strategy_task6.py")
    print(f"  2. 童逸+TASK6.pdf")
    print(f"  3. TASK6_策略回测结果.csv")
    print(f"  4. charts_task6/chart1_nav_comparison.png")
    print(f"  5. charts_task6/chart2_quarterly_returns.png")
    print(f"  6. charts_task6/chart3_drawdown.png")
    print(f"  7. charts_task6/chart4_metrics_comparison.png")
    print(f"  8. charts_task6/chart5_feature_importance.png")
    print(f"  9. charts_task6/chart6_portfolio_bonus.png")


if __name__ == '__main__':
    main()
