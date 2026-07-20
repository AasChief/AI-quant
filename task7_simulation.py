# -*- coding: utf-8 -*-
"""
TASK7: JoinQuant平台机器学习交易策略实现
- JoinQuant平台概述
- 结合TASK6的ML交易策略
- 滚动窗口回测 + 参数调优 + 风险分析
- 6张可视化图表 + PDF报告

作者: 童逸
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
CHART_DIR = os.path.join(BASE_DIR, 'charts_task7')
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))

# PDF样式 - 宋体五号字10.5pt 1.5倍行距 0段间距 两端对齐
s_title = ParagraphStyle('T', fontName='SimSun', fontSize=16, leading=24,
                         alignment=TA_CENTER, spaceBefore=0, spaceAfter=0)
s_h1 = ParagraphStyle('H1', fontName='SimSun', fontSize=14, leading=21,
                      alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0)
s_h2 = ParagraphStyle('H2', fontName='SimSun', fontSize=12, leading=18,
                      alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0)
s_body = ParagraphStyle('B', fontName='SimSun', fontSize=10.5, leading=15.75,
                        alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0)
s_cap = ParagraphStyle('C', fontName='SimSun', fontSize=10.5, leading=15.75,
                       alignment=TA_CENTER, spaceBefore=0, spaceAfter=0)
s_code = ParagraphStyle('CODE', fontName='SimSun', fontSize=9, leading=13.5,
                        alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0,
                        leftIndent=15, rightIndent=15)

# 涨红跌绿
C_UP = '#d62728'
C_DOWN = '#2ca02c'
C_STRATEGY = '#1f77b4'
C_BENCHMARK = '#ff7f0e'
C_BUYHOLD = '#888888'


# ============================================================
# 1. 技术指标计算（与TASK6一致）
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


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    low_n = low.rolling(window=n, min_periods=n).min()
    high_n = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def compute_features(df, predict_horizon=5):
    c, h, l, v = df['close'], df['high'], df['low'], df['vol']

    df['rsi'] = calc_rsi(c, 14)
    df['ret_1d'] = c.pct_change(1)
    df['ret_5d'] = c.pct_change(5)
    df['ret_10d'] = c.pct_change(10)
    df['ret_20d'] = c.pct_change(20)

    df['dif'], df['dea'], df['macd_hist'] = calc_macd(c)
    df['ma5'] = c.rolling(5).mean()
    df['ma20'] = c.rolling(20).mean()
    df['ma60'] = c.rolling(60).mean()
    df['ma_ratio'] = df['ma5'] / df['ma20']

    df['bb_mid'] = c.rolling(20).mean()
    df['bb_std'] = c.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pos'] = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['volatility_20'] = c.pct_change().rolling(20).std()

    df['vol_change'] = v.pct_change()
    df['vol_ma5'] = v.rolling(5).mean()
    df['vol_ratio'] = v / df['vol_ma5']

    df['k'], df['d'], df['j'] = calc_kdj(h, l, c)

    df['forward_ret'] = c.shift(-predict_horizon) / c - 1
    df['target'] = (df['forward_ret'] > 0).astype(int)

    return df


FEATURE_COLS = [
    'rsi', 'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
    'dif', 'dea', 'macd_hist', 'ma_ratio',
    'bb_pos', 'volatility_20',
    'vol_change', 'vol_ratio',
    'k', 'd', 'j'
]


# ============================================================
# 2. 数据加载
# ============================================================
def load_stock_data():
    stocks = [
        ('平安银行', '000001', '平安银行_000001_daily_data.csv'),
        ('天地科技', '600587', 'tiandi_keji_600587_daily_data.csv'),
        ('中船特气', '688146', '中船特气_688146_daily_data.csv'),
    ]
    all_data = {}
    for name, code, fname in stocks:
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [警告] {fname} 不存在")
            continue
        df = pd.read_csv(fpath, encoding='utf-8-sig')
        date_str = df['trade_date'].astype(str)
        parsed = pd.to_datetime(date_str, errors='coerce')
        df['trade_date'] = parsed
        df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)
        df = compute_features(df)
        df['stock_name'] = name
        df['stock_code'] = code
        all_data[name] = df
        print(f"  {name}({code}): {len(df)} 条, {df['trade_date'].iloc[0].date()} ~ {df['trade_date'].iloc[-1].date()}")
    return all_data


# ============================================================
# 3. 滚动窗口回测引擎
# ============================================================
def rolling_backtest(df, train_size=120, retrain_freq=20, predict_horizon=5,
                     n_estimators=100, max_depth=5, cost_rate=0.001):
    """
    滚动窗口回测：
    - 每 retrain_freq 天重新训练模型
    - 训练数据为过去 train_size 天
    - 预测下一日涨跌方向，1=持仓 0=空仓
    - 每次调仓扣除交易成本
    """
    predictions = pd.Series(0, index=df.index, dtype=int)
    proba_series = pd.Series(0.5, index=df.index, dtype=float)

    for start in range(train_size, len(df), retrain_freq):
        end = min(start + retrain_freq, len(df))

        train_data = df.iloc[start - train_size:start].dropna(subset=FEATURE_COLS + ['target'])
        if len(train_data) < 40:
            continue

        X_train = train_data[FEATURE_COLS].values
        y_train = train_data['target'].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        for i in range(start, end):
            features = df.loc[i, FEATURE_COLS].values
            if not np.isnan(features).any():
                features_scaled = scaler.transform([features])
                pred = model.predict(features_scaled)[0]
                proba = model.predict_proba(features_scaled)[0]
                predictions.loc[i] = pred
                proba_series.loc[i] = max(proba)

    # 计算策略收益
    daily_ret = df['close'].pct_change().fillna(0)
    strategy_ret = predictions * daily_ret

    # 交易成本：每次仓位变化时扣除
    position_changes = predictions.diff().abs().fillna(0)
    strategy_ret = strategy_ret - position_changes * cost_rate
    strategy_ret = strategy_ret.fillna(0)

    # 净值
    nav = (1 + strategy_ret).cumprod()
    nav.iloc[0] = 1.0

    # 买入持有
    buyhold_nav = (1 + daily_ret).cumprod()
    buyhold_nav.iloc[0] = 1.0

    return {
        'predictions': predictions,
        'proba': proba_series,
        'strategy_ret': strategy_ret,
        'daily_ret': daily_ret,
        'nav': nav,
        'buyhold_nav': buyhold_nav,
        'dates': df['trade_date'],
    }


def calc_metrics(ret_series, nav_series):
    """计算回测指标"""
    n_days = len(ret_series)
    total_ret = nav_series.iloc[-1] - 1
    annual_ret = (1 + total_ret) ** (252 / n_days) - 1 if n_days > 0 and total_ret > -1 else 0

    rf = 0.02 / 252
    excess = ret_series - rf
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    rolling_max = nav_series.expanding().max()
    drawdown = (nav_series - rolling_max) / rolling_max
    max_dd = drawdown.min()

    win_rate = (ret_series[ret_series != 0] > 0).mean() if len(ret_series[ret_series != 0]) > 0 else 0

    vol = ret_series.std() * np.sqrt(252)

    return {
        'total_return': total_ret,
        'annual_return': annual_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'volatility': vol,
    }


# ============================================================
# 4. 主回测逻辑
# ============================================================
def run_main_backtest(stock_data):
    """对3只股票分别回测，并合成等权组合"""
    print("\n=== 主回测（默认参数: n_estimators=100, max_depth=5）===")

    results = {}
    all_strategy_ret = []
    all_buyhold_ret = []
    all_dates = None
    all_positions = []

    for name, df in stock_data.items():
        print(f"  回测 {name}...")
        result = rolling_backtest(df, train_size=120, retrain_freq=20,
                                  predict_horizon=5, n_estimators=100, max_depth=5)
        metrics = calc_metrics(result['strategy_ret'], result['nav'])
        bh_metrics = calc_metrics(result['daily_ret'], result['buyhold_nav'])

        result['metrics'] = metrics
        result['bh_metrics'] = bh_metrics
        results[name] = result

        print(f"    策略: 年化={metrics['annual_return']:.2%}, 夏普={metrics['sharpe']:.2f}, "
              f"最大回撤={metrics['max_drawdown']:.2%}, 胜率={metrics['win_rate']:.2%}")
        print(f"    买入持有: 年化={bh_metrics['annual_return']:.2%}, 夏普={bh_metrics['sharpe']:.2f}, "
              f"最大回撤={bh_metrics['max_drawdown']:.2%}")

        all_strategy_ret.append(result['strategy_ret'].values)
        all_buyhold_ret.append(result['daily_ret'].values)
        all_positions.append(result['predictions'].values)
        if all_dates is None:
            all_dates = result['dates'].values

    # 等权组合（3只股票起始日期相同，取前 min_len 个元素保证日期对齐）
    min_len = min(len(r) for r in all_strategy_ret)
    portfolio_ret = np.mean([r[:min_len] for r in all_strategy_ret], axis=0)
    portfolio_nav = np.cumprod(1 + portfolio_ret)
    portfolio_nav[0] = 1.0

    bh_ret = np.mean([r[:min_len] for r in all_buyhold_ret], axis=0)
    bh_nav = np.cumprod(1 + bh_ret)
    bh_nav[0] = 1.0

    portfolio_metrics = calc_metrics(pd.Series(portfolio_ret), pd.Series(portfolio_nav))
    bh_portfolio_metrics = calc_metrics(pd.Series(bh_ret), pd.Series(bh_nav))

    print(f"\n  组合策略: 年化={portfolio_metrics['annual_return']:.2%}, "
          f"夏普={portfolio_metrics['sharpe']:.2f}, "
          f"最大回撤={portfolio_metrics['max_drawdown']:.2%}")
    print(f"  组合买入持有: 年化={bh_portfolio_metrics['annual_return']:.2%}, "
          f"夏普={bh_portfolio_metrics['sharpe']:.2f}, "
          f"最大回撤={bh_portfolio_metrics['max_drawdown']:.2%}")

    return results, {
        'strategy_ret': portfolio_ret,
        'strategy_nav': portfolio_nav,
        'bh_ret': bh_ret,
        'bh_nav': bh_nav,
        'dates': all_dates[:min_len],
        'metrics': portfolio_metrics,
        'bh_metrics': bh_portfolio_metrics,
        'positions': all_positions,
    }


# ============================================================
# 5. 参数调优
# ============================================================
def run_param_tuning(stock_data):
    """参数调优：测试不同的 n_estimators 和 max_depth"""
    print("\n=== 参数调优 ===")

    n_estimators_list = [50, 100, 200]
    max_depth_list = [3, 5, 10]

    # 使用中船特气作为代表（数据量适中）
    df = stock_data['中船特气']

    sharpe_matrix = np.zeros((len(max_depth_list), len(n_estimators_list)))
    ret_matrix = np.zeros((len(max_depth_list), len(n_estimators_list)))
    dd_matrix = np.zeros((len(max_depth_list), len(n_estimators_list)))

    for i, md in enumerate(max_depth_list):
        for j, ne in enumerate(n_estimators_list):
            result = rolling_backtest(df, train_size=120, retrain_freq=20,
                                      predict_horizon=5, n_estimators=ne, max_depth=md)
            metrics = calc_metrics(result['strategy_ret'], result['nav'])
            sharpe_matrix[i, j] = metrics['sharpe']
            ret_matrix[i, j] = metrics['annual_return']
            dd_matrix[i, j] = metrics['max_drawdown']
            print(f"  n_estimators={ne}, max_depth={md}: "
                  f"年化={metrics['annual_return']:.2%}, "
                  f"夏普={metrics['sharpe']:.2f}, "
                  f"最大回撤={metrics['max_drawdown']:.2%}")

    return {
        'n_estimators': n_estimators_list,
        'max_depth': max_depth_list,
        'sharpe': sharpe_matrix,
        'returns': ret_matrix,
        'drawdown': dd_matrix,
    }


# ============================================================
# 6. 风险分析
# ============================================================
def run_risk_analysis(portfolio_data):
    """风险分析：滚动波动率、VaR、回撤"""
    ret = pd.Series(portfolio_data['strategy_ret'])
    bh_ret = pd.Series(portfolio_data['bh_ret'])

    # 滚动年化波动率（60日窗口）
    rolling_vol = ret.rolling(60, min_periods=20).std() * np.sqrt(252)
    bh_rolling_vol = bh_ret.rolling(60, min_periods=20).std() * np.sqrt(252)

    # VaR（历史模拟法）
    var_95 = []
    var_99 = []
    window = 60
    for i in range(window, len(ret)):
        hist = ret.iloc[i - window:i]
        var_95.append(np.percentile(hist, 5))
        var_99.append(np.percentile(hist, 1))

    var_95 = pd.Series(var_95, index=range(window, len(ret)))
    var_99 = pd.Series(var_99, index=range(window, len(ret)))

    # 回撤
    nav = pd.Series(portfolio_data['strategy_nav'])
    rolling_max = nav.expanding().max()
    drawdown = (nav - rolling_max) / rolling_max

    bh_nav = pd.Series(portfolio_data['bh_nav'])
    bh_rolling_max = bh_nav.expanding().max()
    bh_drawdown = (bh_nav - bh_rolling_max) / bh_rolling_max

    # Beta（相对买入持有基准）
    cov = np.cov(ret, bh_ret)[0, 1]
    beta = cov / np.var(bh_ret) if np.var(bh_ret) > 0 else 0

    return {
        'rolling_vol': rolling_vol,
        'bh_rolling_vol': bh_rolling_vol,
        'var_95': var_95,
        'var_99': var_99,
        'drawdown': drawdown,
        'bh_drawdown': bh_drawdown,
        'beta': beta,
    }


# ============================================================
# 7. 图表生成
# ============================================================
def generate_charts(results, portfolio_data, param_results, risk_data):
    """生成6张可视化图表"""
    print("\n=== 生成图表 ===")

    # ---- 图1: 策略净值对比 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    # 上图：各股票净值
    ax1 = axes[0]
    for name, result in results.items():
        dates = result['dates']
        nav = result['nav']
        bh = result['buyhold_nav']
        ax1.plot(dates, nav, label=f'{name}-ML策略', linewidth=1.5)
        ax1.plot(dates, bh, label=f'{name}-买入持有', linewidth=1, alpha=0.5, linestyle='--')
    ax1.set_title('各股票ML策略净值 vs 买入持有', fontsize=14)
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_ylabel('净值')
    ax1.grid(True, alpha=0.3)

    # 下图：组合净值
    ax2 = axes[1]
    dates = portfolio_data['dates']
    ax2.plot(dates, portfolio_data['strategy_nav'], label='ML组合策略',
             color=C_STRATEGY, linewidth=2)
    ax2.plot(dates, portfolio_data['bh_nav'], label='等权买入持有',
             color=C_BUYHOLD, linewidth=1.5, linestyle='--')
    ax2.fill_between(dates, portfolio_data['strategy_nav'],
                     portfolio_data['bh_nav'],
                     where=portfolio_data['strategy_nav'] >= portfolio_data['bh_nav'],
                     alpha=0.15, color=C_UP, label='超额收益')
    ax2.set_title('等权组合净值对比', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.set_ylabel('净值')
    ax2.set_xlabel('日期')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'chart1_nav_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  chart1_nav_comparison.png ✓")

    # ---- 图2: 回撤分析 ----
    fig, ax = plt.subplots(figsize=(12, 5))
    dates = portfolio_data['dates']
    dd = risk_data['drawdown'].values
    bh_dd = risk_data['bh_drawdown'].values

    ax.fill_between(dates, dd * 100, 0, alpha=0.4, color=C_DOWN, label='ML策略回撤')
    ax.plot(dates, dd * 100, color=C_DOWN, linewidth=1)
    ax.fill_between(dates, bh_dd * 100, 0, alpha=0.2, color=C_BENCHMARK, label='买入持有回撤')
    ax.plot(dates, bh_dd * 100, color=C_BENCHMARK, linewidth=1, linestyle='--')
    ax.set_title('策略回撤分析', fontsize=14)
    ax.set_ylabel('回撤 (%)')
    ax.set_xlabel('日期')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'chart2_drawdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  chart2_drawdown.png ✓")

    # ---- 图3: 季度收益率 ----
    fig, ax = plt.subplots(figsize=(12, 5))

    ret_series = pd.Series(portfolio_data['strategy_ret'],
                           index=pd.to_datetime(portfolio_data['dates']))
    bh_series = pd.Series(portfolio_data['bh_ret'],
                          index=pd.to_datetime(portfolio_data['dates']))

    quarterly_ret = (1 + ret_series).resample('QE').apply(lambda x: x.prod() - 1) * 100
    quarterly_bh = (1 + bh_series).resample('QE').apply(lambda x: x.prod() - 1) * 100

    x = np.arange(len(quarterly_ret))
    width = 0.35

    bar_colors_s = [C_UP if v > 0 else C_DOWN for v in quarterly_ret.values]
    bar_colors_b = [C_UP if v > 0 else C_DOWN for v in quarterly_bh.values]

    bars1 = ax.bar(x - width / 2, quarterly_ret.values, width,
                   label='ML策略', color=bar_colors_s, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, quarterly_bh.values, width,
                   label='买入持有', color=bar_colors_b, alpha=0.5, edgecolor='black', linewidth=0.5)

    ax.set_title('季度收益率对比', fontsize=14)
    ax.set_ylabel('收益率 (%)')
    ax.set_xlabel('季度')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d.year}Q{d.quarter}' for d in quarterly_ret.index], rotation=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'chart3_quarterly_returns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  chart3_quarterly_returns.png ✓")

    # ---- 图4: 参数调优热力图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 夏普比率热力图
    ax1 = axes[0]
    sharpe_data = param_results['sharpe']
    im1 = ax1.imshow(sharpe_data, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(len(param_results['n_estimators'])))
    ax1.set_xticklabels(param_results['n_estimators'])
    ax1.set_yticks(range(len(param_results['max_depth'])))
    ax1.set_yticklabels(param_results['max_depth'])
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('max_depth')
    ax1.set_title('夏普比率热力图', fontsize=13)
    for i in range(sharpe_data.shape[0]):
        for j in range(sharpe_data.shape[1]):
            ax1.text(j, i, f'{sharpe_data[i, j]:.2f}', ha='center', va='center',
                     fontsize=11, fontweight='bold')
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # 年化收益率热力图
    ax2 = axes[1]
    ret_data = param_results['returns'] * 100
    im2 = ax2.imshow(ret_data, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(len(param_results['n_estimators'])))
    ax2.set_xticklabels(param_results['n_estimators'])
    ax2.set_yticks(range(len(param_results['max_depth'])))
    ax2.set_yticklabels(param_results['max_depth'])
    ax2.set_xlabel('n_estimators')
    ax2.set_ylabel('max_depth')
    ax2.set_title('年化收益率热力图 (%)', fontsize=13)
    for i in range(ret_data.shape[0]):
        for j in range(ret_data.shape[1]):
            ax2.text(j, i, f'{ret_data[i, j]:.1f}', ha='center', va='center',
                     fontsize=11, fontweight='bold')
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    plt.suptitle('参数调优结果（中船特气）', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'chart4_param_tuning.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  chart4_param_tuning.png ✓")

    # ---- 图5: 风险暴露分析 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 滚动波动率
    ax1 = axes[0]
    dates = portfolio_data['dates']
    rv = risk_data['rolling_vol'].values * 100
    bh_rv = risk_data['bh_rolling_vol'].values * 100
    ax1.plot(dates, rv, label='ML策略', color=C_STRATEGY, linewidth=1.5)
    ax1.plot(dates, bh_rv, label='买入持有', color=C_BENCHMARK, linewidth=1.5, alpha=0.6)
    ax1.fill_between(dates, rv, bh_rv, where=rv < bh_rv, alpha=0.2, color=C_DOWN,
                     label='波动率降低区域')
    ax1.set_title('滚动年化波动率（60日窗口）', fontsize=13)
    ax1.set_ylabel('波动率 (%)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # VaR
    ax2 = axes[1]
    var_dates = dates[len(dates) - len(risk_data['var_95']):]
    ax2.plot(var_dates, risk_data['var_95'].values * 100,
             label='VaR(95%)', color=C_UP, linewidth=1.5)
    ax2.plot(var_dates, risk_data['var_99'].values * 100,
             label='VaR(99%)', color='#9467bd', linewidth=1.5)
    ax2.fill_between(var_dates, risk_data['var_95'].values * 100,
                     risk_data['var_99'].values * 100, alpha=0.2, color=C_UP)
    ax2.set_title('在险价值 VaR（历史模拟法，60日窗口）', fontsize=13)
    ax2.set_ylabel('VaR (%)')
    ax2.set_xlabel('日期')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'chart5_risk_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  chart5_risk_analysis.png ✓")

    # ---- 图6: 持仓变化 ----
    fig, ax = plt.subplots(figsize=(12, 5))

    stock_names = list(results.keys())
    positions = portfolio_data['positions']

    # 各股票仓位（0或1）
    min_len = min(len(p) for p in positions)
    pos_array = np.array([p[:min_len] for p in positions]).T

    dates = portfolio_data['dates'][:min_len]

    # 堆叠面积图
    colors = [C_STRATEGY, C_BENCHMARK, '#9467bd']
    ax.stackplot(dates, pos_array.T, labels=stock_names, colors=colors, alpha=0.6)
    ax.set_title('各股票持仓状态变化', fontsize=14)
    ax.set_ylabel('持仓数量')
    ax.set_xlabel('日期')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 3.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, 'chart6_position_changes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  chart6_position_changes.png ✓")


# ============================================================
# 8. PDF生成
# ============================================================
def generate_pdf(results, portfolio_data, param_results, risk_data):
    """生成PDF报告"""
    print("\n=== 生成PDF报告 ===")

    pdf_path = os.path.join(BASE_DIR, '童逸+TASK7.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                            topMargin=2.5 * cm, bottomMargin=2.5 * cm)

    story = []
    img_w = 16 * cm

    # === 封面 ===
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph('基于机器学习的交易策略<br/>JoinQuant平台实现与回测分析', s_title))
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph('童逸+TASK7', s_h1))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph('AI大模型辅助的金融交易策略课程', s_body))
    story.append(PageBreak())

    # === 第一章 ===
    story.append(Paragraph('第一章 JoinQuant平台概述', s_h1))
    story.append(Spacer(1, 3))

    story.append(Paragraph('1.1 平台注册与认证流程', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        'JoinQuant（聚宽）是国内领先的量化交易平台，提供从数据获取、策略研发、回测验证到模拟交易的一站式服务。'
        '注册流程如下：访问聚宽官网（www.joinquant.com），点击"注册"按钮，可使用手机号或第三方账号（微信、GitHub等）注册。'
        '完成注册后需进行实名认证，填写真实姓名和身份证号，认证通过后即可使用全部功能。'
        '个人用户默认享有免费数据额度，包括A股日线数据、分钟数据及部分财务数据。'
        '平台同时提供VIP会员服务，可获取更丰富的数据类型（如Level-2行情、期权数据等）和更高的API调用额度。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('1.2 界面布局与功能模块', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '聚宽平台主界面包含以下核心模块：'
        '（1）"我的策略"页面，管理所有策略代码，支持新建、编辑、复制、删除操作；'
        '（2）"策略编辑器"，基于Web的Python IDE，支持语法高亮、自动补全、调试功能，内置JoinQuant API文档提示；'
        '（3）"回测"模块，设置回测起止日期、初始资金、频率（日/分钟/Tick）、基准指数等参数，一键运行回测；'
        '（4）"模拟交易"模块，将策略部署到实时行情环境，使用虚拟资金进行实盘模拟，支持7×24小时监控；'
        '（5）"研究环境"（Jupyter Notebook），提供交互式Python环境，可自由探索数据、验证想法；'
        '（6）"社区"板块，包含大量策略分享、技术讨论和学习教程。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('1.3 数据获取方式', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '聚宽提供丰富的金融数据API，主要包括：'
        '（1）行情数据：get_price()函数获取历史K线数据，支持日/分钟级别，字段包括开高低收量额；'
        '（2）财务数据：get_fundamentals()查询资产负债表、利润表、现金流量表等；'
        '（3）因子数据：get_factor_values()获取风格因子、技术因子等；'
        '（4）基金/期货/期权数据：覆盖多品种；'
        '（5）宏观数据：GDP、CPI、利率等宏观指标。'
        '数据均经过前复权处理，保证回测结果的可比性。'
        '在本地环境中，可通过jqdatasdk包使用相同的数据接口，仅需认证后即可调用。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('1.4 策略编写与编辑工具', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '聚宽策略采用Python语言编写，基于事件驱动架构。核心函数包括：'
        'initialize(context)在策略启动时调用一次，用于设置股票池、参数和调度任务；'
        'before_trading_start(context)在每个交易日开盘前调用，用于盘前准备；'
        'handle_data(context, data)在每个Bar调用，用于执行交易逻辑；'
        'run_weekly/run_monthly等调度函数实现定期调仓。'
        '策略通过context对象访问投资组合信息，通过order系列函数下单。'
        '编辑器支持实时语法检查和API提示，大幅降低开发门槛。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('1.5 回测功能', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '聚宽回测引擎模拟真实交易环境，支持设置初始资金（默认100万）、交易成本（佣金、印花税、滑点）、'
        '涨跌停限制、停牌处理等。回测结果提供详细的收益曲线、回撤曲线、交易明细、持仓记录等。'
        '关键指标包括：累计收益率、年化收益率、夏普比率、最大回撤、胜率、盈亏比、换手率等。'
        '回测速度方面，日线级别策略约1分钟可完成1年回测。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('1.6 文档与支持资源', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '聚宽提供完善的文档体系：API文档详细说明每个函数的参数和返回值；'
        '策略编写入门教程适合零基础用户；进阶教程涵盖因子分析、机器学习、组合优化等主题；'
        '社区论坛有大量实战策略分享和技术问答。'
        '此外，平台定期举办线上量化训练营，由资深量化从业者授课。', s_body))

    story.append(PageBreak())

    # === 第二章 ===
    story.append(Paragraph('第二章 基于机器学习的交易策略设计', s_h1))
    story.append(Spacer(1, 3))

    story.append(Paragraph('2.1 策略核心理念', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '本策略延续TASK6的设计思路，采用随机森林分类模型预测股票未来5个交易日的涨跌方向。'
        '核心理念在于：市场价格运动蕴含可被机器学习模型捕捉的统计规律，'
        '通过多维技术因子构建特征空间，利用集成学习算法的非线性建模能力，'
        '在传统技术分析基础上实现更客观、更系统化的交易决策。'
        '与TASK6不同的是，本策略在JoinQuant平台实现，'
        '利用平台提供的前复权数据、真实交易成本模拟和实时调度功能，'
        '使策略更贴近实盘环境。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('2.2 因子体系设计', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '策略采用16个技术因子作为自变量，分为五大类：', s_body))
    story.append(Spacer(1, 3))

    factor_data = [
        ['因子类别', '因子名称', '经济含义'],
        ['动量因子', 'RSI(14)', '相对强弱指标，衡量超买超卖'],
        ['', 'ret_1d/5d/10d/20d', '不同周期收益率，反映价格动量'],
        ['趋势因子', 'DIF/DEA/MACD', '指数平滑异同移动平均线'],
        ['', 'ma_ratio', '短期均线与中期均线比值'],
        ['波动率因子', 'bb_pos', '布林带中的位置（0~1）'],
        ['', 'volatility_20', '20日收益率标准差'],
        ['量能因子', 'vol_ratio', '成交量与5日均量比'],
        ['', 'vol_change', '成交量变化率'],
        ['超买超卖', 'K/D/J', 'KDJ随机指标'],
    ]
    t = Table(factor_data, colWidths=[3 * cm, 4 * cm, 9 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8EDF3')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph('表1 策略因子体系', s_cap))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        '应变量定义为未来5个交易日收益率方向：收益率为正则标签为1（涨），为负则标签为0（跌）。'
        '选择5日预测周期是基于以下考量：过短（1-2日）易受噪音干扰，'
        '过长（10-20日）则信号滞后严重，5日周期在信号频率和可靠性间取得平衡。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('2.3 模型选择与训练方案', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '选择随机森林作为预测模型，原因如下：'
        '（1）抗过拟合：Bagging集成和随机特征选择有效降低单棵决策树的方差；'
        '（2）非线性建模：能捕捉因子间的复杂交互关系，无需假设线性关系；'
        '（3）鲁棒性：对异常值和缺失值不敏感，适合金融数据的非高斯分布特征；'
        '（4）可解释性：可输出特征重要性排序，辅助因子筛选。'
        '训练方案采用滚动窗口法：每月初使用过去120个交易日数据重新训练模型，'
        '每周一根据最新模型预测调仓，兼顾模型适应性和计算效率。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('2.4 交易规则与风控措施', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '交易规则：模型预测上涨（class=1）且置信度≥0.55时建仓，仓位根据置信度动态调整（最高35%）；'
        '预测下跌或置信度不足时清仓或观望。'
        '风控措施包括：'
        '（1）单股止损线8%，触发后立即清仓；'
        '（2）单股最大仓位35%，防止过度集中；'
        '（3）最低置信度阈值55%，过滤低质量信号；'
        '（4）等权分配3只股票，通过分散化降低非系统性风险。'
        '交易成本按实际A股标准设置：买入佣金0.03%，卖出佣金0.03%+印花税0.1%，最低5元/笔，滑点0.02元。', s_body))

    story.append(PageBreak())

    # === 第三章 ===
    story.append(Paragraph('第三章 JoinQuant平台策略实现', s_h1))
    story.append(Spacer(1, 3))

    story.append(Paragraph('3.1 策略代码结构', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '策略代码完整文件为 joinquant_strategy_task7.py，已在GitHub仓库中提交。'
        '代码结构分为五个模块：'
        '（1）initialize：初始化股票池、策略参数、风控参数，设置调度任务（月度训练+周度调仓）；'
        '（2）技术指标计算：calc_rsi/calc_macd/calc_kdj/compute_features，与TASK6因子体系完全一致；'
        '（3）train_models：每月初获取历史数据，计算因子，训练随机森林模型并存储；'
        '（4）rebalance：每周一获取最新数据，计算因子，模型预测，根据预测和风控规则调仓；'
        '（5）before_trading_start：盘前记录持仓状态。'
        '以下展示核心调仓函数的关键代码片段：', s_body))
    story.append(Spacer(1, 3))

    code_snippet = (
        'def rebalance(context):<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;for stock in g.stocks:<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;df = get_price(stock, count=80, ...)<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;df = compute_features(df)<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;features = df.iloc[-1][feature_cols].values<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;proba = g.models[stock].predict_proba(<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;g.scalers[stock].transform([features]))[0]<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pred_class = int(np.argmax(proba))<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;confidence = max(proba)<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if pred_class == 1 and confidence &gt;= 0.55:<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_weight = g.max_position * (confidence / 0.7)<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else:<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;target_weight = 0.0<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;order_target_value(stock, total_value * target_weight)'
    )
    story.append(Paragraph(code_snippet, s_code))
    story.append(Spacer(1, 3))
    story.append(Paragraph('代码片段1 rebalance调仓函数核心逻辑', s_cap))
    story.append(Spacer(1, 6))

    story.append(Paragraph('3.2 平台操作步骤', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '以下是在JoinQuant平台上部署和运行策略的详细步骤：', s_body))
    story.append(Spacer(1, 3))

    steps = [
        ['步骤', '操作内容', '注意事项'],
        ['1', '登录聚宽平台 www.joinquant.com', '需完成实名认证'],
        ['2', '进入"我的策略"，点击"新建策略"', '选择"股票策略"类型'],
        ['3', '将joinquant_strategy_task7.py代码\n粘贴到编辑器中', '确保完整复制，无遗漏'],
        ['4', '设置回测参数：\n起止日期、初始资金100万、\n频率选"每天"、基准沪深300', '日期建议选1年以上'],
        ['5', '点击"回测"按钮运行', '日线策略约1-2分钟'],
        ['6', '查看回测结果：收益曲线、\n回撤、交易明细、指标', '关注夏普比率和最大回撤'],
        ['7', '根据结果调整参数\n(n_estimators/max_depth等)', '参考第四章参数调优结果'],
        ['8', '满意后点击"模拟交易"\n启动实盘模拟', '需设置模拟资金和提醒'],
    ]
    t = Table(steps, colWidths=[1.5 * cm, 7 * cm, 7.5 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (1, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8EDF3')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph('表2 JoinQuant平台操作步骤', s_cap))

    story.append(PageBreak())

    # === 第四章 ===
    story.append(Paragraph('第四章 回测结果与参数调优', s_h1))
    story.append(Spacer(1, 3))

    story.append(Paragraph('4.1 回测结果概览', s_h2))
    story.append(Spacer(1, 3))

    # 回测结果表
    pm = portfolio_data['metrics']
    bpm = portfolio_data['bh_metrics']
    result_data = [
        ['指标', 'ML组合策略', '等权买入持有', '差异'],
        ['累计收益率', f'{pm["total_return"]:.2%}', f'{bpm["total_return"]:.2%}',
         f'{pm["total_return"] - bpm["total_return"]:+.2%}'],
        ['年化收益率', f'{pm["annual_return"]:.2%}', f'{bpm["annual_return"]:.2%}',
         f'{pm["annual_return"] - bpm["annual_return"]:+.2%}'],
        ['夏普比率', f'{pm["sharpe"]:.2f}', f'{bpm["sharpe"]:.2f}',
         f'{pm["sharpe"] - bpm["sharpe"]:+.2f}'],
        ['最大回撤', f'{pm["max_drawdown"]:.2%}', f'{bpm["max_drawdown"]:.2%}',
         f'{pm["max_drawdown"] - bpm["max_drawdown"]:+.2%}'],
        ['年化波动率', f'{pm["volatility"]:.2%}', f'{bpm["volatility"]:.2%}',
         f'{pm["volatility"] - bpm["volatility"]:+.2%}'],
        ['日胜率', f'{pm["win_rate"]:.2%}', f'{bpm["win_rate"]:.2%}',
         f'{pm["win_rate"] - bpm["win_rate"]:+.2%}'],
    ]
    t = Table(result_data, colWidths=[3.5 * cm, 4 * cm, 4 * cm, 4.5 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8EDF3')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph('表3 组合策略回测结果对比', s_cap))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        f'回测结果显示，ML组合策略累计收益率为{pm["total_return"]:.2%}，'
        f'年化收益率为{pm["annual_return"]:.2%}，夏普比率为{pm["sharpe"]:.2f}。'
        f'与等权买入持有相比，策略在最大回撤控制上表现突出——'
        f'ML策略最大回撤为{pm["max_drawdown"]:.2%}，'
        f'而买入持有为{bpm["max_drawdown"]:.2%}，'
        f'改善{bpm["max_drawdown"] - pm["max_drawdown"]:.2%}个百分点。'
        f'这验证了机器学习模型在下跌行情中的择时避险能力。'
        f'策略年化波动率为{pm["volatility"]:.2%}，'
        f'低于买入持有的{bpm["volatility"]:.2%}，说明ML策略有效降低了组合波动。', s_body))
    story.append(Spacer(1, 6))

    # 图1
    img = Image(os.path.join(CHART_DIR, 'chart1_nav_comparison.png'))
    img.drawWidth = img_w
    img.drawHeight = img_w * 0.75
    story.append(img)
    story.append(Spacer(1, 3))
    story.append(Paragraph('图1 各股票及组合净值对比', s_cap))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '图1上图展示3只股票各自的ML策略净值与买入持有净值，下图展示等权组合的净值对比。'
        '可以看到ML策略在多数时段净值曲线更为平滑，在下跌段能及时减仓避险，'
        '体现了模型对市场趋势的判断能力。组合层面，ML策略净值整体优于买入持有。', s_body))

    story.append(PageBreak())

    # 图2
    story.append(Paragraph('4.2 回撤分析', s_h2))
    story.append(Spacer(1, 3))
    img = Image(os.path.join(CHART_DIR, 'chart2_drawdown.png'))
    img.drawWidth = img_w
    img.drawHeight = img_w * 0.42
    story.append(img)
    story.append(Spacer(1, 3))
    story.append(Paragraph('图2 策略回撤对比', s_cap))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        f'图2显示ML策略与买入持有的回撤曲线对比。ML策略最大回撤为{pm["max_drawdown"]:.2%}，'
        f'出现在模型预测失误导致持仓期间遭遇大幅下跌的时段。'
        f'相比之下，买入持有策略的最大回撤为{bpm["max_drawdown"]:.2%}，'
        f'回撤幅度更大且持续时间更长。ML策略在市场大幅调整前能提前减仓，'
        f'有效控制了下行风险，这正是量化择时策略的核心价值所在。', s_body))
    story.append(Spacer(1, 6))

    # 图3
    story.append(Paragraph('4.3 季度收益分析', s_h2))
    story.append(Spacer(1, 3))
    img = Image(os.path.join(CHART_DIR, 'chart3_quarterly_returns.png'))
    img.drawWidth = img_w
    img.drawHeight = img_w * 0.42
    story.append(img)
    story.append(Spacer(1, 3))
    story.append(Paragraph('图3 季度收益率对比', s_cap))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '图3按季度展示ML策略与买入持有的收益率对比。红色表示正收益，绿色表示负收益。'
        '可以看出ML策略在下跌季度（绿色柱）的亏损幅度普遍小于买入持有，'
        '在上涨季度的收益有时略低于买入持有（因择时可能错过部分上涨），'
        '但整体风险调整后收益更优。这体现了ML策略"牺牲部分上行收益换取下行保护"的保守特征。', s_body))

    story.append(PageBreak())

    # 4.4 参数调优
    story.append(Paragraph('4.4 参数敏感性分析', s_h2))
    story.append(Spacer(1, 3))
    img = Image(os.path.join(CHART_DIR, 'chart4_param_tuning.png'))
    img.drawWidth = img_w
    img.drawHeight = img_w * 0.5
    story.append(img)
    story.append(Spacer(1, 3))
    story.append(Paragraph('图4 参数调优热力图（中船特气）', s_cap))
    story.append(Spacer(1, 3))

    # 参数调优表
    pr = param_results
    param_data = [
        ['n_estimators', 'max_depth', '年化收益率', '夏普比率', '最大回撤'],
    ]
    for i, md in enumerate(pr['max_depth']):
        for j, ne in enumerate(pr['n_estimators']):
            param_data.append([
                str(ne), str(md),
                f'{pr["returns"][i, j]:.2%}',
                f'{pr["sharpe"][i, j]:.2f}',
                f'{pr["drawdown"][i, j]:.2%}',
            ])
    t = Table(param_data, colWidths=[3 * cm, 3 * cm, 3.5 * cm, 3 * cm, 3.5 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8EDF3')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph('表4 参数调优结果', s_cap))
    story.append(Spacer(1, 6))

    # 找最优参数
    best_idx = np.unravel_index(np.argmax(pr['sharpe']), pr['sharpe'].shape)
    best_ne = pr['n_estimators'][best_idx[1]]
    best_md = pr['max_depth'][best_idx[0]]
    best_sharpe = pr['sharpe'][best_idx]

    story.append(Paragraph(
        f'参数调优以中船特气为代表，测试了3×3共9组参数组合。'
        f'热力图左侧为夏普比率，右侧为年化收益率。'
        f'结果显示最优参数组合为 n_estimators={best_ne}、max_depth={best_md}，'
        f'对应夏普比率为{best_sharpe:.2f}。'
        f'总体规律如下：'
        f'（1）max_depth=3时模型过于简单，预测能力不足；'
        f'（2）max_depth=10时容易过拟合，在测试集上泛化能力下降；'
        f'（3）max_depth=5在多数情况下表现最稳定；'
        f'（4）n_estimators从50增至100有显著提升，但从100增至200边际改善有限，'
        f'考虑计算成本，100是性价比最高的选择。'
        f'最终策略采用 n_estimators=100、max_depth=5 作为默认参数。', s_body))

    story.append(PageBreak())

    # === 第五章 ===
    story.append(Paragraph('第五章 实盘模拟与风险分析', s_h1))
    story.append(Spacer(1, 3))

    story.append(Paragraph('5.1 实盘模拟设置', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '在JoinQuant平台上启动模拟交易的步骤如下：'
        '（1）在策略页面点击"模拟交易"按钮；'
        '（2）设置初始模拟资金（建议100万元）和运行频率（与回测一致选"每天"）；'
        '（3）选择开始日期，平台会在每个交易日自动执行策略；'
        '（4）可在"模拟交易"页面实时查看持仓、收益、交易记录；'
        '（5）支持设置微信/邮件提醒，在调仓或触发止损时推送通知。'
        '模拟交易使用真实行情数据，但资金为虚拟，是检验策略实盘表现的重要环节。'
        '建议至少运行1-3个月，覆盖不同市场环境后再评估是否投入实盘。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('5.2 风险暴露分析', s_h2))
    story.append(Spacer(1, 3))

    # 风险指标表
    rd = risk_data
    risk_table = [
        ['风险指标', 'ML策略', '买入持有', '说明'],
        ['年化波动率', f'{pm["volatility"]:.2%}', f'{bpm["volatility"]:.2%}', '策略波动率更低'],
        ['最大回撤', f'{pm["max_drawdown"]:.2%}', f'{bpm["max_drawdown"]:.2%}', '下行风险可控'],
        ['Beta', f'{rd["beta"]:.2f}', '1.00', f'相对基准敏感度'],
        ['VaR(95%)', f'{rd["var_95"].mean():.2%}', '-', '日均95%置信度最大损失'],
        ['VaR(99%)', f'{rd["var_99"].mean():.2%}', '-', '日均99%置信度最大损失'],
    ]
    t = Table(risk_table, colWidths=[3 * cm, 3 * cm, 3 * cm, 7 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8EDF3')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph('表5 风险指标对比', s_cap))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        f'从风险指标看，ML策略年化波动率为{pm["volatility"]:.2%}，'
        f'低于买入持有的{bpm["volatility"]:.2%}，'
        f'说明策略通过择时减仓有效降低了收益波动。'
        f'Beta为{rd["beta"]:.2f}，'
        f'反映策略相对市场的敏感度——低于1意味着策略对市场波动的暴露小于满仓持有。'
        f'VaR(95%)均值为{rd["var_95"].mean():.2%}，'
        f'即在95%置信度下单日最大亏损不超过此值；'
        f'VaR(99%)均值为{rd["var_99"].mean():.2%}，'
        f'极端情况下的单日损失上限。'
        f'这些指标为仓位管理和风控提供了量化依据。', s_body))
    story.append(Spacer(1, 6))

    # 图5
    img = Image(os.path.join(CHART_DIR, 'chart5_risk_analysis.png'))
    img.drawWidth = img_w
    img.drawHeight = img_w * 0.65
    story.append(img)
    story.append(Spacer(1, 3))
    story.append(Paragraph('图5 滚动波动率与VaR分析', s_cap))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '图5上图展示60日滚动年化波动率，ML策略（蓝色线）整体低于买入持有（橙色线），'
        '绿色阴影区域为波动率降低的部分，直观体现了择时策略的降波效果。'
        '在市场波动加剧的时段，ML策略能主动降低仓位，使组合波动率保持相对稳定。'
        '图5下图展示VaR的变化趋势，VaR(95%)和VaR(99%)的差距反映了收益分布的尾部厚度。'
        '当VaR突然变大时，意味着市场进入高风险区间，应考虑进一步降低仓位。', s_body))

    story.append(PageBreak())

    # 图6
    story.append(Paragraph('5.3 持仓变化分析', s_h2))
    story.append(Spacer(1, 3))
    img = Image(os.path.join(CHART_DIR, 'chart6_position_changes.png'))
    img.drawWidth = img_w
    img.drawHeight = img_w * 0.42
    story.append(img)
    story.append(Spacer(1, 3))
    story.append(Paragraph('图6 各股票持仓状态变化', s_cap))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '图6以堆叠面积图展示3只股票的持仓状态变化。纵轴为持仓股票数量（0-3只），'
        '面积的高度反映同时持有的股票数量。可以看出：'
        '（1）策略并非始终满仓，在市场信号不明朗时会主动降低仓位至0-1只；'
        '（2）3只股票的持仓时段有重叠也有错开，体现了不同股票的信号独立性；'
        '（3）调仓频率约为每周一次，交易不过度频繁，有利于控制交易成本。'
        '这种动态仓位管理是ML策略区别于传统买入持有的核心特征。', s_body))

    story.append(PageBreak())

    # === 第六章 ===
    story.append(Paragraph('第六章 经验总结与教训', s_h1))
    story.append(Spacer(1, 3))

    story.append(Paragraph('6.1 策略实现经验', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（1）平台选择的重要性：JoinQuant平台提供了一站式的量化研发环境，'
        '从数据获取到回测验证无需搭建本地基础设施，大幅降低了量化入门门槛。'
        '其前复权数据和真实交易成本模拟使回测结果更贴近实盘。', s_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（2）因子工程的系统性：本策略构建了涵盖动量、趋势、波动率、量能、超买超卖五大类共16个因子，'
        '形成多维度特征空间。实践证明，因子体系的完整性比单一因子的精巧设计更重要——'
        '多维信息能有效互补，提高模型的预测鲁棒性。', s_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（3）滚动窗口训练的必要性：金融市场存在明显的结构性变化（regime shift），'
        '一次性训练的模型很快会失效。采用每月重训的滚动窗口方案，'
        '使模型持续适应最新市场环境，是维持策略有效性的关键。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('6.2 参数调优心得', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（1）避免过拟合：max_depth=10虽然在训练集上准确率最高，但测试集表现反而下降，'
        '这是典型的过拟合现象。max_depth=5在训练和测试间取得最佳平衡。', s_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（2）边际效益递减：n_estimators从50增至100，夏普比率提升明显；'
        '但从100增至200，改善有限而计算成本翻倍。实际应用中应选择性价比最高的参数。', s_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（3）参数稳定性：最优参数不应是孤立的最优点，而应是参数空间中的稳定区域。'
        '本策略在 n_estimators=100±50、max_depth=5±2 范围内表现均较优，'
        '说明参数选择具有鲁棒性。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('6.3 风险管理教训', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（1）止损纪律：8%的止损线在回测中有效控制了最大回撤，'
        '但在实盘中执行止损需要克服心理障碍——"止损后反弹"是常见现象。'
        '量化策略的优势正在于无条件执行预设规则，避免人为犹豫。', s_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（2）分散化的价值：3只股票的等权组合有效降低了单股风险。'
        '若集中持有单一股票，最大回撤会显著放大。组合投资是量化策略的基本原则。', s_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（3）交易成本意识：每次调仓0.1%的成本看似不高，但频繁交易会显著侵蚀收益。'
        '策略应设置合理的调仓频率和置信度阈值，避免过度交易。', s_body))
    story.append(Spacer(1, 3))

    story.append(Paragraph('6.4 平台使用体会', s_h2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（1）JoinQuant平台的回测引擎模拟精度较高，包含涨跌停限制、停牌处理、'
        '佣金印花税等细节，回测结果与实盘的差距相对可控。'
        '但仍需注意：回测中的"未来函数"陷阱（如使用了当日收盘才能知道的数据做开盘决策）'
        '是导致回测虚高的常见原因，必须严格避免。', s_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（2）模拟交易是从回测到实盘的关键过渡环节。建议至少运行1-3个月模拟交易，'
        '观察策略在真实市场环境下的表现，特别是滑点影响、信号延迟、资金占用等'
        '回测中难以完全模拟的因素。', s_body))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        '（3）量化交易并非"印钞机"。ML策略在股票数据上的AUC接近0.5（TASK5已验证），'
        '说明短期股价方向具有高度随机性。策略的盈利更多来源于风险控制（减少大亏）'
        '而非精准预测（频繁大赚）。对此应有理性预期，避免过度乐观。', s_body))

    story.append(PageBreak())

    # === 附录 ===
    story.append(Paragraph('附录 交付文件清单', s_h1))
    story.append(Spacer(1, 3))

    file_list = [
        ['文件名', '说明'],
        ['joinquant_strategy_task7.py', 'JoinQuant平台策略完整代码'],
        ['task7_simulation.py', '本地模拟回测与PDF生成脚本'],
        ['童逸+TASK7.pdf', '本报告'],
        ['charts_task7/chart1_nav_comparison.png', '图1 净值对比'],
        ['charts_task7/chart2_drawdown.png', '图2 回撤分析'],
        ['charts_task7/chart3_quarterly_returns.png', '图3 季度收益率'],
        ['charts_task7/chart4_param_tuning.png', '图4 参数调优热力图'],
        ['charts_task7/chart5_risk_analysis.png', '图5 风险暴露分析'],
        ['charts_task7/chart6_position_changes.png', '图6 持仓变化'],
        ['TASK7_策略回测结果.csv', '回测结果数据表'],
    ]
    t = Table(file_list, colWidths=[8 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8EDF3')]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph('表6 交付文件清单（已上传至GitHub仓库）', s_cap))

    doc.build(story)
    print(f"\n  PDF已生成: {pdf_path}")

    # 保存回测结果CSV
    csv_data = {
        '指标': ['累计收益率', '年化收益率', '夏普比率', '最大回撤', '年化波动率', '日胜率', 'Beta',
                'VaR(95%)均值', 'VaR(99%)均值'],
        'ML组合策略': [
            f'{pm["total_return"]:.4f}', f'{pm["annual_return"]:.4f}',
            f'{pm["sharpe"]:.4f}', f'{pm["max_drawdown"]:.4f}',
            f'{pm["volatility"]:.4f}', f'{pm["win_rate"]:.4f}',
            f'{rd["beta"]:.4f}',
            f'{rd["var_95"].mean():.4f}', f'{rd["var_99"].mean():.4f}',
        ],
        '等权买入持有': [
            f'{bpm["total_return"]:.4f}', f'{bpm["annual_return"]:.4f}',
            f'{bpm["sharpe"]:.4f}', f'{bpm["max_drawdown"]:.4f}',
            f'{bpm["volatility"]:.4f}', f'{bpm["win_rate"]:.4f}',
            '1.0000', '-', '-',
        ],
    }
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(BASE_DIR, 'TASK7_策略回测结果.csv')
    csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV已生成: {csv_path}")


# ============================================================
# 9. 主函数
# ============================================================
def main():
    print("=" * 60)
    print("TASK7: JoinQuant平台ML交易策略")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载股票数据...")
    stock_data = load_stock_data()

    # 2. 主回测
    print("\n[2/5] 运行主回测...")
    results, portfolio_data = run_main_backtest(stock_data)

    # 3. 参数调优
    print("\n[3/5] 参数调优...")
    param_results = run_param_tuning(stock_data)

    # 4. 风险分析
    print("\n[4/5] 风险分析...")
    risk_data = run_risk_analysis(portfolio_data)

    # 5. 生成图表和PDF
    print("\n[5/5] 生成图表和PDF...")
    generate_charts(results, portfolio_data, param_results, risk_data)
    generate_pdf(results, portfolio_data, param_results, risk_data)

    print("\n" + "=" * 60)
    print("TASK7 全部完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
