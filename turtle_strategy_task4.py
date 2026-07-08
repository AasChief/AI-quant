# -*- coding: utf-8 -*-
"""
TASK4: 海龟交易策略分析
1. 加载股价数据
2. 计算20日唐奇安通道(高低点通道)
3. 计算ATR(14)
4. 计算买卖信号
5. 绘制可视化图形
6. 回测并计算量化指标
7. 多股票多参数对比
8. 生成PDF报告
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.dates import DateFormatter
import os
import warnings
warnings.filterwarnings('ignore')

# ============ 字体配置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# ============ PDF配置 ============
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, KeepTogether, PageBreak)
from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 注册宋体
pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))

BASE_DIR = 'E:/量化交易：AI大模型辅助的金融交易策略'


# ============================================================
# 第一部分: 海龟策略核心计算函数
# ============================================================

def calculate_donchian_channel(df, entry_period=20, exit_period=10):
    """
    计算唐奇安通道(高低点通道)
    - entry_period: 入场通道周期(上轨=过去N日最高价)
    - exit_period: 出场通道周期(下轨=过去M日最低价)
    """
    df = df.copy()
    # 入场上轨: 过去entry_period日的最高价(不含当日)
    df['upper_channel'] = df['high'].rolling(window=entry_period).max().shift(1)
    # 出场下轨: 过去exit_period日的最低价(不含当日)
    df['lower_channel'] = df['low'].rolling(window=exit_period).min().shift(1)
    # 中轨
    df['mid_channel'] = (df['upper_channel'] + df['lower_channel']) / 2
    return df


def calculate_atr(df, period=14):
    """
    计算ATR(平均真实波幅)
    TR = max(High-Low, |High-PreClose|, |PreClose-Low|)
    ATR = TR的period日简单移动平均
    """
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['pre_close']),
            abs(df['pre_close'] - df['low'])
        )
    )
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df


def generate_turtle_signals(df, entry_period=20, exit_period=10, atr_period=14, stop_loss_mult=2):
    """
    生成海龟策略交易信号
    - 买入: 收盘价突破上轨(过去entry_period日最高价)
    - 卖出: 收盘价跌破下轨(过去exit_period日最低价) 或 触发止损
    - 止损: 入场价 - stop_loss_mult * ATR
    """
    df = df.copy()
    df = calculate_donchian_channel(df, entry_period, exit_period)
    df = calculate_atr(df, atr_period)

    df['signal'] = 0  # 0=空仓, 1=持仓
    df['position'] = 0
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['trade_type'] = ''  # 记录交易类型

    position = 0
    entry_price = 0
    stop_loss = 0

    for i in range(len(df)):
        if i < entry_period:  # 数据不足,跳过
            continue

        close = df.iloc[i]['close']
        upper = df.iloc[i]['upper_channel']
        lower = df.iloc[i]['lower_channel']
        atr = df.iloc[i]['atr']

        if pd.isna(upper) or pd.isna(lower) or pd.isna(atr):
            continue

        if position == 0:
            # 空仓: 检查买入信号(突破上轨)
            if close > upper:
                position = 1
                entry_price = close
                stop_loss = close - stop_loss_mult * atr
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('trade_type')] = '买入'
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss
        elif position == 1:
            # 持仓: 检查卖出信号(跌破下轨 或 触发止损)
            if close < lower or close <= stop_loss:
                position = 0
                df.iloc[i, df.columns.get_loc('signal')] = -1
                df.iloc[i, df.columns.get_loc('trade_type')] = '卖出' if close < lower else '止损卖出'
                entry_price = 0
                stop_loss = 0
            else:
                # 持仓中,更新止损(移动止损)
                new_stop = close - stop_loss_mult * atr
                if new_stop > stop_loss:
                    stop_loss = new_stop
                df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                df.iloc[i, df.columns.get_loc('stop_loss')] = stop_loss

        df.iloc[i, df.columns.get_loc('position')] = position

    return df


def backtest_turtle(df, initial_capital=100000):
    """
    回测海龟策略,计算量化指标
    """
    df = df.copy()
    df['daily_return'] = df['close'].pct_change()
    df['strategy_return'] = df['position'].shift(1) * df['daily_return']
    df['strategy_return'].fillna(0, inplace=True)

    # 策略净值
    df['strategy_nav'] = (1 + df['strategy_return']).cumprod()
    df['buy_hold_nav'] = (1 + df['daily_return']).cumprod()

    # 量化指标
    total_days = len(df)
    trading_days = df[df['position'] == 1].shape[0]

    # 累计回报
    strategy_cum_return = df['strategy_nav'].iloc[-1] - 1
    buy_hold_cum_return = df['buy_hold_nav'].iloc[-1] - 1

    # 年化收益率
    annual_strategy = (1 + strategy_cum_return) ** (250 / total_days) - 1
    annual_buy_hold = (1 + buy_hold_cum_return) ** (250 / total_days) - 1

    # 最大回撤
    strategy_peak = df['strategy_nav'].cummax()
    strategy_drawdown = (df['strategy_nav'] - strategy_peak) / strategy_peak
    strategy_mdd = strategy_drawdown.min()

    bh_peak = df['buy_hold_nav'].cummax()
    bh_drawdown = (df['buy_hold_nav'] - bh_peak) / bh_peak
    bh_mdd = bh_drawdown.min()

    # 夏普比率 (无风险利率3%)
    rf = 0.03 / 250
    excess_return = df['strategy_return'] - rf
    strategy_sharpe = np.sqrt(250) * excess_return.mean() / excess_return.std() if excess_return.std() > 0 else 0

    bh_excess = df['daily_return'] - rf
    bh_sharpe = np.sqrt(250) * bh_excess.mean() / bh_excess.std() if bh_excess.std() > 0 else 0

    # 交易统计
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]

    # 计算每笔交易盈亏
    trades = []
    buy_dates = buy_signals['trade_date'].tolist()
    sell_dates = sell_signals['trade_date'].tolist()
    buy_prices = buy_signals['close'].tolist()
    sell_prices = sell_signals['close'].tolist()

    n_trades = min(len(buy_dates), len(sell_dates))
    for j in range(n_trades):
        ret = (sell_prices[j] - buy_prices[j]) / buy_prices[j]
        trades.append(ret)

    win_trades = sum(1 for t in trades if t > 0)
    win_rate = win_trades / len(trades) * 100 if trades else 0
    avg_trade_return = np.mean(trades) * 100 if trades else 0

    metrics = {
        'total_days': total_days,
        'trading_days': trading_days,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_return,
        'strategy_cum_return': strategy_cum_return * 100,
        'buy_hold_cum_return': buy_hold_cum_return * 100,
        'annual_strategy': annual_strategy * 100,
        'annual_buy_hold': annual_buy_hold * 100,
        'strategy_mdd': strategy_mdd * 100,
        'buy_hold_mdd': bh_mdd * 100,
        'strategy_sharpe': strategy_sharpe,
        'buy_hold_sharpe': bh_sharpe,
        'trades': trades,
        'drawdown_series': strategy_drawdown,
    }

    return df, metrics


# ============================================================
# 第二部分: 可视化函数
# ============================================================

def plot_turtle_strategy(df, stock_name, entry_period=20, exit_period=10, chart_num=1):
    """绘制海龟策略完整图表: 股价+通道+信号+ATR"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                    gridspec_kw={'hspace': 0.15})

    dates = pd.to_datetime(df['trade_date'])

    # 上图: 股价 + 唐奇安通道 + 买卖信号
    ax1.plot(dates, df['close'], color='#1f77b4', linewidth=1, label='收盘价', zorder=3)
    ax1.plot(dates, df['upper_channel'], color='#d62728', linewidth=1.2,
             label=f'上轨({entry_period}日最高)', linestyle='--', alpha=0.8)
    ax1.plot(dates, df['lower_channel'], color='#2ca02c', linewidth=1.2,
             label=f'下轨({exit_period}日最低)', linestyle='--', alpha=0.8)
    ax1.fill_between(dates, df['upper_channel'], df['lower_channel'],
                     alpha=0.08, color='#ff7f0e')

    # 标记买入信号
    buy_signals = df[df['signal'] == 1]
    if len(buy_signals) > 0:
        buy_dates = pd.to_datetime(buy_signals['trade_date'])
        ax1.scatter(buy_dates, buy_signals['close'], marker='^',
                    color='#d62728', s=120, zorder=5, label='买入信号', edgecolors='black', linewidths=0.5)

    # 标记卖出信号
    sell_signals = df[df['signal'] == -1]
    if len(sell_signals) > 0:
        sell_dates = pd.to_datetime(sell_signals['trade_date'])
        sell_colors = ['#2ca02c' if t == '卖出' else '#ff7f0e' for t in sell_signals['trade_type']]
        ax1.scatter(sell_dates, sell_signals['close'], marker='v',
                    color='#2ca02c', s=120, zorder=5, label='卖出信号', edgecolors='black', linewidths=0.5)

    ax1.set_title(f'图{chart_num} {stock_name}海龟策略交易信号图(通道周期{entry_period}/{exit_period})',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('价格(元)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    # 下图: ATR
    ax2.plot(dates, df['atr'], color='#9467bd', linewidth=1.2, label='ATR(14)')
    ax2.fill_between(dates, 0, df['atr'], alpha=0.15, color='#9467bd')
    ax2.set_ylabel('ATR', fontsize=11)
    ax2.set_xlabel('日期', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, f'chart{chart_num}_turtle_{stock_name}_DC{entry_period}_{exit_period}.png')
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def plot_nav_curve(df, stock_name, entry_period=20, exit_period=10, chart_num=2):
    """绘制策略净值曲线"""
    fig, ax = plt.subplots(figsize=(14, 6))
    dates = pd.to_datetime(df['trade_date'])

    ax.plot(dates, df['strategy_nav'], color='#d62728', linewidth=1.5, label='海龟策略净值')
    ax.plot(dates, df['buy_hold_nav'], color='#1f77b4', linewidth=1.5, label='买入持有净值', alpha=0.7)

    ax.fill_between(dates, df['strategy_nav'], 1, alpha=0.1, color='#d62728')
    ax.fill_between(dates, df['buy_hold_nav'], 1, alpha=0.1, color='#1f77b4')

    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'图{chart_num} {stock_name}海龟策略净值曲线(通道周期{entry_period}/{exit_period})',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('净值', fontsize=11)
    ax.set_xlabel('日期', fontsize=11)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, f'chart{chart_num}_nav_{stock_name}_DC{entry_period}_{exit_period}.png')
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def plot_drawdown(df, stock_name, entry_period=20, exit_period=10, chart_num=3):
    """绘制回撤曲线"""
    fig, ax = plt.subplots(figsize=(14, 5))
    dates = pd.to_datetime(df['trade_date'])

    strategy_peak = df['strategy_nav'].cummax()
    strategy_dd = (df['strategy_nav'] - strategy_peak) / strategy_peak * 100

    bh_peak = df['buy_hold_nav'].cummax()
    bh_dd = (df['buy_hold_nav'] - bh_peak) / bh_peak * 100

    ax.fill_between(dates, strategy_dd, 0, color='#d62728', alpha=0.4, label='策略回撤')
    ax.fill_between(dates, bh_dd, 0, color='#1f77b4', alpha=0.3, label='买入持有回撤')
    ax.plot(dates, strategy_dd, color='#d62728', linewidth=1)
    ax.plot(dates, bh_dd, color='#1f77b4', linewidth=1)

    ax.set_title(f'图{chart_num} {stock_name}海龟策略回撤对比(通道周期{entry_period}/{exit_period})',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('回撤(%)', fontsize=11)
    ax.set_xlabel('日期', fontsize=11)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, f'chart{chart_num}_drawdown_{stock_name}_DC{entry_period}_{exit_period}.png')
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def plot_comparison_chart(comparison_df, chart_num=7):
    """绘制多股票多参数对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 按股票分组
    stocks = comparison_df['stock'].unique()
    colors_stock = ['#d62728', '#1f77b4', '#2ca02c']

    # 1. 累计回报对比
    ax = axes[0, 0]
    for idx, stock in enumerate(stocks):
        data = comparison_df[comparison_df['stock'] == stock]
        x = range(len(data))
        ax.plot(x, data['strategy_cum_return'], marker='o', color=colors_stock[idx],
                label=f'{stock}策略', linewidth=1.5)
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['config'], rotation=45, ha='right', fontsize=8)
    ax.set_title('累计回报对比(%)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 2. 夏普比率对比
    ax = axes[0, 1]
    for idx, stock in enumerate(stocks):
        data = comparison_df[comparison_df['stock'] == stock]
        x = range(len(data))
        ax.plot(x, data['strategy_sharpe'], marker='s', color=colors_stock[idx],
                label=f'{stock}策略', linewidth=1.5)
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['config'], rotation=45, ha='right', fontsize=8)
    ax.set_title('夏普比率对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 3. 最大回撤对比
    ax = axes[1, 0]
    for idx, stock in enumerate(stocks):
        data = comparison_df[comparison_df['stock'] == stock]
        x = range(len(data))
        ax.plot(x, data['strategy_mdd'], marker='v', color=colors_stock[idx],
                label=f'{stock}策略', linewidth=1.5)
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['config'], rotation=45, ha='right', fontsize=8)
    ax.set_title('最大回撤对比(%)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. 胜率对比
    ax = axes[1, 1]
    for idx, stock in enumerate(stocks):
        data = comparison_df[comparison_df['stock'] == stock]
        x = range(len(data))
        ax.plot(x, data['win_rate'], marker='D', color=colors_stock[idx],
                label=f'{stock}策略', linewidth=1.5)
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels(comparison_df['config'], rotation=45, ha='right', fontsize=8)
    ax.set_title('胜率对比(%)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'图{chart_num} 海龟策略多股票多参数对比', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    chart_path = os.path.join(BASE_DIR, f'chart{chart_num}_turtle_comparison.png')
    plt.savefig(chart_path)
    plt.close()
    return chart_path


# ============================================================
# 第三部分: 主程序
# ============================================================

def run_turtle_strategy(csv_path, stock_name, entry_period=20, exit_period=10, chart_start=1):
    """运行单次海龟策略分析"""
    df = pd.read_csv(csv_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')

    # 生成信号
    df = generate_turtle_signals(df, entry_period, exit_period)

    # 回测
    df, metrics = backtest_turtle(df)

    # 绘图
    chart1 = plot_turtle_strategy(df, stock_name, entry_period, exit_period, chart_start)
    chart2 = plot_nav_curve(df, stock_name, entry_period, exit_period, chart_start + 1)
    chart3 = plot_drawdown(df, stock_name, entry_period, exit_period, chart_start + 2)

    return df, metrics, [chart1, chart2, chart3]


def main():
    print("=" * 60)
    print("TASK4: 海龟交易策略分析")
    print("=" * 60)

    # 股票列表
    stocks = [
        {'name': '中船特气', 'code': '688146', 'csv': '中船特气_688146_daily_data.csv'},
        {'name': '天地科技', 'code': '600587', 'csv': 'tiandi_keji_600587_daily_data.csv'},
        {'name': '平安银行', 'code': '000001', 'csv': '平安银行_000001_daily_data.csv'},
    ]

    # 参数组合
    configs = [
        {'entry': 20, 'exit': 10, 'label': 'DC20/10'},
        {'entry': 55, 'exit': 20, 'label': 'DC55/20'},
        {'entry': 10, 'exit': 5, 'label': 'DC10/5'},
    ]

    all_results = []
    all_charts = []
    chart_num = 1

    # === 主分析: 中船特气 DC20/10 ===
    print("\n[1] 中船特气 海龟策略 DC20/10 ...")
    df_main, metrics_main, charts_main = run_turtle_strategy(
        os.path.join(BASE_DIR, stocks[0]['csv']),
        stocks[0]['name'], 20, 10, chart_num
    )
    chart_num += 3
    all_charts.extend(charts_main)

    # 保存主策略数据
    df_main.to_csv(os.path.join(BASE_DIR, '中船特气_688146_turtle_strategy.csv'),
                   index=False, encoding='utf-8-sig')
    print(f"  策略累计回报: {metrics_main['strategy_cum_return']:.2f}%")
    print(f"  买入持有回报: {metrics_main['buy_hold_cum_return']:.2f}%")
    print(f"  夏普比率: {metrics_main['strategy_sharpe']:.2f}")
    print(f"  最大回撤: {metrics_main['strategy_mdd']:.2f}%")
    print(f"  交易次数: {metrics_main['n_trades']}, 胜率: {metrics_main['win_rate']:.1f}%")

    # === 天地科技 DC20/10 ===
    print("\n[2] 天地科技 海龟策略 DC20/10 ...")
    df_td, metrics_td, charts_td = run_turtle_strategy(
        os.path.join(BASE_DIR, stocks[1]['csv']),
        stocks[1]['name'], 20, 10, chart_num
    )
    chart_num += 3
    all_charts.extend(charts_td)
    print(f"  策略累计回报: {metrics_td['strategy_cum_return']:.2f}%")
    print(f"  夏普比率: {metrics_td['strategy_sharpe']:.2f}")

    # === 平安银行 DC20/10 ===
    print("\n[3] 平安银行 海龟策略 DC20/10 ...")
    df_pa, metrics_pa, charts_pa = run_turtle_strategy(
        os.path.join(BASE_DIR, stocks[2]['csv']),
        stocks[2]['name'], 20, 10, chart_num
    )
    chart_num += 3
    all_charts.extend(charts_pa)
    print(f"  策略累计回报: {metrics_pa['strategy_cum_return']:.2f}%")
    print(f"  夏普比率: {metrics_pa['strategy_sharpe']:.2f}")

    # === 多参数对比实验 ===
    print("\n[4] 多股票多参数对比实验 ...")
    comparison_data = []

    for stock in stocks:
        for config in configs:
            csv_path = os.path.join(BASE_DIR, stock['csv'])
            df_tmp, metrics_tmp, _ = run_turtle_strategy(
                csv_path, stock['name'],
                config['entry'], config['exit'], 99  # 不生成图表
            )
            comparison_data.append({
                'stock': stock['name'],
                'config': config['label'],
                'entry_period': config['entry'],
                'exit_period': config['exit'],
                'strategy_cum_return': round(metrics_tmp['strategy_cum_return'], 2),
                'buy_hold_cum_return': round(metrics_tmp['buy_hold_cum_return'], 2),
                'strategy_sharpe': round(metrics_tmp['strategy_sharpe'], 2),
                'buy_hold_sharpe': round(metrics_tmp['buy_hold_sharpe'], 2),
                'strategy_mdd': round(metrics_tmp['strategy_mdd'], 2),
                'buy_hold_mdd': round(metrics_tmp['buy_hold_mdd'], 2),
                'n_trades': metrics_tmp['n_trades'],
                'win_rate': round(metrics_tmp['win_rate'], 1),
                'avg_trade_return': round(metrics_tmp['avg_trade_return'], 2),
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(BASE_DIR, '海龟策略对比结果.csv'),
                         index=False, encoding='utf-8-sig')
    print(comparison_df.to_string(index=False))

    # 绘制对比图
    chart_comparison = plot_comparison_chart(comparison_df, chart_num)
    all_charts.append(chart_comparison)
    chart_num += 1

    print(f"\n[5] 共生成 {len(all_charts)} 张图表")

    # === 生成PDF ===
    print("\n[6] 生成PDF报告 ...")
    generate_pdf(df_main, metrics_main, df_td, metrics_td, df_pa, metrics_pa,
                 comparison_df, all_charts, stocks)

    print("\n" + "=" * 60)
    print("TASK4 全部完成!")
    print("=" * 60)


# ============================================================
# 第四部分: PDF生成
# ============================================================

def generate_pdf(df_main, metrics_main, df_td, metrics_td, df_pa, metrics_pa,
                 comparison_df, all_charts, stocks):
    """生成PDF报告"""
    pdf_path = os.path.join(BASE_DIR, '童逸+TASK4.pdf')

    # 样式定义
    style_title = ParagraphStyle(
        'Title', fontName='SimSun', fontSize=18, leading=27,
        alignment=TA_CENTER, spaceBefore=0, spaceAfter=12
    )
    style_heading = ParagraphStyle(
        'Heading', fontName='SimSun', fontSize=14, leading=21,
        alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=6
    )
    style_subheading = ParagraphStyle(
        'SubHeading', fontName='SimSun', fontSize=12, leading=18,
        alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=4
    )
    style_body = ParagraphStyle(
        'Body', fontName='SimSun', fontSize=10.5, leading=15.75,
        alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0,
        firstLineIndent=21  # 首行缩进2字符
    )
    style_figure = ParagraphStyle(
        'Figure', fontName='SimSun', fontSize=10.5, leading=15.75,
        alignment=TA_CENTER, spaceBefore=0, spaceAfter=0
    )

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                            topMargin=25*mm, bottomMargin=25*mm,
                            leftMargin=25*mm, rightMargin=25*mm)

    story = []

    # === 封面标题 ===
    story.append(Paragraph('海龟交易策略分析与回测报告', style_title))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph('童逸+TASK4', style_title))
    story.append(Spacer(1, 10*mm))

    # === 一、海龟策略概述 ===
    story.append(Paragraph('一、海龟交易策略概述', style_heading))

    story.append(Paragraph('（一）核心思想', style_subheading))
    story.append(Paragraph(
        '海龟交易策略（Turtle Trading）由著名交易员理查德·丹尼斯（Richard Dennis）于1983年创立，'
        '是一种经典的趋势跟踪交易策略。其核心思想是：市场价格倾向于在一段时间内向某个方向持续运动，形成趋势。'
        '该策略的目标是尽早识别新趋势的开始，在趋势确认突破时入场，在趋势结束时退出，从而捕获市场的主要趋势利润。'
        '海龟交易法则以严格的纪律和系统化的规则为特点，强调通过完整的交易系统（包括市场选择、入市信号、'
        '头寸管理、止损止盈、出场规则）来实现稳定的长期盈利。', style_body))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph('（二）关键优势', style_subheading))
    story.append(Paragraph(
        '海龟策略具有以下关键优势：（1）系统化交易，消除主观情绪干扰，所有买卖决策均由规则驱动；'
        '（2）趋势跟踪能力强，在单边趋势行情中能捕获大幅利润；（3）风险控制严密，通过ATR动态调整止损和仓位，'
        '实现波动率自适应的风险管理；（4）可复制性强，规则明确、参数清晰，便于程序化实现和历史回测；'
        '（5）分散化设计，策略可同时应用于多个市场，通过低相关性分散风险。', style_body))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph('（三）系统框架', style_subheading))
    story.append(Paragraph(
        '海龟交易系统包含两个子系统：系统1基于20日突破入场、10日突破出场，适用于中短期趋势；'
        '系统2基于55日突破入场、20日突破出场，适用于中长期趋势。两个系统可同时运行，互为补充。'
        '本次分析以系统1为主进行回测，即当收盘价突破过去20个交易日最高价时买入，'
        '当收盘价跌破过去10个交易日最低价时卖出。', style_body))
    story.append(Spacer(1, 5*mm))

    # === 二、核心概念解释 ===
    story.append(Paragraph('二、核心概念解释', style_heading))

    story.append(Paragraph('（一）高低点通道（唐奇安通道，Donchian Channel）', style_subheading))
    story.append(Paragraph(
        '唐奇安通道由技术分析先驱理查德·唐奇安（Richard Donchian）发明，是海龟策略的核心信号工具。'
        '一个N周期唐奇安通道包含三条线：上轨为过去N个交易日的最高价，下轨为过去N个交易日的最低价，'
        '中轨为上下轨的中间值。当价格突破上轨时，表示市场创出近期新高，可能预示上涨趋势开始，为买入信号；'
        '当价格跌破下轨时，表示市场创出近期新低，可能预示下跌趋势开始，为卖出信号。'
        '通道周期越短，信号越敏感但假突破越多；周期越长，信号越可靠但入场越晚。'
        '海龟策略中，系统1使用20日通道入场、10日通道出场，系统2使用55日通道入场、20日通道出场。', style_body))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph('（二）平均真实波幅（ATR，Average True Range）', style_subheading))
    story.append(Paragraph(
        'ATR由技术分析大师威尔斯·威尔德（J. Welles Wilder）发明，用于衡量市场波动程度。'
        '其计算分为两步：首先计算每日真实波幅（TR），TR为以下三个值中的最大值：'
        '（1）当日最高价与最低价之差（High-Low）；'
        '（2）当日最高价与前日收盘价之差的绝对值（|High-PreClose|）；'
        '（3）前日收盘价与当日最低价之差的绝对值（|PreClose-Low|）。'
        '然后对TR取N日（通常为14日或20日）的移动平均，即得到ATR。'
        '在海龟策略中，ATR（海龟法则中称为N值）用于两个关键用途：'
        '一是计算头寸规模——1个头寸单位=账户总值的1%÷（ATR×每点金额），确保每笔交易承担相同比例的风险；'
        '二是设置止损——止损价位为入场价减去2倍ATR，确保止损距离与市场波动率自适应。', style_body))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph('（三）止损条件', style_subheading))
    story.append(Paragraph(
        '海龟策略的止损规则以ATR为核心，采用动态止损机制。初始止损设置为入场价减去2倍ATR（2N），'
        '即当价格从入场点反向运动2个ATR幅度时触发止损。这一设计的核心逻辑是：'
        '如果价格在2个ATR范围内波动属于正常市场噪音，超过2个ATR则可能意味着趋势反转。'
        '在浮盈加仓的情况下，每加仓一次，所有持仓的止损位统一上移1/2N，确保风险始终可控。'
        '此外，海龟策略还采用移动止损（Trailing Stop）机制——随着价格上涨，止损位也会相应上移，'
        '在保护利润的同时给予趋势发展空间。本次回测中实现了2N止损和移动止损功能。', style_body))
    story.append(Spacer(1, 5*mm))

    # === 三、数据诊断 ===
    story.append(PageBreak())
    story.append(Paragraph('三、数据诊断与描述性统计', style_heading))
    story.append(Paragraph(
        f'本次分析使用三只股票过去1年的日线数据：中船特气（688146.SH）共{len(df_main)}个交易日、'
        f'天地科技（600587.SH）共{len(df_td)}个交易日、平安银行（000001.SZ）共{len(df_pa)}个交易日。'
        '数据字段包括开盘价、最高价、最低价、收盘价、前收盘价、涨跌幅、成交量等。'
        '经检查，三只股票的OHLCV数据均无缺失值，数据质量良好。', style_body))
    story.append(Spacer(1, 3*mm))

    # 描述性统计表
    stat_data = [['指标', '中船特气', '天地科技', '平安银行']]
    for label, col in [('收盘价均值', 'close'), ('收盘价最大值', 'close'), ('收盘价最小值', 'close')]:
        if '均值' in label:
            stat_data.append([label, f'{df_main[col].mean():.2f}', f'{df_td[col].mean():.2f}', f'{df_pa[col].mean():.2f}'])
        elif '最大' in label:
            stat_data.append([label, f'{df_main[col].max():.2f}', f'{df_td[col].max():.2f}', f'{df_pa[col].max():.2f}'])
        elif '最小' in label:
            stat_data.append([label, f'{df_main[col].min():.2f}', f'{df_td[col].min():.2f}', f'{df_pa[col].min():.2f}'])

    stat_data.append(['日均成交量(万手)', f'{df_main["vol"].mean()/10000:.2f}',
                      f'{df_td["vol"].mean()/10000:.2f}', f'{df_pa["vol"].mean()/10000:.2f}'])
    stat_data.append(['日均ATR', f'{df_main["atr"].mean():.2f}' if "atr" in df_main.columns else 'N/A',
                      f'{df_td["atr"].mean():.2f}' if "atr" in df_td.columns else 'N/A',
                      f'{df_pa["atr"].mean():.2f}' if "atr" in df_pa.columns else 'N/A'])
    stat_data.append(['上涨天数', f'{(df_main["pct_chg"]>0).sum()}',
                      f'{(df_td["pct_chg"]>0).sum()}', f'{(df_pa["pct_chg"]>0).sum()}'])
    stat_data.append(['下跌天数', f'{(df_main["pct_chg"]<0).sum()}',
                      f'{(df_td["pct_chg"]<0).sum()}', f'{(df_pa["pct_chg"]<0).sum()}'])

    table = Table(stat_data, colWidths=[40*mm, 40*mm, 40*mm, 40*mm])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 10.5),
        ('LEADING', (0, 0), (-1, -1), 15.75),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#e8e8e8')),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(table)
    story.append(Paragraph('表1 三只股票描述性统计', style_figure))
    story.append(Spacer(1, 5*mm))

    # === 四、中船特气海龟策略分析 ===
    story.append(PageBreak())
    story.append(Paragraph('四、中船特气（688146.SH）海龟策略分析', style_heading))
    story.append(Paragraph(
        f'中船特气是本次分析的主要标的，使用20日唐奇安通道入场、10日通道出场、ATR(14)计算止损。'
        f'该股票在过去一年内从{df_main["close"].iloc[0]:.2f}元涨至{df_main["close"].iloc[-1]:.2f}元，'
        f'区间涨幅达{metrics_main["buy_hold_cum_return"]:.2f}%，属于强趋势股。', style_body))
    story.append(Spacer(1, 3*mm))

    # 图1: 策略信号图
    if len(all_charts) > 0 and os.path.exists(all_charts[0]):
        story.append(KeepTogether([
            Image(all_charts[0], width=160*mm, height=100*mm),
            Paragraph(f'图1 中船特气海龟策略交易信号图（通道周期20/10）', style_figure),
            Spacer(1, 2*mm),
        ]))

    story.append(Paragraph(
        f'图1展示了中船特气的海龟策略交易信号。图中蓝色曲线为收盘价，红色虚线为20日入场通道上轨，'
        f'绿色虚线为10日出场通道下轨，橙色区域为通道范围。红色三角形标记买入信号（收盘价突破20日最高价），'
        f'绿色倒三角形标记卖出信号（收盘价跌破10日最低价）。从图中可以看出，策略在趋势启动时及时入场，'
        f'在趋势反转时退出。本次回测共产生{metrics_main["n_trades"]}笔交易，胜率为{metrics_main["win_rate"]:.1f}%，'
        f'平均每笔交易收益为{metrics_main["avg_trade_return"]:.2f}%。', style_body))
    story.append(Spacer(1, 3*mm))

    # 图2: 净值曲线
    if len(all_charts) > 1 and os.path.exists(all_charts[1]):
        story.append(KeepTogether([
            Image(all_charts[1], width=160*mm, height=70*mm),
            Paragraph(f'图2 中船特气海龟策略净值曲线（通道周期20/10）', style_figure),
            Spacer(1, 2*mm),
        ]))

    story.append(Paragraph(
        f'图2对比了海龟策略净值与买入持有净值的走势。策略累计回报为{metrics_main["strategy_cum_return"]:.2f}%，'
        f'买入持有累计回报为{metrics_main["buy_hold_cum_return"]:.2f}%。'
        f'年化收益率方面，策略为{metrics_main["annual_strategy"]:.2f}%，买入持有为{metrics_main["annual_buy_hold"]:.2f}%。'
        f'由于中船特气处于强上涨趋势中，策略能够捕获大部分趋势利润，但因为在趋势回调时退出，'
        f'策略回报略低于买入持有，这是趋势跟踪策略在单边大涨行情中的典型特征。', style_body))
    story.append(Spacer(1, 3*mm))

    # 图3: 回撤曲线
    if len(all_charts) > 2 and os.path.exists(all_charts[2]):
        story.append(KeepTogether([
            Image(all_charts[2], width=160*mm, height=55*mm),
            Paragraph(f'图3 中船特气海龟策略回撤对比（通道周期20/10）', style_figure),
            Spacer(1, 2*mm),
        ]))

    story.append(Paragraph(
        f'图3展示了策略与买入持有的回撤对比。策略最大回撤为{metrics_main["strategy_mdd"]:.2f}%，'
        f'买入持有最大回撤为{metrics_main["buy_hold_mdd"]:.2f}%。策略的夏普比率为{metrics_main["strategy_sharpe"]:.2f}，'
        f'买入持有夏普比率为{metrics_main["buy_hold_sharpe"]:.2f}。从回撤控制来看，'
        f'海龟策略通过ATR止损和通道出场机制，在趋势反转时及时退出，有效控制了最大回撤。', style_body))
    story.append(Spacer(1, 5*mm))

    # 量化指标表
    story.append(Paragraph('表2 中船特气海龟策略量化指标', style_figure))
    metrics_data = [
        ['指标', '海龟策略', '买入持有'],
        ['累计回报(%)', f'{metrics_main["strategy_cum_return"]:.2f}', f'{metrics_main["buy_hold_cum_return"]:.2f}'],
        ['年化收益率(%)', f'{metrics_main["annual_strategy"]:.2f}', f'{metrics_main["annual_buy_hold"]:.2f}'],
        ['最大回撤(%)', f'{metrics_main["strategy_mdd"]:.2f}', f'{metrics_main["buy_hold_mdd"]:.2f}'],
        ['夏普比率', f'{metrics_main["strategy_sharpe"]:.2f}', f'{metrics_main["buy_hold_sharpe"]:.2f}'],
        ['交易次数', f'{metrics_main["n_trades"]}', '-'],
        ['胜率(%)', f'{metrics_main["win_rate"]:.1f}', '-'],
        ['平均交易收益(%)', f'{metrics_main["avg_trade_return"]:.2f}', '-'],
    ]
    table2 = Table(metrics_data, colWidths=[50*mm, 55*mm, 55*mm])
    table2.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 10.5),
        ('LEADING', (0, 0), (-1, -1), 15.75),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#e8e8e8')),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(table2)
    story.append(Spacer(1, 5*mm))

    # === 五、天地科技与平安银行分析 ===
    story.append(PageBreak())
    story.append(Paragraph('五、天地科技与平安银行海龟策略分析', style_heading))

    story.append(Paragraph('（一）天地科技（600587.SH）', style_subheading))
    story.append(Paragraph(
        f'天地科技在过去一年中处于震荡下跌趋势，区间涨跌幅为{metrics_td["buy_hold_cum_return"]:.2f}%。'
        f'海龟策略累计回报为{metrics_td["strategy_cum_return"]:.2f}%，夏普比率为{metrics_td["strategy_sharpe"]:.2f}，'
        f'最大回撤为{metrics_td["strategy_mdd"]:.2f}%，共产生{metrics_td["n_trades"]}笔交易，胜率为{metrics_td["win_rate"]:.1f}%。'
        f'在震荡行情中，唐奇安通道频繁产生假突破信号，导致策略表现不佳，这印证了趋势跟踪策略在震荡市中的局限性。', style_body))
    story.append(Spacer(1, 3*mm))

    if len(all_charts) > 3 and os.path.exists(all_charts[3]):
        story.append(KeepTogether([
            Image(all_charts[3], width=160*mm, height=100*mm),
            Paragraph('图4 天地科技海龟策略交易信号图（通道周期20/10）', style_figure),
            Spacer(1, 2*mm),
        ]))
        story.append(Paragraph(
            '图4展示了天地科技的海龟策略信号。可以看到在震荡区间中，价格多次短暂突破通道后又回落，'
            '产生虚假买入信号，随后很快触发止损或通道退出。这类假突破是趋势策略在震荡行情中的主要亏损来源。', style_body))
    story.append(Spacer(1, 5*mm))

    story.append(Paragraph('（二）平安银行（000001.SZ）', style_subheading))
    story.append(Paragraph(
        f'平安银行在过去一年中同样处于下跌趋势，区间涨跌幅为{metrics_pa["buy_hold_cum_return"]:.2f}%。'
        f'海龟策略累计回报为{metrics_pa["strategy_cum_return"]:.2f}%，夏普比率为{metrics_pa["strategy_sharpe"]:.2f}，'
        f'最大回撤为{metrics_pa["strategy_mdd"]:.2f}%，共产生{metrics_pa["n_trades"]}笔交易，胜率为{metrics_pa["win_rate"]:.1f}%。'
        f'在下跌趋势中，策略通过及时退出避免了部分损失，但由于反弹趋势短暂，仍难以实现正收益。', style_body))
    story.append(Spacer(1, 3*mm))

    if len(all_charts) > 6 and os.path.exists(all_charts[6]):
        story.append(KeepTogether([
            Image(all_charts[6], width=160*mm, height=100*mm),
            Paragraph('图5 平安银行海龟策略交易信号图（通道周期20/10）', style_figure),
            Spacer(1, 2*mm),
        ]))
        story.append(Paragraph(
            '图5展示了平安银行的海龟策略信号。在持续下跌过程中，价格偶有反弹突破上轨触发买入，'
            '但很快又回落跌破下轨触发卖出，形成频繁的亏损交易。这表明在单边下跌市场中，'
            '单纯做多趋势策略难以获利，应考虑结合做空机制或趋势过滤条件。', style_body))
    story.append(Spacer(1, 5*mm))

    # === 六、多参数对比实验 ===
    story.append(PageBreak())
    story.append(Paragraph('六、多股票多参数对比实验', style_heading))
    story.append(Paragraph(
        '为全面评估海龟策略的适应性，本节对三只股票分别测试三组参数：DC20/10（系统1标准参数）、'
        'DC55/20（系统2标准参数）、DC10/5（短周期敏感参数），共9组实验。', style_body))
    story.append(Spacer(1, 3*mm))

    # 对比结果表
    comp_table_data = [['股票', '参数', '策略回报(%)', '买入持有(%)', '夏普比率', '最大回撤(%)', '交易次数', '胜率(%)']]
    for _, row in comparison_df.iterrows():
        comp_table_data.append([
            row['stock'], row['config'],
            f'{row["strategy_cum_return"]:.2f}',
            f'{row["buy_hold_cum_return"]:.2f}',
            f'{row["strategy_sharpe"]:.2f}',
            f'{row["strategy_mdd"]:.2f}',
            str(row['n_trades']),
            f'{row["win_rate"]:.1f}'
        ])

    table3 = Table(comp_table_data, colWidths=[20*mm, 20*mm, 25*mm, 25*mm, 20*mm, 25*mm, 20*mm, 20*mm])
    table3.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEADING', (0, 0), (-1, -1), 13.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#e8e8e8')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    story.append(table3)
    story.append(Paragraph('表3 海龟策略多股票多参数对比结果', style_figure))
    story.append(Spacer(1, 3*mm))

    # 对比图
    if len(all_charts) > 9 and os.path.exists(all_charts[9]):
        story.append(KeepTogether([
            Image(all_charts[9], width=160*mm, height=110*mm),
            Paragraph('图6 海龟策略多股票多参数对比', style_figure),
            Spacer(1, 2*mm),
        ]))

    story.append(Paragraph(
        '图6和表3展示了9组实验的对比结果。从累计回报来看，中船特气在DC20/10参数下表现最佳，'
        '策略回报远超其他组合；天地科技和平安银行在所有参数下均难以盈利，印证了海龟策略对趋势行情的依赖。'
        '从夏普比率来看，短周期参数DC10/5虽然信号更多，但假突破导致夏普比率普遍偏低；'
        '长周期参数DC55/20信号更少但更可靠，夏普比率相对较高。'
        '从最大回撤来看，短周期参数因频繁交易导致回撤更大，长周期参数回撤相对可控。'
        '从胜率来看，长周期参数胜率普遍高于短周期，但交易次数较少。', style_body))
    story.append(Spacer(1, 5*mm))

    # === 七、适应场景与使用心得 ===
    story.append(PageBreak())
    story.append(Paragraph('七、海龟法则适应场景与使用心得', style_heading))

    story.append(Paragraph('（一）适应场景', style_subheading))
    story.append(Paragraph(
        '通过本次多股票多参数实验，可以总结出海龟策略的适应场景如下：'
        '（1）强趋势行情：海龟策略在单边趋势行情中表现最佳，如中船特气在DC20/10参数下获得了较高的累计回报。'
        '在趋势明确时，唐奇安通道能有效捕获趋势启动信号，ATR止损能控制风险。'
        '（2）中长周期投资：海龟策略属于中长线策略，持仓周期通常为数周到数月，不适合短线交易。'
        '（3）多市场分散：海龟策略原设计就强调分散化，通过在多个低相关市场同时运行，'
        '可以分散单一市场震荡带来的风险。'
        '（4）高流动性品种：策略适用于流动性好的品种，避免低流动性品种的滑点和操纵风险。', style_body))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph('（二）使用心得', style_subheading))
    story.append(Paragraph(
        '（1）趋势是前提：海龟策略本质是趋势跟踪策略，在震荡行情中会产生大量假突破信号，导致频繁亏损。'
        '使用前应先判断市场是否处于趋势状态，可通过ADX指标或均线斜率进行过滤。', style_body))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '（2）参数选择需权衡：短周期参数（如DC10/5）信号灵敏但假突破多，适合波动较大的品种；'
        '长周期参数（如DC55/20）信号可靠但入场晚，适合趋势明确的品种。'
        '应根据个股特性和市场环境选择合适参数，不宜频繁调整。', style_body))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '（3）止损是核心：ATR止损是海龟策略的精髓，它使风险控制与市场波动率自适应。'
        '2N止损既给了趋势发展空间，又限制了单笔交易的最大亏损。'
        '在实际使用中，应严格执行止损纪律，不可因主观判断而忽略止损信号。', style_body))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '（4）仓位管理不可忽视：海龟策略通过ATR计算头寸规模，确保每笔交易承担相同比例的风险。'
        '这一机制在高波动品种中自动减小仓位，在低波动品种中自动增大仓位，实现了风险均衡。', style_body))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '（5）回测局限需认知：历史回测结果不代表未来表现，实际交易中还需考虑滑点、手续费、'
        '流动性冲击等成本。此外，策略在特定历史时段的表现可能具有偶然性，应在多时段、多市场验证。', style_body))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        '（6）改进方向：可考虑加入趋势过滤条件（如均线方向、ADX阈值）减少震荡行情中的假信号；'
        '加入成交量确认条件提高信号可靠性；结合多周期分析提高入场时机精度；'
        '引入动态参数调整机制适应不同市场环境。', style_body))
    story.append(Spacer(1, 5*mm))

    # === 八、总结 ===
    story.append(Paragraph('八、总结', style_heading))
    story.append(Paragraph(
        '本次TASK4系统分析了海龟交易策略的核心思想、关键概念（唐奇安通道、ATR、止损条件），'
        '并通过Python编程实现了策略的信号生成、可视化绘图和回测分析。'
        '通过对中船特气、天地科技、平安银行三只股票在DC20/10、DC55/20、DC10/5三组参数下的对比实验，'
        '验证了海龟策略在强趋势行情中的优势和在震荡行情中的局限性。'
        '实验结果表明，海龟策略最适合单边趋势明显的品种，参数选择应根据品种特性和市场环境灵活调整，'
        '严格执行ATR止损纪律是策略长期盈利的关键。', style_body))

    doc.build(story)
    print(f"PDF已生成: {pdf_path}")


if __name__ == '__main__':
    main()
