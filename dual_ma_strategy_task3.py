# -*- coding: utf-8 -*-
"""
TASK3: 双均线策略分析
1. 加载股价数据，计算MA5/MA15均线
2. 生成金叉/死叉买卖信号
3. 绘制可视化图表（股价、均线、买卖信号标记）
4. 策略回测，计算MDD、夏普比率、累计回报等指标
5. 多股票多周期参数对比实验
6. 生成PDF报告
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 字体配置
# ============================================================
font_path = 'C:/Windows/Fonts/simsun.ttc'
font_prop = fm.FontProperties(fname=font_path, size=12)
fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# ============================================================
# 策略核心函数
# ============================================================

def load_data(filepath):
    """加载股价数据"""
    df = pd.read_csv(filepath)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').reset_index(drop=True)
    return df


def calculate_ma(df, short_window=5, long_window=15):
    """计算短期和长期移动平均线"""
    df = df.copy()
    df['ma_short'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['ma_long'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    return df


def generate_signals(df):
    """
    生成交易信号
    金叉: 短期均线上穿长期均线 -> 买入信号 (+1)
    死叉: 短期均线下穿长期均线 -> 卖出信号 (-1)
    """
    df = df.copy()
    df['signal'] = 0
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
    df['position_change'] = df['signal'].diff()

    # 标记买入卖出点
    df['buy_signal'] = df['position_change'] == 2   # 从空头(-1)到多头(1)
    df['sell_signal'] = df['position_change'] == -2  # 从多头(1)到空头(-1)

    # 兼容从0到1的情况
    df.loc[(df['position_change'] == 1) & (df['signal'] == 1), 'buy_signal'] = True
    df.loc[(df['position_change'] == -1) & (df['signal'] == 0), 'sell_signal'] = True

    return df


def backtest(df, initial_capital=100000):
    """
    策略回测
    - 满仓买入/卖出策略
    - 计算每日收益、累计收益、资金曲线
    """
    df = df.copy()
    # 每日收益率
    df['daily_return'] = df['close'].pct_change()

    # 策略持仓：signal=1时持仓，signal=0时空仓
    # 使用前一天的signal决定今天是否持仓
    df['position'] = df['signal'].shift(1).fillna(0)

    # 策略每日收益
    df['strategy_return'] = df['position'] * df['daily_return']

    # 累计收益
    df['cum_market'] = (1 + df['daily_return']).cumprod()
    df['cum_strategy'] = (1 + df['strategy_return']).cumprod()

    # 资金曲线
    df['market_value'] = initial_capital * df['cum_market']
    df['strategy_value'] = initial_capital * df['cum_strategy']

    return df


def calculate_metrics(df, annual_factor=252, risk_free_rate=0.03):
    """计算量化评价指标"""
    returns = df['strategy_return'].dropna()

    # 累计回报
    cum_return = df['cum_strategy'].iloc[-1] - 1
    market_cum_return = df['cum_market'].iloc[-1] - 1

    # 年化收益率
    n_days = len(returns)
    annual_return = (1 + cum_return) ** (annual_factor / n_days) - 1

    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(annual_factor)

    # 夏普比率
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0

    # 最大回撤
    cummax = df['strategy_value'].cummax()
    drawdown = (df['strategy_value'] - cummax) / cummax
    max_drawdown = drawdown.min()

    # 交易次数
    buy_count = df['buy_signal'].sum()
    sell_count = df['sell_signal'].sum()
    trade_count = min(buy_count, sell_count)

    # 胜率
    trade_returns = []
    in_position = False
    entry_price = 0
    for _, row in df.iterrows():
        if row['buy_signal'] and not in_position:
            entry_price = row['close']
            in_position = True
        elif row['sell_signal'] and in_position:
            trade_return = (row['close'] - entry_price) / entry_price
            trade_returns.append(trade_return)
            in_position = False

    win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns) if trade_returns else 0

    return {
        'cum_return': cum_return,
        'market_cum_return': market_cum_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'buy_count': int(buy_count),
        'sell_count': int(sell_count),
        'trade_count': trade_count,
        'win_rate': win_rate,
        'n_days': n_days
    }


# ============================================================
# 可视化函数
# ============================================================

def plot_strategy(df, stock_name, short_w, long_w, chart_num, output_dir):
    """绘制策略可视化图：股价、均线、买卖信号"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)

    # 上图：股价和均线
    ax1.plot(df['trade_date'], df['close'], color='#333333', linewidth=1, label='收盘价', zorder=1)
    ax1.plot(df['trade_date'], df['ma_short'], color='#e74c3c', linewidth=1.2,
             label=f'MA{short_w}(短期)', zorder=2)
    ax1.plot(df['trade_date'], df['ma_long'], color='#2980b9', linewidth=1.2,
             label=f'MA{long_w}(长期)', zorder=2)

    # 标记买入信号（红色上三角）
    buy_points = df[df['buy_signal']]
    if len(buy_points) > 0:
        ax1.scatter(buy_points['trade_date'], buy_points['close'],
                    marker='^', color='red', s=120, zorder=5, label='买入信号(金叉)')

    # 标记卖出信号（绿色下三角）
    sell_points = df[df['sell_signal']]
    if len(sell_points) > 0:
        ax1.scatter(sell_points['trade_date'], sell_points['close'],
                    marker='v', color='green', s=120, zorder=5, label='卖出信号(死叉)')

    ax1.set_title(f'图{chart_num} {stock_name}双均线策略(MA{short_w}/MA{long_w})-股价与交易信号',
                  fontproperties=font_prop, fontsize=13, fontweight='bold')
    ax1.set_ylabel('价格(元)', fontproperties=font_prop, fontsize=12)
    ax1.legend(prop=font_prop, fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)

    # 下图：资金曲线对比
    ax2.plot(df['trade_date'], df['market_value'], color='#95a5a6', linewidth=1,
             label='买入持有策略', alpha=0.8)
    ax2.plot(df['trade_date'], df['strategy_value'], color='#e74c3c', linewidth=1.2,
             label='双均线策略')
    ax2.set_title(f'资金曲线对比', fontproperties=font_prop, fontsize=12)
    ax2.set_ylabel('资金(元)', fontproperties=font_prop, fontsize=12)
    ax2.set_xlabel('日期', fontproperties=font_prop, fontsize=12)
    ax2.legend(prop=font_prop, fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = f'{output_dir}/chart{chart_num}_strategy_{stock_name}_MA{short_w}_{long_w}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def plot_drawdown(df, stock_name, short_w, long_w, chart_num, output_dir):
    """绘制回撤曲线"""
    fig, ax = plt.subplots(figsize=(14, 4))

    cummax = df['strategy_value'].cummax()
    drawdown = (df['strategy_value'] - cummax) / cummax * 100

    ax.fill_between(df['trade_date'], drawdown, 0, color='#e74c3c', alpha=0.3, label='策略回撤')
    ax.plot(df['trade_date'], drawdown, color='#c0392b', linewidth=1)

    # 标记最大回撤点
    mdd_idx = drawdown.idxmin()
    mdd_date = df.loc[mdd_idx, 'trade_date']
    mdd_val = drawdown[mdd_idx]
    ax.annotate(f'最大回撤: {mdd_val:.2f}%',
                xy=(mdd_date, mdd_val),
                xytext=(mdd_date, mdd_val * 0.5),
                fontproperties=font_prop, fontsize=11,
                arrowprops=dict(arrowstyle='->', color='black'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    ax.set_title(f'图{chart_num} {stock_name}双均线策略(MA{short_w}/MA{long_w})-回撤分析',
                 fontproperties=font_prop, fontsize=13, fontweight='bold')
    ax.set_ylabel('回撤幅度(%)', fontproperties=font_prop, fontsize=12)
    ax.set_xlabel('日期', fontproperties=font_prop, fontsize=12)
    ax.legend(prop=font_prop, fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = f'{output_dir}/chart{chart_num}_drawdown_{stock_name}_MA{short_w}_{long_w}.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def plot_comparison(results, chart_num, output_dir):
    """绘制多参数对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = [r['label'] for r in results]

    # 累计回报对比
    cum_returns = [r['metrics']['cum_return'] * 100 for r in results]
    market_returns = [r['metrics']['market_cum_return'] * 100 for r in results]
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w/2, cum_returns, w, color='#e74c3c', label='双均线策略')
    axes[0].bar(x + w/2, market_returns, w, color='#3498db', label='买入持有')
    axes[0].set_title(f'累计回报对比(%)', fontproperties=font_prop, fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontproperties=font_prop, fontsize=9, rotation=30, ha='right')
    axes[0].legend(prop=font_prop, fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 夏普比率对比
    sharpes = [r['metrics']['sharpe_ratio'] for r in results]
    axes[1].bar(x, sharpes, color='#2ecc71', width=0.5)
    axes[1].set_title(f'夏普比率对比', fontproperties=font_prop, fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontproperties=font_prop, fontsize=9, rotation=30, ha='right')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')

    # 最大回撤对比
    mdds = [abs(r['metrics']['max_drawdown']) * 100 for r in results]
    axes[2].bar(x, mdds, color='#e67e22', width=0.5)
    axes[2].set_title(f'最大回撤对比(%)', fontproperties=font_prop, fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontproperties=font_prop, fontsize=9, rotation=30, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'图{chart_num} 多股票多周期双均线策略效果对比',
                 fontproperties=font_prop, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = f'{output_dir}/chart{chart_num}_comparison.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ============================================================
# 主执行流程
# ============================================================

output_dir = 'E:/量化交易：AI大模型辅助的金融交易策略'

# --- 1. 主策略：中船特气 MA5/MA15 ---
print("=" * 60)
print("1. 中船特气(688146.SH) MA5/MA15 双均线策略")
print("=" * 60)

df1 = load_data(f'{output_dir}/中船特气_688146_daily_data.csv')
df1 = calculate_ma(df1, short_window=5, long_window=15)
df1 = generate_signals(df1)
df1 = backtest(df1, initial_capital=100000)
metrics1 = calculate_metrics(df1)

print(f"数据范围: {df1['trade_date'].iloc[0].strftime('%Y-%m-%d')} ~ {df1['trade_date'].iloc[-1].strftime('%Y-%m-%d')}")
print(f"交易天数: {metrics1['n_days']}")
print(f"买入信号: {metrics1['buy_count']}次, 卖出信号: {metrics1['sell_count']}次")
print(f"累计回报: {metrics1['cum_return']:.2%} (买入持有: {metrics1['market_cum_return']:.2%})")
print(f"年化收益: {metrics1['annual_return']:.2%}")
print(f"年化波动: {metrics1['annual_volatility']:.2%}")
print(f"夏普比率: {metrics1['sharpe_ratio']:.4f}")
print(f"最大回撤: {metrics1['max_drawdown']:.2%}")
print(f"胜率: {metrics1['win_rate']:.2%}")

chart1_path = plot_strategy(df1, '中船特气', 5, 15, 1, output_dir)
chart2_path = plot_drawdown(df1, '中船特气', 5, 15, 2, output_dir)
print(f"图表已保存: {chart1_path}, {chart2_path}")

# --- 2. 多股票多周期对比实验 ---
print("\n" + "=" * 60)
print("2. 多股票多周期参数对比实验")
print("=" * 60)

stocks = [
    {'name': '中船特气', 'code': '688146.SH', 'file': f'{output_dir}/中船特气_688146_daily_data.csv'},
    {'name': '天地科技', 'code': '600587.SH', 'file': f'{output_dir}/tiandi_keji_600587_daily_data.csv'},
    {'name': '平安银行', 'code': '000001.SZ', 'file': f'{output_dir}/平安银行_000001_daily_data.csv'},
]

# 多组参数
param_sets = [
    (5, 15),
    (5, 20),
    (10, 20),
    (5, 30),
    (10, 30),
]

all_results = []
for stock in stocks:
    for short_w, long_w in param_sets:
        df = load_data(stock['file'])
        df = calculate_ma(df, short_window=short_w, long_window=long_w)
        df = generate_signals(df)
        df = backtest(df, initial_capital=100000)
        m = calculate_metrics(df)

        label = f"{stock['name']}\nMA{short_w}/{long_w}"
        all_results.append({
            'label': label,
            'stock': stock['name'],
            'code': stock['code'],
            'short_w': short_w,
            'long_w': long_w,
            'metrics': m,
            'df': df
        })

        print(f"{stock['name']} MA{short_w}/{long_w}: "
              f"累计回报={m['cum_return']:.2%}, 夏普={m['sharpe_ratio']:.4f}, "
              f"最大回撤={m['max_drawdown']:.2%}, 胜率={m['win_rate']:.2%}")

# 对比图
chart3_path = plot_comparison(all_results, 3, output_dir)
print(f"\n对比图已保存: {chart3_path}")

# --- 3. 保存带信号的数据 ---
df1_out = df1[['trade_date', 'close', 'ma_short', 'ma_long', 'signal',
               'buy_signal', 'sell_signal', 'daily_return', 'strategy_return',
               'cum_market', 'cum_strategy', 'market_value', 'strategy_value']].copy()
df1_out['trade_date'] = df1_out['trade_date'].dt.strftime('%Y%m%d')
df1_out.to_csv(f'{output_dir}/中船特气_688146_ma_strategy.csv', index=False, encoding='utf-8-sig')
print("策略数据已保存: 中船特气_688146_ma_strategy.csv")

# --- 4. 对比结果表 ---
comparison_df = pd.DataFrame([
    {
        '股票': r['stock'],
        '代码': r['code'],
        '短均线': r['short_w'],
        '长均线': r['long_w'],
        '累计回报': f"{r['metrics']['cum_return']:.2%}",
        '买入持有回报': f"{r['metrics']['market_cum_return']:.2%}",
        '年化收益': f"{r['metrics']['annual_return']:.2%}",
        '年化波动': f"{r['metrics']['annual_volatility']:.2%}",
        '夏普比率': f"{r['metrics']['sharpe_ratio']:.4f}",
        '最大回撤': f"{r['metrics']['max_drawdown']:.2%}",
        '交易次数': r['metrics']['trade_count'],
        '胜率': f"{r['metrics']['win_rate']:.2%}",
    }
    for r in all_results
])
comparison_df.to_csv(f'{output_dir}/双均线策略对比结果.csv', index=False, encoding='utf-8-sig')
print("对比结果已保存: 双均线策略对比结果.csv")

# --- 5. 选取代表性股票绘图(天地科技 MA5/MA15 和 平安银行 MA5/MA15) ---
df2 = load_data(stocks[1]['file'])
df2 = calculate_ma(df2, 5, 15)
df2 = generate_signals(df2)
df2 = backtest(df2)
chart4_path = plot_strategy(df2, '天地科技', 5, 15, 4, output_dir)

df3 = load_data(stocks[2]['file'])
df3 = calculate_ma(df3, 5, 15)
df3 = generate_signals(df3)
df3 = backtest(df3)
chart5_path = plot_strategy(df3, '平安银行', 5, 15, 5, output_dir)

metrics2 = calculate_metrics(df2)
metrics3 = calculate_metrics(df3)
print(f"\n天地科技 MA5/15: 累计回报={metrics2['cum_return']:.2%}, 夏普={metrics2['sharpe_ratio']:.4f}")
print(f"平安银行 MA5/15: 累计回报={metrics3['cum_return']:.2%}, 夏普={metrics3['sharpe_ratio']:.4f}")

# ============================================================
# PDF 报告生成
# ============================================================
print("\n" + "=" * 60)
print("3. 生成PDF报告")
print("=" * 60)

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, KeepTogether, PageBreak)
from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# 注册宋体
pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))

# 段落样式
style_title = ParagraphStyle(
    'CustomTitle', fontName='SimSun', fontSize=16, leading=24,
    alignment=TA_CENTER, spaceBefore=0, spaceAfter=12
)
style_h1 = ParagraphStyle(
    'CustomH1', fontName='SimSun', fontSize=14, leading=21,
    alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=6
)
style_body = ParagraphStyle(
    'CustomBody', fontName='SimSun', fontSize=10.5, leading=15.75,
    alignment=TA_JUSTIFY, spaceBefore=0, spaceAfter=0
)
style_caption = ParagraphStyle(
    'Caption', fontName='SimSun', fontSize=10.5, leading=15.75,
    alignment=TA_CENTER, spaceBefore=0, spaceAfter=0
)

def add_image(doc, img_path, caption_text, max_width=480, max_height=350):
    """添加图片和标题"""
    from PIL import Image as PILImage
    pil_img = PILImage.open(img_path)
    w, h = pil_img.size
    ratio = min(max_width / w, max_height / h)
    display_w = w * ratio
    display_h = h * ratio
    doc.append(Image(img_path, width=display_w, height=display_h))
    doc.append(Spacer(1, 3 * mm))
    doc.append(Paragraph(caption_text, style_caption))
    doc.append(Spacer(1, 5 * mm))

doc_elements = []

# === 封面标题 ===
doc_elements.append(Paragraph("双均线策略分析与回测报告", style_title))
doc_elements.append(Spacer(1, 8 * mm))
doc_elements.append(Paragraph("——基于中船特气、天地科技、平安银行的量化策略研究", style_caption))
doc_elements.append(Spacer(1, 10 * mm))

# === 一、双均线策略概述 ===
doc_elements.append(Paragraph("一、双均线策略概述", style_h1))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph(
    "双均线策略（Dual Moving Average Strategy）是量化交易中最经典的趋势跟踪策略之一。"
    "其核心思想是利用两条不同周期的移动平均线（短期均线和长期均线）的交叉关系来判断市场趋势的转折，"
    "从而产生买入和卖出信号。短期均线反映近期价格变化，对价格波动较为敏感；"
    "长期均线反映中长期趋势，较为稳定。当短期均线与长期均线发生交叉时，意味着市场趋势可能发生反转。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>1. 金叉（Golden Cross）</b>", style_body))
doc_elements.append(Paragraph(
    "金叉是指短期移动平均线从下方向上穿越长期移动平均线，形成向上交叉的形态。"
    "这一信号表明近期价格走势开始强于中长期趋势，市场可能由跌转涨，是买入信号。"
    "具体而言，当MA5（5日均线）从下方上穿MA15（15日均线）时，说明短期买盘力量增强，"
    "价格有望继续上涨，投资者应在金叉出现时买入建仓。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>2. 死叉（Death Cross）</b>", style_body))
doc_elements.append(Paragraph(
    "死叉是指短期移动平均线从上方向下穿越长期移动平均线，形成向下交叉的形态。"
    "这一信号表明近期价格走势开始弱于中长期趋势，市场可能由涨转跌，是卖出信号。"
    "具体而言，当MA5从上方下穿MA15时，说明短期卖盘力量增强，价格可能继续下跌，"
    "投资者应在死叉出现时卖出平仓。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>3. 策略优缺点</b>", style_body))
doc_elements.append(Paragraph(
    "优点：策略逻辑简单直观，易于理解和编程实现；能够有效捕捉趋势性行情，在单边上涨或下跌市场中表现优异；"
    "信号客观明确，避免了主观情绪干扰。"
    "缺点：在震荡行情中容易产生频繁的虚假信号，导致反复亏损；"
    "均线本身具有滞后性，信号出现时趋势可能已经走了一段，存在利润回吐问题；"
    "对参数选择敏感，不同周期组合效果差异较大。", style_body))
doc_elements.append(Spacer(1, 5 * mm))

# === 二、量化评价指标说明 ===
doc_elements.append(Paragraph("二、量化策略效果评价指标", style_h1))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>1. 最大回撤（Maximum Drawdown, MDD）</b>", style_body))
doc_elements.append(Paragraph(
    "最大回撤是衡量策略风险的重要指标，描述在选定周期内，策略资产净值从任一历史最高点"
    "到后续最低点的最大跌幅。计算公式为：MDD = max(1 - 当日净值 / 当日之前最高净值)。"
    "最大回撤反映了策略可能出现的最糟糕情况，即投资者跟随该策略可能面临的最大亏损幅度。"
    "MDD越小，说明策略风险控制能力越强。一般认为MDD低于20%为可接受范围，超过40%则风险较高。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>2. 夏普比率（Sharpe Ratio）</b>", style_body))
doc_elements.append(Paragraph(
    "夏普比率是衡量风险调整后收益的核心指标，表示每承受一单位总风险能获得多少超额回报。"
    "计算公式为：Sharpe Ratio = (策略年化收益率 - 无风险利率) / 策略收益年化标准差。"
    "其中无风险利率通常取一年期国债收益率（本文取3%）。夏普比率越大，说明策略在同等风险下获得的收益越高。"
    "一般而言，夏普比率大于1视为良好，大于2视为优秀，小于0则表示策略收益不及无风险利率。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>3. 累计回报（Cumulative Return）</b>", style_body))
doc_elements.append(Paragraph(
    "累计回报是策略在整个回测期间的总收益率，反映策略的绝对盈利能力。"
    "计算公式为：Cumulative Return = (期末净值 - 期初净值) / 期初净值。"
    "本文同时计算买入持有策略的累计回报作为基准，用于对比双均线策略是否跑赢市场。"
    "需要注意的是，累计回报仅衡量收益不考虑风险，高回报可能伴随高风险，"
    "因此应结合夏普比率和最大回撤综合评估策略效果。", style_body))
doc_elements.append(Spacer(1, 5 * mm))

doc_elements.append(PageBreak())

# === 三、中船特气双均线策略分析 ===
doc_elements.append(Paragraph("三、中船特气(688146.SH)双均线策略分析", style_h1))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph(
    f"本节以中船特气(688146.SH)为标的，设定短期均线周期为5日、长期均线周期为15日，"
    f"回测区间为{df1['trade_date'].iloc[0].strftime('%Y-%m-%d')}至{df1['trade_date'].iloc[-1].strftime('%Y-%m-%d')}，"
    f"共{metrics1['n_days']}个交易日，初始资金10万元。"
    f"回测期间共产生{metrics1['buy_count']}次买入信号和{metrics1['sell_count']}次卖出信号。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

# 指标汇总表
metrics_table_data = [
    ['指标', '双均线策略', '买入持有'],
    ['累计回报', f"{metrics1['cum_return']:.2%}", f"{metrics1['market_cum_return']:.2%}"],
    ['年化收益率', f"{metrics1['annual_return']:.2%}", '-'],
    ['年化波动率', f"{metrics1['annual_volatility']:.2%}", '-'],
    ['夏普比率', f"{metrics1['sharpe_ratio']:.4f}", '-'],
    ['最大回撤', f"{metrics1['max_drawdown']:.2%}", '-'],
    ['交易次数', str(metrics1['trade_count']), '-'],
    ['胜率', f"{metrics1['win_rate']:.2%}", '-'],
]
table = Table(metrics_table_data, colWidths=[120, 150, 150])
table.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
    ('FONTSIZE', (0, 0), (-1, -1), 10.5),
    ('LEADING', (0, 0), (-1, -1), 15.75),
    ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#34495e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#ecf0f1')]),
]))
doc_elements.append(table)
doc_elements.append(Spacer(1, 5 * mm))

# 图1
add_image(doc_elements, chart1_path,
          "图1 中船特气(688146.SH)双均线策略(MA5/MA15)股价走势与交易信号")

doc_elements.append(Paragraph(
    f"<b>图1解读：</b>上图展示了中船特气的收盘价走势及MA5、MA15两条均线。"
    f"红色上三角标记金叉买入信号，绿色下三角标记死叉卖出信号。"
    f"从图中可见，中船特气在过去一年经历了大幅上涨，股价从约28元飙升至近370元。"
    f"在强劲趋势中，MA5/MA15双均线策略能较好地捕捉上涨段，但由于均线滞后性，"
    f"每次死叉卖出时价格已有一定回撤。下图资金曲线对比显示，"
    f"策略累计回报为{metrics1['cum_return']:.2%}，而买入持有回报为{metrics1['market_cum_return']:.2%}。"
    f"在单边大涨行情中，双均线策略的收益通常低于买入持有，因为策略在趋势回调时会卖出离场，"
    f"错失部分涨幅，但也降低了回撤风险。", style_body))
doc_elements.append(Spacer(1, 5 * mm))

# 图2
add_image(doc_elements, chart2_path,
          "图2 中船特气双均线策略(MA5/MA15)回撤分析")

doc_elements.append(Paragraph(
    f"<b>图2解读：</b>回撤曲线展示了策略净值相对历史最高点的下跌幅度。"
    f"中船特气双均线策略的最大回撤为{metrics1['max_drawdown']:.2%}，"
    f"出现在股价大幅波动期间。回撤幅度反映了策略在最坏情况下的亏损风险，"
    f"投资者需确保自己能承受该水平的回撤。"
    f"相比买入持有策略在回调期间可能遭受更大损失，双均线策略通过及时卖出"
    f"在一定程度上控制了回撤幅度。", style_body))
doc_elements.append(Spacer(1, 5 * mm))

doc_elements.append(PageBreak())

# === 四、多股票多周期对比实验 ===
doc_elements.append(Paragraph("四、多股票多周期参数对比实验", style_h1))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph(
    "为探究双均线策略在不同股票和不同参数下的表现差异，本文选取三只具有不同走势特征的股票"
    "（中船特气688146.SH、天地科技600587.SH、平安银行000001.SZ），"
    "分别测试5组均线参数组合（MA5/15、MA5/20、MA10/20、MA5/30、MA10/30），"
    "共15组实验，对比各组合的累计回报、夏普比率和最大回撤。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

# 对比结果表
comp_table_data = [['股票', '参数', '累计回报', '买入持有', '夏普比率', '最大回撤', '胜率']]
for r in all_results:
    comp_table_data.append([
        r['stock'],
        f"MA{r['short_w']}/{r['long_w']}",
        f"{r['metrics']['cum_return']:.2%}",
        f"{r['metrics']['market_cum_return']:.2%}",
        f"{r['metrics']['sharpe_ratio']:.2f}",
        f"{r['metrics']['max_drawdown']:.2%}",
        f"{r['metrics']['win_rate']:.0%}",
    ])

comp_table = Table(comp_table_data, colWidths=[60, 65, 70, 70, 60, 65, 50])
comp_table.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('LEADING', (0, 0), (-1, -1), 13.5),
    ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#34495e')),
    ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#ecf0f1')]),
]))
doc_elements.append(comp_table)
doc_elements.append(Spacer(1, 5 * mm))

# 图3 对比图
add_image(doc_elements, chart3_path,
          "图3 多股票多周期双均线策略效果对比")

doc_elements.append(Paragraph(
    "<b>图3解读：</b>对比图展示了15组实验的累计回报、夏普比率和最大回撤三个维度的对比。"
    "从累计回报来看，中船特气在所有参数组合下均获得正收益，这主要得益于该股票过去一年的大幅上涨趋势；"
    "天地科技的策略表现因参数而异，部分组合获得正收益；"
    "平安银行由于过去一年整体呈下跌趋势，双均线策略在多数参数下出现亏损，但也通过及时卖出控制了损失幅度。"
    "从夏普比率来看，中船特气MA5/15和MA5/20组合表现较好，说明在强趋势行情中较短周期均线更为灵敏有效。"
    "从最大回撤来看，长周期参数（如MA10/30）通常回撤更小，因为交易频率降低，但同时也可能错过部分机会。", style_body))
doc_elements.append(Spacer(1, 5 * mm))

# 图4 天地科技
add_image(doc_elements, chart4_path,
          "图4 天地科技(600587.SH)双均线策略(MA5/MA15)股价与交易信号")

doc_elements.append(Paragraph(
    f"<b>图4解读：</b>天地科技过去一年股价在11.60~18.14元区间波动，呈现震荡走势。"
    f"在震荡行情中，MA5/MA15双均线策略产生了较多买卖信号，但部分信号为虚假信号。"
    f"策略累计回报为{metrics2['cum_return']:.2%}，买入持有回报为{metrics2['market_cum_return']:.2%}。"
    f"这印证了双均线策略在震荡市中表现不佳的特点——频繁的金叉死叉信号导致反复进出，"
    f"交易成本累积且容易被假突破误导。对于震荡型股票，建议延长均线周期或加入其他过滤条件。", style_body))
doc_elements.append(Spacer(1, 5 * mm))

# 图5 平安银行
add_image(doc_elements, chart5_path,
          "图5 平安银行(000001.SZ)双均线策略(MA5/MA15)股价与交易信号")

doc_elements.append(Paragraph(
    f"<b>图5解读：</b>平安银行过去一年股价从约12元下跌至约10.5元，整体呈下跌趋势。"
    f"在下跌行情中，双均线策略通过及时卖出避免了部分损失，策略累计回报为{metrics3['cum_return']:.2%}，"
    f"而买入持有回报为{metrics3['market_cum_return']:.2%}。"
    f"策略在下跌趋势中表现优于买入持有，因为死叉信号帮助投资者及时离场，避免了持续下跌的损失。"
    f"但即便如此，策略仍可能出现亏损，因为反弹时的金叉买入信号可能被随后的下跌再次打脸。"
    f"对于下跌趋势中的股票，双均线策略虽能减损但难以盈利，更合理的做法是暂时回避或做空。", style_body))
doc_elements.append(Spacer(1, 5 * mm))

doc_elements.append(PageBreak())

# === 五、总结与应用心得 ===
doc_elements.append(Paragraph("五、双均线策略应用总结与心得", style_h1))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>1. 适用场景</b>", style_body))
doc_elements.append(Paragraph(
    "通过上述多股票多周期的对比实验，可以总结出双均线策略的适用场景如下："
    "（1）单边趋势行情是双均线策略的最佳应用场景。当中船特气处于强势上涨趋势时，"
    "策略能较好地捕捉上涨段，获得可观的正收益。"
    "（2）下跌趋势中，双均线策略通过及时卖出可以减少损失，表现优于买入持有策略，"
    "但难以获得正收益。"
    "（3）震荡行情中，双均线策略表现最差，频繁的虚假信号导致反复亏损，应尽量避免使用或增加过滤条件。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>2. 参数选择心得</b>", style_body))
doc_elements.append(Paragraph(
    "（1）短周期均线（如MA5）对价格变化更敏感，信号产生更早，但虚假信号也更多；"
    "长周期均线（如MA20、MA30）更稳定，信号更可靠，但滞后性更强。"
    "（2）短长均线周期差距越小（如MA5/MA10），交易越频繁，适合短线交易；"
    "差距越大（如MA10/MA30），交易越少，适合中长线投资。"
    "（3）对于波动性大的股票（如中船特气），建议使用较短周期（MA5/MA15）以快速捕捉趋势；"
    "对于波动性小的股票（如平安银行），建议使用较长周期（MA10/MA30）以减少噪音。"
    "（4）没有 universally 最优的参数组合，投资者应根据个股特性和市场环境灵活调整。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>3. 策略改进建议</b>", style_body))
doc_elements.append(Paragraph(
    "（1）加入成交量过滤：仅在金叉伴随放量时买入，可有效减少虚假信号。"
    "（2）加入趋势强度过滤：通过均线斜率或ADX指标判断趋势强度，仅在强趋势中交易。"
    "（3）多时间框架确认：日线金叉需周线趋势配合，降低假突破概率。"
    "（4）设置止损止盈：在买入后设置止损位（如-5%）和止盈位（如+15%），控制单笔交易风险。"
    "（5）结合其他指标：将双均线与RSI、MACD、布林带等技术指标结合使用，形成多因子共振，提高信号可靠性。"
    "（6）动态参数调整：根据市场波动率自动调整均线周期，高波动时缩短周期，低波动时延长周期。", style_body))
doc_elements.append(Spacer(1, 3 * mm))

doc_elements.append(Paragraph("<b>4. 风险提示</b>", style_body))
doc_elements.append(Paragraph(
    "双均线策略作为趋势跟踪策略，本质上是对历史数据的统计总结，无法保证未来表现。"
    "在实际投资中，投资者应充分认识策略的局限性，结合基本面分析和风险管理，"
    "切勿盲目依赖单一技术指标。同时，回测结果可能存在过度拟合问题，"
    "建议在样本外数据上验证策略稳健性后再投入实盘使用。", style_body))

# 生成PDF
pdf_path = f'{output_dir}/童逸+TASK3.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                        leftMargin=25*mm, rightMargin=25*mm,
                        topMargin=25*mm, bottomMargin=25*mm)
doc.build(doc_elements)
print(f"\nPDF报告已生成: {pdf_path}")

print("\n" + "=" * 60)
print("全部完成！")
print("=" * 60)
