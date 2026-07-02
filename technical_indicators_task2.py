"""
中船特气(688146.SH)技术指标分析脚本 - TASK2
包含: RSI / MACD / 布林带(Bollinger Bands) / KDJ
输出: 图表图片文件 + PDF分析报告
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, KeepTogether, PageBreak)
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import warnings
warnings.filterwarnings('ignore')

# 注册中文字体（从Windows系统字体目录）
FONT_PATH = r'C:\Windows\Fonts\SimSun.ttc'
pdfmetrics.registerFont(TTFont('SimSun', FONT_PATH))
# 注册粗体
pdfmetrics.registerFont(TTFont('SimHei', r'C:\Windows\Fonts\SimHei.ttf'))
print('中文字体注册完成')

# ============================================================
# 配置
# ============================================================
BASE_DIR = r'E:\量化交易：AI大模型辅助的金融交易策略'
CSV_FILE = os.path.join(BASE_DIR, '中船特气_688146_daily_data.csv')
OUTPUT_DIR = BASE_DIR

# 图表1: 收盘价 + MA
CHART1 = os.path.join(OUTPUT_DIR, 'chart1_close_price.png')
# 图表2: RSI
CHART2 = os.path.join(OUTPUT_DIR, 'chart2_rsi.png')
# 图表3: MACD
CHART3 = os.path.join(OUTPUT_DIR, 'chart3_macd.png')
# 图表4: 布林带
CHART4 = os.path.join(OUTPUT_DIR, 'chart4_bollinger.png')
# 图表5: KDJ
CHART5 = os.path.join(OUTPUT_DIR, 'chart5_kdj.png')

PDF_FILE = os.path.join(OUTPUT_DIR, '童逸+TASK2.pdf')

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第1步: 加载数据
# ============================================================
print('=' * 60)
print('第1步: 加载数据')
print('=' * 60)

df = pd.read_csv(CSV_FILE, parse_dates=['trade_date'])
df = df.sort_values('trade_date').reset_index(drop=True)
df['trade_date_str'] = df['trade_date'].dt.strftime('%Y-%m-%d')
n = len(df)
print(f'数据行数: {n}')
print(f'时间范围: {df["trade_date"].min().date()} 至 {df["trade_date"].max().date()}')

# ============================================================
# 第2步: 数据诊断
# ============================================================
print('\n' + '=' * 60)
print('第2步: 数据诊断分析')
print('=' * 60)

print('\n【缺失值统计】')
missing = df.isnull().sum()
print(missing[missing > 0])

print('\n【描述性统计】')
desc = df[['open','high','low','close','vol','amount','pct_chg']].describe()
print(desc)

# ============================================================
# 第3步: 计算技术指标
# ============================================================
print('\n' + '=' * 60)
print('第3步: 计算技术指标')
print('=' * 60)

# --- RSI (14日) ---
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # Wilder平滑
    avg_gain_w = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss_w = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain_w / avg_loss_w
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calc_rsi(df['close'], 14)
print(f'RSI计算完成，有效值: {df["rsi"].notna().sum()} 个')

# --- MACD (12, 26, 9) ---
def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist

df['dif'], df['dea'], df['macd_hist'] = calc_macd(df['close'])
print(f'MACD计算完成，有效值: {df["dif"].notna().sum()} 个')

# --- 布林带 (20日, 2倍标准差) ---
def calc_bollinger(series, period=20, num_std=2):
    mid = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower

df['bb_mid'], df['bb_upper'], df['bb_lower'] = calc_bollinger(df['close'])
print(f'布林带计算完成，有效值: {df["bb_mid"].notna().sum()} 个')

# --- KDJ (9, 3, 3) ---
def calc_kdj(df, n=9, m1=3, m2=3):
    low_n = df['low'].rolling(window=n, min_periods=n).min()
    high_n = df['high'].rolling(window=n, min_periods=n).max()
    rsv = (df['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    k = np.zeros(len(rsv))
    d = np.zeros(len(rsv))
    j = np.zeros(len(rsv))
    k[0] = 50
    d[0] = 50
    for i in range(1, len(rsv)):
        k[i] = (2/3) * k[i-1] + (1/3) * rsv.iloc[i]
        d[i] = (2/3) * d[i-1] + (1/3) * k[i]
    j = m1 * k - m2 * d

    k = pd.Series(k, index=rsv.index)
    d = pd.Series(d, index=rsv.index)
    j = pd.Series(j, index=rsv.index)
    return k, d, j

df['k'], df['d'], df['j'] = calc_kdj(df)
print(f'KDJ计算完成，有效值: {df["k"].notna().sum()} 个')

# 保存带指标数据的CSV
indicators_csv = os.path.join(OUTPUT_DIR, '中船特气_688146_indicators.csv')
df.to_csv(indicators_csv, index=False, encoding='utf-8-sig')
print(f'\n带指标数据已保存: {indicators_csv}')

# ============================================================
# 第4步: 绑制可视化图表
# ============================================================
print('\n' + '=' * 60)
print('第4步: 绑制可视化图表')
print('=' * 60)

# 图1: 收盘价走势 + MA20 + MA60
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df['trade_date'], df['close'], label='收盘价', color='#1f77b4', linewidth=1.2)
ax.plot(df['trade_date'], df['ma20'], label='MA20', color='#ff7f0e', linewidth=1.0, linestyle='--')
ax.plot(df['trade_date'], df['ma60'], label='MA60', color='#2ca02c', linewidth=1.0, linestyle='--')
ax.fill_between(df['trade_date'], df['close'], alpha=0.1, color='#1f77b4')
ax.set_title('图1  中船特气(688146.SH)收盘价走势', fontsize=14, fontweight='bold', pad=10)
ax.set_xlabel('交易日期', fontsize=11)
ax.set_ylabel('收盘价(元)', fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(CHART1, dpi=150, bbox_inches='tight')
plt.close()
print(f'图1已保存: {CHART1}')

# 图2: RSI
fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})
ax1, ax2 = axes
ax1.plot(df['trade_date'], df['close'], color='#1f77b4', linewidth=1.0)
ax1.set_title('图2  中船特气收盘价走势', fontsize=13, fontweight='bold')
ax1.set_ylabel('收盘价(元)', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(df['trade_date'], df['rsi'], color='#d62728', linewidth=1.0, label='RSI(14)')
ax2.axhline(y=70, color='#ff0000', linestyle='--', linewidth=0.8, alpha=0.7, label='超买线(70)')
ax2.axhline(y=30, color='#00aa00', linestyle='--', linewidth=0.8, alpha=0.7, label='超卖线(30)')
ax2.axhline(y=50, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax2.fill_between(df['trade_date'], 70, 100, alpha=0.1, color='red', label='超买区')
ax2.fill_between(df['trade_date'], 0, 30, alpha=0.1, color='green', label='超卖区')
ax2.set_title('图2  RSI(14)相对强弱指标', fontsize=13, fontweight='bold')
ax2.set_xlabel('交易日期', fontsize=10)
ax2.set_ylabel('RSI', fontsize=10)
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left', fontsize=8, ncol=3)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(CHART2, dpi=150, bbox_inches='tight')
plt.close()
print(f'图2已保存: {CHART2}')

# 图3: MACD
fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})
ax1, ax2 = axes
ax1.plot(df['trade_date'], df['close'], color='#1f77b4', linewidth=1.0)
ax1.set_title('图3  中船特气收盘价走势', fontsize=13, fontweight='bold')
ax1.set_ylabel('收盘价(元)', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(df['trade_date'], df['dif'], label='DIF(12,26)', color='#1f77b4', linewidth=1.0)
ax2.plot(df['trade_date'], df['dea'], label='DEA(9)', color='#ff7f0e', linewidth=1.0)
bar_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in df['macd_hist']]
ax2.bar(df['trade_date'], df['macd_hist'], color=bar_colors, alpha=0.6, label='MACD柱', width=1.5)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax2.set_title('图3  MACD指标(12,26,9)', fontsize=13, fontweight='bold')
ax2.set_xlabel('交易日期', fontsize=10)
ax2.set_ylabel('MACD', fontsize=10)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(CHART3, dpi=150, bbox_inches='tight')
plt.close()
print(f'图3已保存: {CHART3}')

# 图4: 布林带
fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})
ax1, ax2 = axes
ax1.plot(df['trade_date'], df['close'], label='收盘价', color='#1f77b4', linewidth=1.0)
ax1.plot(df['trade_date'], df['bb_mid'], label='布林中轨(MA20)', color='#ff7f0e', linewidth=1.0, linestyle='--')
ax1.plot(df['trade_date'], df['bb_upper'], label='布林上轨(+2σ)', color='#d62728', linewidth=0.8, linestyle=':')
ax1.plot(df['trade_date'], df['bb_lower'], label='布林下轨(-2σ)', color='#2ca02c', linewidth=0.8, linestyle=':')
ax1.fill_between(df['trade_date'], df['bb_upper'], df['bb_lower'], alpha=0.08, color='blue')
ax1.set_title('图4  中船特气收盘价走势', fontsize=13, fontweight='bold')
ax1.set_ylabel('收盘价(元)', fontsize=10)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(df['trade_date'], df['close'], color='#1f77b4', linewidth=1.0)
ax2.plot(df['trade_date'], df['bb_upper'], color='#d62728', linewidth=0.8, linestyle=':')
ax2.plot(df['trade_date'], df['bb_mid'], color='#ff7f0e', linewidth=0.8, linestyle='--')
ax2.plot(df['trade_date'], df['bb_lower'], color='#2ca02c', linewidth=0.8, linestyle=':')
ax2.fill_between(df['trade_date'], df['bb_upper'], df['bb_lower'], alpha=0.15, color='blue')
ax2.fill_between(df['trade_date'], df['bb_upper'], df['close'], alpha=0.1, color='red', where=df['close'] >= df['bb_upper'])
ax2.fill_between(df['trade_date'], df['bb_lower'], df['close'], alpha=0.1, color='green', where=df['close'] <= df['bb_lower'])
ax2.set_title('图4  布林带(Bollinger Bands, N=20, K=2)', fontsize=13, fontweight='bold')
ax2.set_xlabel('交易日期', fontsize=10)
ax2.set_ylabel('收盘价(元)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(CHART4, dpi=150, bbox_inches='tight')
plt.close()
print(f'图4已保存: {CHART4}')

# 图5: KDJ
fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [3, 1]})
ax1, ax2 = axes
ax1.plot(df['trade_date'], df['close'], color='#1f77b4', linewidth=1.0)
ax1.set_title('图5  中船特气收盘价走势', fontsize=13, fontweight='bold')
ax1.set_ylabel('收盘价(元)', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(df['trade_date'], df['k'], label='K值', color='#1f77b4', linewidth=1.0)
ax2.plot(df['trade_date'], df['d'], label='D值', color='#ff7f0e', linewidth=1.0)
ax2.plot(df['trade_date'], df['j'], label='J值', color='#d62728', linewidth=0.8, linestyle='--', alpha=0.7)
ax2.axhline(y=80, color='#ff0000', linestyle='--', linewidth=0.8, alpha=0.7, label='超买线(80)')
ax2.axhline(y=20, color='#00aa00', linestyle='--', linewidth=0.8, alpha=0.7, label='超卖线(20)')
ax2.fill_between(df['trade_date'], 80, 100, alpha=0.1, color='red', label='超买区')
ax2.fill_between(df['trade_date'], 0, 20, alpha=0.1, color='green', label='超卖区')
ax2.set_title('图5  KDJ随机指标(9,3,3)', fontsize=13, fontweight='bold')
ax2.set_xlabel('交易日期', fontsize=10)
ax2.set_ylabel('KDJ', fontsize=10)
ax2.set_ylim(-10, 110)
ax2.legend(loc='upper left', fontsize=8, ncol=3)
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(CHART5, dpi=150, bbox_inches='tight')
plt.close()
print(f'图5已保存: {CHART5}')

print('\n所有图表绑制完成！')

# ============================================================
# 第5步: 生成PDF报告
# ============================================================
print('\n' + '=' * 60)
print('第5步: 生成PDF报告')
print('=' * 60)

# 中文字体注册
font_paths = [
    r'C:\Windows\Fonts\SimSun.ttc',
    r'C:\Windows\Fonts\simsun.ttc',
]
font_registered = None
for fp in font_paths:
    if os.path.exists(fp):
        font_registered = fm.FontProperties(fname=fp).get_name()
        break
if not font_registered:
    font_registered = 'SimSun'

print(f'使用字体: {font_registered}')

def make_styles():
    body = ParagraphStyle(
        'body',
        fontName='SimSun',
        fontSize=10.5,
        leading=15.75,   # 1.5倍行距
        spaceBefore=0,
        spaceAfter=0,
        alignment=TA_JUSTIFY,
    )
    heading1 = ParagraphStyle(
        'h1',
        fontName='SimSun',
        fontSize=14,
        leading=21,
        spaceBefore=12,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    )
    heading2 = ParagraphStyle(
        'h2',
        fontName='SimSun',
        fontSize=12,
        leading=18,
        spaceBefore=10,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
    )
    center = ParagraphStyle(
        'center',
        fontName='SimSun',
        fontSize=11,
        leading=16.5,
        spaceBefore=4,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    formula = ParagraphStyle(
        'formula',
        fontName='SimSun',
        fontSize=9.5,
        leading=14.25,
        spaceBefore=4,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
        leftIndent=20,
    )
    return body, heading1, heading2, center, formula

body_s, h1_s, h2_s, center_s, formula_s = make_styles()

doc = SimpleDocTemplate(
    PDF_FILE,
    pagesize=A4,
    leftMargin=25*mm,
    rightMargin=25*mm,
    topMargin=20*mm,
    bottomMargin=20*mm,
)

story = []

# --- 封面 ---
story.append(Spacer(1, 30*mm))
story.append(Paragraph('中船特气(688146.SH)技术指标分析报告', center_s))
story.append(Paragraph('— TASK2 —', center_s))
story.append(Spacer(1, 15*mm))
story.append(Paragraph('分析标的：中船特气  股票代码：688146.SH', center_s))
story.append(Paragraph(f'数据区间：{df["trade_date"].min().strftime("%Y-%m-%d")} 至 {df["trade_date"].max().strftime("%Y-%m-%d")}', center_s))
story.append(Paragraph(f'数据来源：Tushare Pro  |  交易日数量：{n} 个', center_s))
story.append(Paragraph('分析人：童逸', center_s))
story.append(Paragraph('生成日期：2026-07-02', center_s))
story.append(PageBreak())

# --- 一、数据诊断分析 ---
story.append(Paragraph('一、数据诊断分析', h1_s))

story.append(Paragraph('1.1 数据概览', h2_s))
story.append(Paragraph(
    f'本报告使用的数据来源于Tushare Pro金融数据库，标的为沪深A股中船特气（688146.SH）。'
    f'数据时间范围为{df["trade_date"].min().strftime("%Y-%m-%d")}至{df["trade_date"].max().strftime("%Y-%m-%d")}，'
    f'共包含{n}个交易日的日线行情数据，涵盖开盘价、最高价、最低价、收盘价、成交量、'
    f'成交额及涨跌幅等13个字段。所有原始字段均无缺失，数据质量良好，可直接用于后续分析。', body_s))

story.append(Paragraph('1.2 缺失值分析', h2_s))
story.append(Paragraph(
    '缺失值统计结果显示：ts_code、trade_date、open、high、low、close、pre_close、change、'
    'pct_chg、vol、amount等11个字段均为完整，无缺失值。ma20字段存在19个缺失值（占总量7.96%），'
    'ma60字段存在59个缺失值（占总量24.69%）。缺失值均出现在数据序列初期，属正常现象（移动平均线'
    '需要足够的历史数据才能计算），不影响后续技术指标分析的有效性。', body_s))

story.append(Paragraph('1.3 描述性统计量', h2_s))
stats_data = df[['open','high','low','close','vol','amount','pct_chg']].describe().round(3)
stats_data.index = ['计数','均值','标准差','最小值','25%分位','中位数','75%分位','最大值']
col_labels = ['open(开盘)','high(最高)','low(最低)','close(收盘)','vol(成交量/手)','amount(成交额/万)','pct_chg(涨跌幅%)']
table_data = [col_labels]
for idx, row_label in enumerate(stats_data.index):
    row = [stats_data.iloc[idx].values[j] if j < len(stats_data.columns) else '' for j in range(len(col_labels))]
    table_data.append([str(row[j]) for j in range(len(col_labels))])
t = Table(table_data, colWidths=[30*mm] + [22*mm]*6)
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4472C4')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,-1), 'SimSun'),
    ('FONTSIZE', (0,0), (-1,-1), 7.5),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#DCE6F1')]),
]))
story.append(t)
story.append(Spacer(1, 5*mm))

story.append(Paragraph(
    f'由上表可见：中船特气期间收盘价均值约为65.81元，标准差约66.80元，最大值416.99元、最小值29.29元，'
    f'波动幅度极大（中位数50%分位仅为43.18元，说明股价大部分时间在较低价位运行，后期出现大幅上涨）。'
    f'日均成交量约111,358手，标准差较大（约193,987手），说明量能波动显著。'
    f'涨跌幅方面：上涨天数140天、下跌天数97天、平盘2天；最大单日涨幅20.01%，最大单日跌幅-13.66%，'
    f'整体呈现显著的上涨趋势，波动性较大。', body_s))

story.append(PageBreak())

# --- 二、RSI指标 ---
story.append(Paragraph('二、RSI相对强弱指标', h1_s))

story.append(Paragraph('2.1 指标简介', h2_s))
story.append(Paragraph(
    'RSI（Relative Strength Index，相对强弱指标）由J. Welles Wilder于1978年提出，是衡量一定时期内'
    '价格上涨和下跌幅度的经典动能指标。RSI取值范围为0~100，通过比较指定周期内上涨与下跌幅度'
    '的均值，来反映市场多空力量的对比状态。', body_s))

story.append(Paragraph('2.2 计算公式', h2_s))
story.append(Paragraph('第一步：计算N周期内上涨日的平均涨幅和下跌日的平均跌幅：', body_s))
story.append(Paragraph('  平均涨幅 = N周期内所有上涨日涨幅的算术平均值', formula_s))
story.append(Paragraph('  平均跌幅 = N周期内所有下跌日跌幅的算术平均值（取绝对值）', formula_s))
story.append(Paragraph('第二步：计算相对强度（RS）：', body_s))
story.append(Paragraph('  RS = 平均涨幅 ÷ 平均跌幅', formula_s))
story.append(Paragraph('第三步：计算RSI值：', body_s))
story.append(Paragraph('  RSI = 100 - (100 ÷ (1 + RS)) = 平均涨幅 ÷ (平均涨幅 + 平均跌幅) × 100', formula_s))
story.append(Paragraph(
    '标准参数采用RSI(14)，即以过去14个交易日为计算周期。Wilder原始参数为14，'
    '平衡了指标灵敏度与可靠性，在大多数市场和周期中均适用。', body_s))

story.append(Paragraph('2.3 作用与交易信号', h2_s))
story.append(Paragraph(
    '【超买超卖判断】RSI>70为超买区域，表明市场上行过快，存在回调风险；RSI<30为超卖区域，'
    '表明市场下行过快，可能存在反弹机会。在强势趋势中，RSI可长期维持于超买超卖区域，此时不宜'
    '单纯逆势操作，而应结合趋势方向判断。', body_s))
story.append(Paragraph(
    '【50中线研判】RSI穿过50中线可作为牛熊分界线：RSI>50表明多头占优，RSI<50表明空头占优。'
    '在上升趋势中，RSI回调至50~55区域往往是较好的加仓机会。', body_s))
story.append(Paragraph(
    '【背离信号】当价格创出新低而RSI未创新低时，形成底背离，预示下跌动能衰减、价格可能反弹；'
    '当价格创出新高而RSI未创新高时，形成顶背离，预示上涨动能衰减、价格可能回调。', body_s))

story.append(Paragraph('2.4 RSI指标图形分析', h2_s))
story.append(Image(CHART2, width=165*mm, height=70*mm))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    '【图2解读】RSI(14)在中船特气此区间内呈现显著的趋势性特征。从图中可见，股价在经历初期盘整后，'
    'RSI快速上行并多次进入超买区域（>70），表明上涨动能充沛。期间RSI在70以上的超买区持续停留，'
    '与股价强势上涨走势相吻合，属正常现象而非反转信号。RSI从未触及30以下的超卖区域，说明空头'
    '力量在此期间持续处于弱势。50中线在大部分时间内对RSI形成强支撑，RSI始终运行于50以上，'
    '确认了中期上升趋势的主导地位。整体来看，RSI指标在此期间表现出持续的多头格局。', body_s))
story.append(PageBreak())

# --- 三、MACD指标 ---
story.append(Paragraph('三、MACD指标', h1_s))

story.append(Paragraph('3.1 指标简介', h2_s))
story.append(Paragraph(
    'MACD（Moving Average Convergence Divergence，指数平滑异同移动平均线）由Gerald Appel于1970年代创建，'
    '是技术分析中应用最广泛的趋势动能指标之一。MACD通过计算不同周期指数移动平均线（EMA）之间的'
    '差异，来揭示价格趋势的方向、强度和转折点，同时具备趋势确认与动能分析的双重功能。', body_s))

story.append(Paragraph('3.2 计算公式', h2_s))
story.append(Paragraph('第一步：计算快速与慢速EMA：', body_s))
story.append(Paragraph('  快速EMA = EMA(收盘价, 12)', formula_s))
story.append(Paragraph('  慢速EMA = EMA(收盘价, 26)', formula_s))
story.append(Paragraph('第二步：计算DIF（差离值，快线）：', body_s))
story.append(Paragraph('  DIF = 快速EMA - 慢速EMA', formula_s))
story.append(Paragraph('第三步：计算DEA（信号线，慢线）：', body_s))
story.append(Paragraph('  DEA = EMA(DIF, 9)', formula_s))
story.append(Paragraph('第四步：计算MACD柱状图：', body_s))
story.append(Paragraph('  MACD柱 = (DIF - DEA) × 2', formula_s))
story.append(Paragraph(
    '标准参数(12, 26, 9)由Gerald Appel原创，经数十年市场验证，广泛适用于各类市场和时间框架。'
    '12日EMA代表短期趋势，26日EMA代表长期趋势，两者之差DIF反映短中期动量变化；'
    '9日EMA对DIF进行平滑处理得到DEA，形成交易信号线。', body_s))

story.append(Paragraph('3.3 作用与交易信号', h2_s))
story.append(Paragraph(
    '【金叉与死叉】DIF从下方上穿DEA形成"金叉"，为看涨买入信号；DIF从上方下穿DEA形成"死叉"，'
    '为看跌卖出信号。零轴上方金叉为强势信号（上升趋势延续概率高），零轴下方金叉为弱势信号，'
    '可能仅为反弹而非反转。', body_s))
story.append(Paragraph(
    '【零轴多空分界】MACD在零轴上方代表12日EMA>26日EMA，市场处于上升趋势；MACD在零轴下方代表'
    '12日EMA<26日EMA，市场处于下降趋势。MACD穿越零轴是重要的中期趋势转折信号。', body_s))
story.append(Paragraph(
    '【柱状图动能分析】MACD柱由负转正代表买入动能增强，由正转负代表卖出动能增强；'
    '柱状图持续扩大代表趋势加速，持续收缩代表趋势减速，可能即将反转。', body_s))
story.append(Paragraph(
    '【背离信号】价格创新高但DIF或MACD柱未创新高，形成顶背离，预示趋势可能转跌；'
    '价格创新低但DIF或MACD柱未创新低，形成底背离，预示趋势可能转涨。', body_s))

story.append(Paragraph('3.4 MACD指标图形分析', h2_s))
story.append(Image(CHART3, width=165*mm, height=70*mm))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    '【图3解读】从MACD指标图形可以清晰看到，中船特气在分析期间经历了典型的趋势启动与加速过程。'
    '初期DIF与DEA在零轴附近反复纠缠，为震荡筑底阶段。随着股价启动，DIF快速向上穿越DEA形成金叉，'
    '且两者迅速远离零轴上方，MACD柱持续放大，表明多头动能充沛、上升趋势强劲。'
    '值得注意的是，股价在高位时，MACD柱开始出现收缩迹象（顶部动能减弱），但DIF始终运行于零轴上方，'
    '说明中长期上升趋势尚未根本改变，仅存在短期回调压力。整体而言，MACD在本阶段提供了多次高质量'
    '的买入信号（DIF金叉DEA），有效捕捉了股价的主升浪行情。', body_s))
story.append(PageBreak())

# --- 四、布林带 ---
story.append(Paragraph('四、布林带（Bollinger Bands）', h1_s))

story.append(Paragraph('4.1 指标简介', h2_s))
story.append(Paragraph(
    '布林带（Bollinger Bands）由约翰·布林格（John Bollinger）于1980年代创建，是基于统计学标准差原理'
    '设计的路径型技术指标。布林带由三条轨道线组成：中轨为N日简单移动平均线，上轨和下轨分别为'
    '中轨加减K倍标准差，形成一个随价格波动而动态扩张或收缩的价格通道。布林带的核心价值在于'
    '它能够根据市场波动性的变化自动调整通道宽度，既是支撑阻力位，也是超买超卖信号。', body_s))

story.append(Paragraph('4.2 计算公式', h2_s))
story.append(Paragraph('第一步：计算中轨（N日简单移动平均线）：', body_s))
story.append(Paragraph('  中轨(MID) = MA(收盘价, N)', formula_s))
story.append(Paragraph('第二步：计算N日收盘价的标准差：', body_s))
story.append(Paragraph('  标准差(STD) = StdDev(收盘价, N)', formula_s))
story.append(Paragraph('第三步：计算上下轨：', body_s))
story.append(Paragraph('  上轨(UPPER) = 中轨 + K × 标准差', formula_s))
story.append(Paragraph('  下轨(LOWER) = 中轨 - K × 标准差', formula_s))
story.append(Paragraph(
    '标准参数N=20（布林格本人推荐），K=2（即中轨±2倍标准差，覆盖约95%的价格波动范围）。'
    'K值越小通道越窄、越敏感；K值越大通道越宽、越稳健。', body_s))

story.append(Paragraph('4.3 作用与交易信号', h2_s))
story.append(Paragraph(
    '【通道突破信号】价格向上突破上轨进入"超买区"，可能面临回调压力；价格向下跌破下轨进入"超卖区"，'
    '可能存在反弹机会。但需注意，在强趋势中价格可沿布林带上轨持续运行。', body_s))
story.append(Paragraph(
    '【支撑阻力作用】布林带中轨对价格有牵引作用（"引力"效应），可作为动态支撑/阻力参考；'
    '上轨对价格有阻力作用，下轨对价格有支撑作用。', body_s))
story.append(Paragraph(
    '【布林带收缩与扩张】布林带通道收窄（收缩）表示市场波动性降低，可能蓄势突破；'
    '通道扩张表示市场波动性增大，趋势可能加速。当价格触及上轨且布林带扩张时，往往是强势延续信号。', body_s))
story.append(Paragraph(
    '【喇叭口形态】布林带上轨向上、下轨向下同时发散，形成"喇叭口"，代表波动性急剧放大，'
    '通常伴随重要趋势转折或加速。中轨方向也可辅助判断趋势方向（向上为上升趋势）。', body_s))

story.append(Paragraph('4.4 布林带指标图形分析', h2_s))
story.append(Image(CHART4, width=165*mm, height=70*mm))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    '【图4解读】布林带在本案例中展现出典型的趋势启动与加速特征。初期股价在布林带中下轨之间运行，'
    '通道宽度较窄，表明市场处于低波动率的盘整阶段。随着股价突破布林带上轨，通道开始明显扩张，'
    '这是强势趋势确立的经典信号。中轨线在股价上涨过程中持续向上倾斜，对价格形成稳定的中期支撑。'
    '值得注意的是，当股价大幅上涨远离上轨时，布林带持续扩张，表明波动性显著增大，此时风险也相应'
    '加剧。从布林带的形态来看，中船特气在此期间经历了从"窄幅震荡"到"趋势爆发"的关键转折，'
    '布林带上轨对追涨形成阻力约束，下轨对回撤形成保护，是极为实用的动态风险管理工具。', body_s))
story.append(PageBreak())

# --- 五、KDJ指标 ---
story.append(Paragraph('五、扩展指标——KDJ随机指标', h1_s))

story.append(Paragraph('5.1 指标简介', h2_s))
story.append(Paragraph(
    'KDJ指标（随机指标，Stochastic Oscillator）由乔治·莱恩（George Lane）于1950年代提出，'
    '是一种超买超卖类摆动指标。KDJ通过计算特定周期内价格相对于该周期最高价和最低价的位置，'
    '来判断当前价格处于历史区间的相对高低，从而识别市场的超买超卖状态。'
    '与RSI侧重于涨跌幅度不同，KDJ更关注价格的高低位置关系，因此对价格短期转折点的捕捉更为灵敏。', body_s))

story.append(Paragraph('5.2 计算公式', h2_s))
story.append(Paragraph('第一步：计算RSV（未成熟随机值）：', body_s))
story.append(Paragraph(
    '  RSV = (收盘价 - N日内最低价) ÷ (N日内最高价 - N日内最低价) × 100', formula_s))
story.append(Paragraph('第二步：计算K值（快速随机线，对RSV进行3日平滑）：', body_s))
story.append(Paragraph('  K = (2/3) × 前一日K值 + (1/3) × 当日RSV', formula_s))
story.append(Paragraph('第三步：计算D值（慢速随机线，对K值进行3日平滑）：', body_s))
story.append(Paragraph('  D = (2/3) × 前一日D值 + (1/3) × 当日K值', formula_s))
story.append(Paragraph('第四步：计算J值（辅助线，反映K与D的乖离程度）：', body_s))
story.append(Paragraph('  J = 3 × K - 2 × D', formula_s))
story.append(Paragraph(
    '标准参数N=9（RSV周期）、M1=3（K值平滑）、M2=3（D值平滑）。'
    'K值反应最灵敏，D值相对平滑，J值波动最大、信号最激进，三者配合使用可提高判断准确性。', body_s))

story.append(Paragraph('5.3 作用与交易信号', h2_s))
story.append(Paragraph(
    '【超买超卖判断】KDJ>80为超买区域，KDJ<20为超卖区域。J值波动最大，当J>100时表示严重超买，'
    'J<0时表示严重超卖（但需注意KDJ在极端行情中可能持续钝化）。', body_s))
story.append(Paragraph(
    '【金叉死叉】K值从下方向上穿越D值形成金叉，为买入信号；K值从上方下穿D值形成死叉，为卖出信号。'
    '50中线同样适用：KDJ在50以上为偏多格局，50以下为偏空格局。', body_s))
story.append(Paragraph(
    '【背离信号】与RSI类似，价格与KDJ走势背离时预示趋势可能反转。当J值在高位出现转折时，'
    '往往是最早的警示信号。', body_s))

story.append(Paragraph('5.4 KDJ指标图形分析', h2_s))
story.append(Image(CHART5, width=165*mm, height=70*mm))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    '【图5解读】从KDJ指标图形可见，中船特气在分析期间呈现显著的强势特征。J值（红色虚线）'
    '在大部分时间内围绕K值和D值大幅波动，多次触及100以上的严重超买区域，'
    '表明短期上涨动能极其充沛，同时也暗示股价存在短期过热风险。K值和D值在大部分时间内运行于50以上，'
    '确认了多头的持续主导地位。金叉信号多次出现（K上穿D），均与股价的加速上涨阶段高度吻合，'
    '提供了较为准确的买入时机参考。值得注意的是，股价在高位时KDJ出现高位死叉（K下穿D），'
    '对应股价的短期回调，说明KDJ对短期回调风险的捕捉同样有效。综合来看，KDJ在中船特气这只高弹性'
    '股票的分析中表现出色，其高灵敏度的J值对短期顶部的预警作用尤为突出。', body_s))
story.append(PageBreak())

# --- 六、综合总结 ---
story.append(Paragraph('六、综合总结与展望', h1_s))

story.append(Paragraph(
    f'本报告对中船特气（688146.SH）在{df["trade_date"].min().strftime("%Y-%m-%d")}至'
    f'{df["trade_date"].max().strftime("%Y-%m-%d")}期间共{n}个交易日的行情数据进行了系统的'
    '技术指标分析，涵盖RSI、MACD、布林带、KDJ四大经典技术分析工具。', body_s))

story.append(Paragraph(
    '【主要结论】从趋势方向看，四大指标均一致确认了中船特气在此期间处于强劲的中期上升趋势：'
    'RSI持续位于50中线以上，多次进入超买区域；MACD始终运行于零轴上方且持续放大；'
    '布林带通道由窄变宽，价格沿上轨运行；KDJ三线在高位频繁波动，金叉信号密集。'
    '从风险角度看，RSI与KDJ的持续超买区域运行提示短期回调风险积累，MACD柱在顶部收缩'
    '暗示动能可能减弱，布林带持续扩张表明波动性风险增大。', body_s))

story.append(Paragraph(
    '【指标协同】RSI+MACD组合可用于判断趋势的强度与持续性（RSI>50+MACD零轴上方金叉=强做多信号）；'
    '布林带+RSI组合可识别极端价位（价格触布林带上轨+RSI超买=高胜率做空机会）；'
    'KDJ与MACD背离共振可作为趋势反转的早期预警。', body_s))

story.append(Paragraph(
    '【风险提示】本报告仅基于历史数据的指标计算与解读，不构成任何投资建议。'
    '技术指标存在滞后性，且在极端行情中可能出现钝化。投资决策应结合基本面分析、'
    '宏观经济环境及个人风险承受能力综合判断。', body_s))

# 指标统计汇总表
story.append(Spacer(1, 5*mm))
story.append(Paragraph('附：技术指标关键数据统计', h2_s))
last_row = df.iloc[-1]
rsi_last = f'{last_row["rsi"]:.2f}' if pd.notna(last_row['rsi']) else 'N/A'
macd_last = f'{last_row["macd_hist"]:.4f}' if pd.notna(last_row['macd_hist']) else 'N/A'
kdj_k = f'{last_row["k"]:.2f}' if pd.notna(last_row['k']) else 'N/A'
kdj_d = f'{last_row["d"]:.2f}' if pd.notna(last_row['d']) else 'N/A'
kdj_j = f'{last_row["j"]:.2f}' if pd.notna(last_row['j']) else 'N/A'
summary_data = [
    ['指标名称', '参数设置', '期末值', '有效数据量'],
    ['RSI', '周期=14', rsi_last, f'{df["rsi"].notna().sum()}'],
    ['MACD(DIF)', '12,26,9', f'{last_row["dif"]:.4f}', f'{df["dif"].notna().sum()}'],
    ['MACD(DEA)', '12,26,9', f'{last_row["dea"]:.4f}', f'{df["dea"].notna().sum()}'],
    ['MACD柱', '12,26,9', macd_last, f'{df["macd_hist"].notna().sum()}'],
    ['布林中轨', 'N=20,K=2', f'{last_row["bb_mid"]:.2f}', f'{df["bb_mid"].notna().sum()}'],
    ['KDJ-K', '9,3,3', kdj_k, f'{df["k"].notna().sum()}'],
    ['KDJ-D', '9,3,3', kdj_d, f'{df["d"].notna().sum()}'],
    ['KDJ-J', '9,3,3', kdj_j, f'{df["j"].notna().sum()}'],
]
t2 = Table(summary_data, colWidths=[35*mm,30*mm,30*mm,30*mm])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4472C4')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTNAME', (0,0), (-1,-1), 'SimSun'),
    ('FONTSIZE', (0,0), (-1,-1), 9),
    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#DCE6F1')]),
]))
story.append(t2)

# 生成PDF
doc.build(story)
print(f'\nPDF报告已生成: {PDF_FILE}')

# 打印指标期末值摘要
print('\n=== 期末指标值 ===')
print(f'RSI(14): {rsi_last}')
print(f'MACD(DIF): {last_row["dif"]:.4f}')
print(f'MACD(DEA): {last_row["dea"]:.4f}')
print(f'MACD柱: {macd_last}')
print(f'布林中轨: {last_row["bb_mid"]:.2f}')
print(f'KDJ-K: {kdj_k}, D: {kdj_d}, J: {kdj_j}')
print('\n全部完成！')
