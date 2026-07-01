"""
中船特气(688146.SH)过去1年交易日数据获取、绘图、CSV保存与PDF报告生成
使用 Tushare Pro API
"""

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import numpy as np

# ===== 配置 =====
TUSHARE_TOKEN = '5907c6e4dc666a6920da3c31435ea985428e8ed5a6c9a70681e3fcb0'
STOCK_CODE = '688146'
STOCK_NAME = '中船特气'
TS_CODE = f'{STOCK_CODE}.SH'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置 Tushare token
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# ===== 1) 获取过去1年交易日数据 =====
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

print(f"正在获取{STOCK_NAME}({TS_CODE})从 {start_date} 到 {end_date} 的日线数据...")

df = pro.daily(ts_code=TS_CODE, start_date=start_date, end_date=end_date)

if df.empty:
    print("未获取到数据，请检查股票代码或网络连接")
    exit(1)

# 按日期升序排列
df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

# 计算均线
if len(df) >= 20:
    df['ma20'] = df['close'].rolling(window=20).mean()
if len(df) >= 60:
    df['ma60'] = df['close'].rolling(window=60).mean()

# 计算日涨跌幅（pct_chg 字段已包含，确认一下）
print(f"成功获取 {len(df)} 条交易日数据")
print(f"日期范围: {df['trade_date'].min().strftime('%Y-%m-%d')} ~ {df['trade_date'].max().strftime('%Y-%m-%d')}")
print(f"数据字段: {list(df.columns)}")
print(f"\n前5条数据预览:")
print(df.head())

# ===== 2) 绘制统计图表 =====

# --- 图1: 收盘价走势图（含均线）---
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(df['trade_date'], df['close'], color='#D43030', linewidth=1.8, label='收盘价')
if 'ma20' in df.columns:
    ax.plot(df['trade_date'], df['ma20'], color='#FFB347', linewidth=1.2, linestyle='--', label='20日均线')
if 'ma60' in df.columns:
    ax.plot(df['trade_date'], df['ma60'], color='#4169E1', linewidth=1.2, linestyle='--', label='60日均线')

# 标注最高最低收盘价
max_close_idx = df['close'].idxmax()
min_close_idx = df['close'].idxmin()
ax.annotate(f'最高: {df.loc[max_close_idx, "close"]:.2f}',
            xy=(df.loc[max_close_idx, 'trade_date'], df.loc[max_close_idx, 'close']),
            xytext=(10, 15), textcoords='offset points',
            fontsize=10, color='#D43030', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#D43030'))
ax.annotate(f'最低: {df.loc[min_close_idx, "close"]:.2f}',
            xy=(df.loc[min_close_idx, 'trade_date'], df.loc[min_close_idx, 'close']),
            xytext=(10, -20), textcoords='offset points',
            fontsize=10, color='#00AA00', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#00AA00'))

ax.set_title(f'图1  {STOCK_NAME}({TS_CODE}) 过去1年收盘价走势', fontsize=16, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('收盘价(元)', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

latest_close = df.iloc[-1]['close']
first_close = df.iloc[0]['close']
change_pct = (latest_close - first_close) / first_close * 100
color = '#D43030' if change_pct > 0 else '#00AA00'
sign = '+' if change_pct > 0 else ''
ax.text(0.98, 0.02, f'区间涨跌幅: {sign}{change_pct:.2f}%',
        transform=ax.transAxes, fontsize=12, color=color,
        ha='right', va='bottom', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

plt.tight_layout()
chart1_path = os.path.join(OUTPUT_DIR, 'chart1_close_price.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
print(f"\n图1已保存至: {chart1_path}")
plt.close()

# --- 图2: 成交量变化图 ---
fig, ax = plt.subplots(figsize=(14, 5))

# 涨跌颜色
colors = ['#D43030' if df.iloc[i]['close'] >= df.iloc[i]['pre_close'] else '#00AA00'
          for i in range(len(df))]

ax.bar(df['trade_date'], df['vol'], color=colors, width=1.0, alpha=0.8)
ax.set_title(f'图2  {STOCK_NAME}({TS_CODE}) 过去1年日成交量变化', fontsize=16, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('成交量(手)', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# 添加平均成交量线
avg_vol = df['vol'].mean()
ax.axhline(y=avg_vol, color='#4169E1', linewidth=1.5, linestyle='--', label=f'平均成交量: {avg_vol:.0f}手')
ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()
chart2_path = os.path.join(OUTPUT_DIR, 'chart2_volume.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
print(f"图2已保存至: {chart2_path}")
plt.close()

# --- 图3: 日涨跌幅分布图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: 日涨跌幅时间序列
ax1 = axes[0]
colors_pct = ['#D43030' if p > 0 else '#00AA00' for p in df['pct_chg']]
ax1.bar(df['trade_date'], df['pct_chg'], color=colors_pct, width=1.0, alpha=0.8)
ax1.set_title('日涨跌幅时间序列', fontsize=13, fontweight='bold')
ax1.set_xlabel('日期', fontsize=11)
ax1.set_ylabel('涨跌幅(%)', fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.axhline(y=0, color='black', linewidth=0.8)

# 右图: 日涨跌幅直方图
ax2 = axes[1]
n, bins, patches = ax2.hist(df['pct_chg'], bins=30, color='#4169E1', alpha=0.7, edgecolor='white')
ax2.set_title('日涨跌幅分布直方图', fontsize=13, fontweight='bold')
ax2.set_xlabel('涨跌幅(%)', fontsize=11)
ax2.set_ylabel('频次', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axvline(x=0, color='black', linewidth=0.8, linestyle='-')
ax2.axvline(x=df['pct_chg'].mean(), color='#D43030', linewidth=1.5, linestyle='--',
            label=f'均值: {df["pct_chg"].mean():.2f}%')
ax2.legend(fontsize=10)

fig.suptitle(f'图3  {STOCK_NAME}({TS_CODE}) 日涨跌幅分析', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
chart3_path = os.path.join(OUTPUT_DIR, 'chart3_pct_chg.png')
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
print(f"图3已保存至: {chart3_path}")
plt.close()

# ===== 3) 保存CSV数据 =====
csv_path = os.path.join(OUTPUT_DIR, f'{STOCK_NAME}_{STOCK_CODE}_daily_data.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nCSV数据已保存至: {csv_path}")
print(f"数据共 {len(df)} 行, {len(df.columns)} 列")

# ===== 数据统计摘要 =====
stats = {
    'data_count': len(df),
    'date_start': df['trade_date'].min().strftime('%Y-%m-%d'),
    'date_end': df['trade_date'].max().strftime('%Y-%m-%d'),
    'close_mean': df['close'].mean(),
    'close_max': df['close'].max(),
    'close_min': df['close'].min(),
    'close_std': df['close'].std(),
    'close_first': df.iloc[0]['close'],
    'close_last': df.iloc[-1]['close'],
    'change_pct': (df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close'] * 100,
    'vol_mean': df['vol'].mean(),
    'amount_mean': df['amount'].mean(),
    'pct_chg_mean': df['pct_chg'].mean(),
    'pct_chg_std': df['pct_chg'].std(),
    'pct_chg_max': df['pct_chg'].max(),
    'pct_chg_min': df['pct_chg'].min(),
    'up_days': len(df[df['pct_chg'] > 0]),
    'down_days': len(df[df['pct_chg'] < 0]),
    'flat_days': len(df[df['pct_chg'] == 0]),
    'max_close_date': df.loc[df['close'].idxmax(), 'trade_date'].strftime('%Y-%m-%d'),
    'min_close_date': df.loc[df['close'].idxmin(), 'trade_date'].strftime('%Y-%m-%d'),
}

print(f"\n===== 数据统计摘要 =====")
for k, v in stats.items():
    print(f"  {k}: {v}")

# ===== 4) 生成PDF报告 =====
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

# 注册宋体字体
pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))

# 五号字 = 10.5pt
FONT_SIZE = 10.5
# 1.5倍行距
LINE_HEIGHT = FONT_SIZE * 1.5

# 定义段落样式
style_title = ParagraphStyle(
    'TitleStyle',
    fontName='SimSun',
    fontSize=16,
    leading=16 * 1.5,
    alignment=TA_CENTER,
    spaceBefore=0,
    spaceAfter=0,
)

style_heading = ParagraphStyle(
    'HeadingStyle',
    fontName='SimSun',
    fontSize=14,
    leading=14 * 1.5,
    alignment=TA_JUSTIFY,
    spaceBefore=0,
    spaceAfter=0,
)

style_body = ParagraphStyle(
    'BodyStyle',
    fontName='SimSun',
    fontSize=FONT_SIZE,
    leading=LINE_HEIGHT,
    alignment=TA_JUSTIFY,
    spaceBefore=0,
    spaceAfter=0,
    firstLineIndent=FONT_SIZE * 2,  # 首行缩进2字符
)

style_figure_caption = ParagraphStyle(
    'FigureCaption',
    fontName='SimSun',
    fontSize=FONT_SIZE,
    leading=LINE_HEIGHT,
    alignment=TA_CENTER,
    spaceBefore=0,
    spaceAfter=0,
)

# 构建PDF内容
pdf_path = os.path.join(OUTPUT_DIR, '童逸+TASK1.pdf')
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=A4,
    leftMargin=2.5 * cm,
    rightMargin=2.5 * cm,
    topMargin=2.5 * cm,
    bottomMargin=2.5 * cm,
)

elements = []

# --- 标题 ---
elements.append(Paragraph(f'{STOCK_NAME}({TS_CODE})过去1年交易日数据分析报告', style_title))
elements.append(Spacer(1, LINE_HEIGHT))

# --- 摘要 ---
elements.append(Paragraph('一、数据概览', style_heading))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

overview_text = (
    f'本报告基于Tushare Pro API获取的{STOCK_NAME}（股票代码：{TS_CODE}）'
    f'日线行情数据，数据时间范围为{stats["date_start"]}至{stats["date_end"]}，'
    f'共计{stats["data_count"]}个交易日。数据字段包括开盘价（open）、'
    f'最高价（high）、最低价（low）、收盘价（close）、前收盘价（pre_close）、'
    f'涨跌额（change）、涨跌幅（pct_chg）、成交量（vol，单位：手）和成交额'
    f'（amount，单位：千元）。在此基础上，本报告计算了20日移动平均线（MA20）'
    f'和60日移动平均线（MA60）以辅助趋势分析。'
)
elements.append(Paragraph(overview_text, style_body))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

# 关键统计指标表格
table_data = [
    ['统计指标', '数值', '统计指标', '数值'],
    ['收盘价均值', f'{stats["close_mean"]:.2f} 元', '日均成交量', f'{stats["vol_mean"]:.0f} 手'],
    ['收盘价最高', f'{stats["close_max"]:.2f} 元', '日均成交额', f'{stats["amount_mean"]:.0f} 千元'],
    ['收盘价最低', f'{stats["close_min"]:.2f} 元', '日涨跌幅均值', f'{stats["pct_chg_mean"]:.2f}%'],
    ['收盘价标准差', f'{stats["close_std"]:.2f} 元', '日涨跌幅标准差', f'{stats["pct_chg_std"]:.2f}%'],
    ['期初收盘价', f'{stats["close_first"]:.2f} 元', '期末收盘价', f'{stats["close_last"]:.2f} 元'],
    ['区间涨跌幅', f'{stats["change_pct"]:.2f}%', '交易日总数', f'{stats["data_count"]} 天'],
    ['上涨天数', f'{stats["up_days"]} 天', '下跌天数', f'{stats["down_days"]} 天'],
]

table = Table(table_data, colWidths=[3.5 * cm, 3 * cm, 3.5 * cm, 3 * cm])
table.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
    ('FONTSIZE', (0, 0), (-1, -1), FONT_SIZE),
    ('LEADING', (0, 0), (-1, -1), LINE_HEIGHT),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#999999')),
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#E8E8E8')),
    ('TOPPADDING', (0, 0), (-1, -1), 0),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
]))
elements.append(table)
elements.append(Spacer(1, LINE_HEIGHT))

# --- 图1解读 ---
elements.append(Paragraph('二、收盘价走势分析', style_heading))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

# 插入图1
img1 = Image(chart1_path, width=16 * cm, height=8 * cm)
elements.append(img1)
elements.append(Paragraph('图1  中船特气(688146.SH)过去1年收盘价走势', style_figure_caption))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

chart1_analysis = (
    f'图1展示了{STOCK_NAME}在过去1年内的每日收盘价走势，并叠加了20日和60日移动平均线。'
    f'从图中可以看出，该股票在过去1年期间收盘价最高达到{stats["close_max"]:.2f}元'
    f'（出现于{stats["max_close_date"]}），最低触及{stats["close_min"]:.2f}元'
    f'（出现于{stats["min_close_date"]}），期间收盘价均值为{stats["close_mean"]:.2f}元，'
    f'标准差为{stats["close_std"]:.2f}元。从期初的{stats["close_first"]:.2f}元到期末的'
    f'{stats["close_last"]:.2f}元，区间涨跌幅为{stats["change_pct"]:.2f}%。'
    f'20日均线和60日均线分别反映了该股票中短期和中长期的价格趋势，'
    f'当20日均线上穿60日均线时形成"金叉"，通常被视为买入信号；'
    f'反之则形成"死叉"，被视为卖出信号。从整体走势来看，该股票在过去1年中经历了'
    f'一定的价格波动，投资者应关注均线交叉点位以及价格与均线的偏离程度，'
    f'以判断未来趋势方向。'
)
elements.append(Paragraph(chart1_analysis, style_body))
elements.append(Spacer(1, LINE_HEIGHT))

# --- 图2解读 ---
elements.append(Paragraph('三、成交量分析', style_heading))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

# 插入图2
img2 = Image(chart2_path, width=16 * cm, height=5.7 * cm)
elements.append(img2)
elements.append(Paragraph('图2  中船特气(688146.SH)过去1年日成交量变化', style_figure_caption))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

chart2_analysis = (
    f'图2展示了{STOCK_NAME}过去1年的日成交量变化情况，其中红色柱表示当日收盘价高于'
    f'前收盘价（上涨日），绿色柱表示当日收盘价低于前收盘价（下跌日）。'
    f'该股票日均成交量为{stats["vol_mean"]:.0f}手，日均成交额为{stats["amount_mean"]:.0f}千元。'
    f'从图中可以看出，成交量在部分交易日出现明显放量，通常对应着股价的关键转折点'
    f'或重大消息发布期。一般而言，价涨量增表示市场买盘积极，趋势有望延续；'
    f'价跌量增则可能意味着恐慌性抛售。投资者应关注异常放量交易日，'
    f'结合价格走势判断市场情绪和资金流向的变化。当成交量持续低于均值时，'
    f'说明市场交投清淡，股价可能处于盘整阶段。'
)
elements.append(Paragraph(chart2_analysis, style_body))
elements.append(Spacer(1, LINE_HEIGHT))

# --- 图3解读 ---
elements.append(Paragraph('四、日涨跌幅分析', style_heading))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

# 插入图3
img3 = Image(chart3_path, width=16 * cm, height=5.7 * cm)
elements.append(img3)
elements.append(Paragraph('图3  中船特气(688146.SH)日涨跌幅时间序列与分布', style_figure_caption))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

chart3_analysis = (
    f'图3左图为日涨跌幅的时间序列图，右图为日涨跌幅的分布直方图。'
    f'在过去{stats["data_count"]}个交易日中，上涨天数为{stats["up_days"]}天，'
    f'下跌天数为{stats["down_days"]}天，平盘天数为{stats["flat_days"]}天。'
    f'日涨跌幅均值为{stats["pct_chg_mean"]:.2f}%，标准差为{stats["pct_chg_std"]:.2f}%，'
    f'最大单日涨幅为{stats["pct_chg_max"]:.2f}%，最大单日跌幅为{stats["pct_chg_min"]:.2f}%。'
    f'从直方图可以看出，日涨跌幅整体近似呈正态分布，集中在均值附近，'
    f'但两侧尾部存在极端值，表明该股票偶发大幅波动。'
    f'涨跌幅标准差反映了股票的波动性水平，数值越大说明价格波动越剧烈，'
    f'投资风险也相应越高。从涨跌天数比例来看，'
    f'上涨天数占比为{stats["up_days"] / stats["data_count"] * 100:.1f}%，'
    f'下跌天数占比为{stats["down_days"] / stats["data_count"] * 100:.1f}%，'
    f'该比例可在一定程度上反映市场对该股票的整体情绪倾向。'
)
elements.append(Paragraph(chart3_analysis, style_body))
elements.append(Spacer(1, LINE_HEIGHT))

# --- 结论 ---
elements.append(Paragraph('五、总结', style_heading))
elements.append(Spacer(1, LINE_HEIGHT * 0.5))

conclusion_text = (
    f'综合以上分析，{STOCK_NAME}({TS_CODE})在过去1年共{stats["data_count"]}个交易日中，'
    f'收盘价在{stats["close_min"]:.2f}元至{stats["close_max"]:.2f}元之间波动，'
    f'区间涨跌幅为{stats["change_pct"]:.2f}%。从成交量来看，日均成交{stats["vol_mean"]:.0f}手，'
    f'市场交投活跃度适中。日涨跌幅均值为{stats["pct_chg_mean"]:.2f}%，'
    f'波动率（标准差）为{stats["pct_chg_std"]:.2f}%，反映出该股票具有一定的价格波动特征。'
    f'本报告所用原始数据已保存为CSV格式文件（{STOCK_NAME}_{STOCK_CODE}_daily_data.csv），'
    f'包含开盘价、最高价、最低价、收盘价、涨跌幅、成交量等完整字段，可供后续量化分析和策略回测使用。'
    f'投资者在使用上述数据进行决策时，应结合宏观经济环境、行业基本面和公司财务状况等因素综合判断，'
    f'并注意股市投资风险。'
)
elements.append(Paragraph(conclusion_text, style_body))

# 生成PDF
doc.build(elements)
print(f"\nPDF报告已生成: {pdf_path}")
print("全部任务完成!")
