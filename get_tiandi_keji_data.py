"""
天地科技(600587)过去1年交易日数据获取、绘图与保存
使用 Tushare Pro API
"""

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# ===== 配置 =====
TUSHARE_TOKEN = '5907c6e4dc666a6920da3c31435ea985428e8ed5a6c9a70681e3fcb0'
STOCK_CODE = '600587'  # 天地科技
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置 Tushare token
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# ===== 1) 获取过去1年交易日数据 =====
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

print(f"正在获取天地科技({STOCK_CODE})从 {start_date} 到 {end_date} 的日线数据...")

df = pro.daily(ts_code=f'{STOCK_CODE}.SH', start_date=start_date, end_date=end_date)

if df.empty:
    print("未获取到数据，请检查股票代码或网络连接")
    exit(1)

# 按日期升序排列
df = df.sort_values('trade_date', ascending=True).reset_index(drop=True)
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

print(f"成功获取 {len(df)} 条交易日数据")
print(f"日期范围: {df['trade_date'].min().strftime('%Y-%m-%d')} ~ {df['trade_date'].max().strftime('%Y-%m-%d')}")
print(f"\n数据字段: {list(df.columns)}")
print(f"\n前5条数据预览:")
print(df.head())

# ===== 2) 绘制收盘价曲线图 =====
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 7))

# 收盘价曲线
ax.plot(df['trade_date'], df['close'], color='#D43030', linewidth=1.8, label='收盘价')

# 添加均线
if len(df) >= 20:
    df['ma20'] = df['close'].rolling(window=20).mean()
    ax.plot(df['trade_date'], df['ma20'], color='#FFB347', linewidth=1.2, linestyle='--', label='20日均线')

if len(df) >= 60:
    df['ma60'] = df['close'].rolling(window=60).mean()
    ax.plot(df['trade_date'], df['ma60'], color='#4169E1', linewidth=1.2, linestyle='--', label='60日均线')

# 标注最高最低收盘价
max_close_idx = df['close'].idxmax()
min_close_idx = df['close'].idxmax()  # fix below
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

# 美化图表
ax.set_title(f'天地科技({STOCK_CODE}) 过去1年收盘价走势', fontsize=16, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('收盘价(元)', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.xticks(rotation=45)

# 添加涨跌幅信息
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

# 保存图片
chart_path = os.path.join(OUTPUT_DIR, f'tiandi_keji_{STOCK_CODE}_close_price.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存至: {chart_path}")
plt.close()

# ===== 3) 保存CSV数据 =====
csv_path = os.path.join(OUTPUT_DIR, f'tiandi_keji_{STOCK_CODE}_daily_data.csv')
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"CSV数据已保存至: {csv_path}")
print(f"数据共 {len(df)} 行, {len(df.columns)} 列")

# 数据统计摘要
print(f"\n===== 数据统计摘要 =====")
print(f"收盘价均值: {df['close'].mean():.2f}")
print(f"收盘价最高: {df['close'].max():.2f}")
print(f"收盘价最低: {df['close'].min():.2f}")
print(f"收盘价标准差: {df['close'].std():.2f}")
print(f"日均成交量: {df['vol'].mean():.0f} 手")
print(f"日均成交额: {df['amount'].mean():.0f} 千元")
