# -*- coding: utf-8 -*-
"""
GitHub Actions 自动更新股票数据脚本
从 Tushare 获取最新日线数据，更新CSV文件，重新生成dashboard
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 获取Tushare Token
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', '')
if not TUSHARE_TOKEN:
    print("Warning: TUSHARE_TOKEN not set, using default")
    TUSHARE_TOKEN = '5907c6e4dc666a6920da3c31435ea985428e8ed5a6c9a70681e3fcb0'

import tushare as ts
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 股票列表
STOCKS = [
    {'name': '中船特气', 'code': '688146.SH', 'csv': '中船特气_688146_daily_data.csv'},
    {'name': '天地科技', 'code': '600587.SH', 'csv': 'tiandi_keji_600587_daily_data.csv'},
    {'name': '平安银行', 'code': '000001.SZ', 'csv': '平安银行_000001_daily_data.csv'},
]

# 计算日期范围：过去1年
END_DATE = datetime.now().strftime('%Y%m%d')
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

print(f"Updating stock data from {START_DATE} to {END_DATE}")

for stock in STOCKS:
    print(f"\nFetching {stock['name']} ({stock['code']})...")
    try:
        df = pro.daily(ts_code=stock['code'], start_date=START_DATE, end_date=END_DATE)
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 计算均线
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()

        # 保存CSV
        df.to_csv(stock['csv'], index=False, encoding='utf-8-sig')
        print(f"  Saved {stock['csv']}: {len(df)} records")
        print(f"  Date range: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")
        print(f"  Close range: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    except Exception as e:
        print(f"  Error: {e}")

# 重新生成dashboard
print("\nRegenerating dashboard...")
try:
    # 内联dashboard生成逻辑（避免路径依赖）
    data = {}
    for stock in STOCKS:
        if os.path.exists(stock['csv']):
            df = pd.read_csv(stock['csv'])
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
            key = stock['name'][:2]  # 简化key
            data[f'{key}_daily'] = df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'pct_chg']].to_dict('records')

    # 生成简化版dashboard
    import json
    json_data = json.dumps(data, ensure_ascii=False, default=str)

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI量化交易策略分析面板</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Microsoft YaHei', sans-serif; background: #0f1923; color: #e0e0e0; }}
        .header {{ background: linear-gradient(135deg, #1a2a3a 0%, #0f1923 100%); padding: 20px 40px; border-bottom: 2px solid #2a3f5f; }}
        .header h1 {{ font-size: 24px; color: #4fc3f7; }}
        .nav {{ display: flex; background: #1a2a3a; padding: 0 40px; border-bottom: 1px solid #2a3f5f; }}
        .nav-item {{ padding: 12px 24px; cursor: pointer; color: #78909c; font-size: 14px; border-bottom: 2px solid transparent; }}
        .nav-item.active {{ color: #4fc3f7; border-bottom-color: #4fc3f7; }}
        .container {{ padding: 20px 40px; max-width: 1600px; margin: 0 auto; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .chart-box {{ background: #1a2a3a; border-radius: 8px; padding: 20px; border: 1px solid #2a3f5f; margin-bottom: 20px; }}
        .chart-box h3 {{ color: #4fc3f7; margin-bottom: 12px; }}
        .chart {{ width: 100%; height: 450px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }}
        .stat-card {{ background: #1a2a3a; border-radius: 8px; padding: 16px; border: 1px solid #2a3f5f; text-align: center; }}
        .stat-card .label {{ font-size: 12px; color: #78909c; }}
        .stat-card .value {{ font-size: 22px; font-weight: bold; color: #4fc3f7; }}
        .up {{ color: #ef5350 !important; }}
        .down {{ color: #26a69a !important; }}
        .footer {{ text-align: center; padding: 20px; color: #546e7a; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AI量化交易策略 - 股票数据自动更新面板</h1>
        <p style="color:#78909c;font-size:13px;margin-top:4px;">数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    <div class="container">
        <div id="stats" class="stats-grid"></div>
        <div class="chart-box">
            <h3>收盘价走势对比</h3>
            <div class="chart" id="chartClose"></div>
        </div>
    </div>
    <div class="footer">GitHub Actions Auto-Update | AasChief/AI-quant</div>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <script>
    const data = {json_data};
    const stocks = {json.dumps([s['name'] for s in STOCKS])};
    const keys = {json.dumps([s['name'][:2] for s in STOCKS])};
    const chart = echarts.init(document.getElementById('chartClose'), 'dark');
    const series = [];
    const dates = new Set();
    keys.forEach((k, i) => {{
        const sd = data[k + '_daily'] || [];
        sd.forEach(d => dates.add(d.trade_date));
        series.push({{ name: stocks[i], type: 'line', data: sd.map(d => [d.trade_date, d.close]), smooth: true }});
    }});
    chart.setOption({{
        backgroundColor: 'transparent',
        tooltip: {{ trigger: 'axis' }},
        legend: {{ data: stocks, top: 0 }},
        xAxis: {{ type: 'category' }},
        yAxis: {{ type: 'value', scale: true }},
        series: series,
        dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
    }});
    // Stats
    let html = '';
    keys.forEach((k, i) => {{
        const sd = data[k + '_daily'] || [];
        if (sd.length) {{
            const ret = ((sd[sd.length-1].close - sd[0].close) / sd[0].close * 100).toFixed(2);
            html += `<div class="stat-card"><div class="label">${{stocks[i]}}</div><div class="value ${{ret>0?'up':'down'}}">${{ret>0?'+':''}}${{ret}}%</div></div>`;
        }}
    }});
    document.getElementById('stats').innerHTML = html;
    </script>
</body>
</html>'''

    os.makedirs('dashboard', exist_ok=True)
    with open('dashboard/index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print("Dashboard regenerated successfully!")
except Exception as e:
    print(f"Dashboard regeneration error: {e}")

print("\nDone!")
