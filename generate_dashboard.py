# -*- coding: utf-8 -*-
"""
生成四次作业可视化面板网页
读取所有CSV数据，生成包含交互式图表的HTML dashboard
"""
import pandas as pd
import json
import os

BASE_DIR = 'E:/量化交易：AI大模型辅助的金融交易策略'

def load_csv_data():
    """加载所有CSV数据"""
    data = {}

    # 中船特气日线数据
    df = pd.read_csv(os.path.join(BASE_DIR, '中船特气_688146_daily_data.csv'))
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
    data['zc_daily'] = df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'pct_chg']].to_dict('records')

    # 天地科技日线数据
    df2 = pd.read_csv(os.path.join(BASE_DIR, 'tiandi_keji_600587_daily_data.csv'))
    df2['trade_date'] = pd.to_datetime(df2['trade_date']).dt.strftime('%Y-%m-%d')
    data['td_daily'] = df2[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'pct_chg']].to_dict('records')

    # 平安银行日线数据
    df3 = pd.read_csv(os.path.join(BASE_DIR, '平安银行_000001_daily_data.csv'))
    df3['trade_date'] = pd.to_datetime(df3['trade_date']).dt.strftime('%Y-%m-%d')
    data['pa_daily'] = df3[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'pct_chg']].to_dict('records')

    # 技术指标数据
    if os.path.exists(os.path.join(BASE_DIR, '中船特气_688146_indicators.csv')):
        df_ind = pd.read_csv(os.path.join(BASE_DIR, '中船特气_688146_indicators.csv'))
        df_ind['trade_date'] = pd.to_datetime(df_ind['trade_date']).dt.strftime('%Y-%m-%d')
        cols = ['trade_date', 'close', 'rsi', 'macd_dif', 'macd_dea', 'macd_hist',
                'boll_upper', 'boll_mid', 'boll_lower', 'kdj_k', 'kdj_d', 'kdj_j']
        cols = [c for c in cols if c in df_ind.columns]
        data['indicators'] = df_ind[cols].to_dict('records')

    # 双均线策略数据
    if os.path.exists(os.path.join(BASE_DIR, '中船特气_688146_ma_strategy.csv')):
        df_ma = pd.read_csv(os.path.join(BASE_DIR, '中船特气_688146_ma_strategy.csv'))
        df_ma['trade_date'] = pd.to_datetime(df_ma['trade_date']).dt.strftime('%Y-%m-%d')
        cols = ['trade_date', 'close', 'ma_short', 'ma_long', 'signal', 'strategy_nav', 'buy_hold_nav']
        cols = [c for c in cols if c in df_ma.columns]
        data['ma_strategy'] = df_ma[cols].to_dict('records')

    # 海龟策略数据
    if os.path.exists(os.path.join(BASE_DIR, '中船特气_688146_turtle_strategy.csv')):
        df_tur = pd.read_csv(os.path.join(BASE_DIR, '中船特气_688146_turtle_strategy.csv'))
        df_tur['trade_date'] = pd.to_datetime(df_tur['trade_date']).dt.strftime('%Y-%m-%d')
        cols = ['trade_date', 'close', 'upper_channel', 'lower_channel', 'atr',
                'signal', 'position', 'strategy_nav', 'buy_hold_nav']
        cols = [c for c in cols if c in df_tur.columns]
        data['turtle_strategy'] = df_tur[cols].to_dict('records')

    # 策略对比结果
    if os.path.exists(os.path.join(BASE_DIR, '海龟策略对比结果.csv')):
        data['turtle_comparison'] = pd.read_csv(os.path.join(BASE_DIR, '海龟策略对比结果.csv')).to_dict('records')

    if os.path.exists(os.path.join(BASE_DIR, '双均线策略对比结果.csv')):
        data['ma_comparison'] = pd.read_csv(os.path.join(BASE_DIR, '双均线策略对比结果.csv')).to_dict('records')

    return data


def generate_dashboard():
    data = load_csv_data()
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
        body {{
            font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
            background: #0f1923;
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, #1a2a3a 0%, #0f1923 100%);
            padding: 20px 40px;
            border-bottom: 2px solid #2a3f5f;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 24px;
            color: #4fc3f7;
            letter-spacing: 2px;
        }}
        .header .subtitle {{
            font-size: 13px;
            color: #78909c;
            margin-top: 4px;
        }}
        .header .date {{
            font-size: 13px;
            color: #78909c;
        }}
        .nav {{
            display: flex;
            gap: 0;
            background: #1a2a3a;
            padding: 0 40px;
            border-bottom: 1px solid #2a3f5f;
            overflow-x: auto;
        }}
        .nav-item {{
            padding: 12px 24px;
            cursor: pointer;
            color: #78909c;
            font-size: 14px;
            border-bottom: 2px solid transparent;
            transition: all 0.3s;
            white-space: nowrap;
        }}
        .nav-item:hover {{ color: #4fc3f7; }}
        .nav-item.active {{
            color: #4fc3f7;
            border-bottom-color: #4fc3f7;
        }}
        .container {{
            padding: 20px 40px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .chart-grid.full {{ grid-template-columns: 1fr; }}
        .chart-box {{
            background: #1a2a3a;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #2a3f5f;
        }}
        .chart-box h3 {{
            font-size: 15px;
            color: #4fc3f7;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .chart-box h3::before {{
            content: '';
            width: 4px;
            height: 16px;
            background: #4fc3f7;
            border-radius: 2px;
        }}
        .chart {{
            width: 100%;
            height: 350px;
        }}
        .chart.tall {{ height: 450px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: #1a2a3a;
            border-radius: 8px;
            padding: 16px 20px;
            border: 1px solid #2a3f5f;
            text-align: center;
        }}
        .stat-card .label {{
            font-size: 12px;
            color: #78909c;
            margin-bottom: 6px;
        }}
        .stat-card .value {{
            font-size: 22px;
            font-weight: bold;
        }}
        .stat-card .value.up {{ color: #ef5350; }}
        .stat-card .value.down {{ color: #26a69a; }}
        .stat-card .value.neutral {{ color: #4fc3f7; }}
        .stock-selector {{
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .stock-btn {{
            padding: 6px 16px;
            border: 1px solid #2a3f5f;
            background: #1a2a3a;
            color: #78909c;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.3s;
        }}
        .stock-btn:hover {{ border-color: #4fc3f7; color: #4fc3f7; }}
        .stock-btn.active {{
            background: #4fc3f7;
            color: #0f1923;
            border-color: #4fc3f7;
        }}
        .table-box {{
            background: #1a2a3a;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #2a3f5f;
            overflow-x: auto;
        }}
        .table-box table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .table-box th {{
            background: #243447;
            padding: 10px 12px;
            text-align: center;
            color: #4fc3f7;
            border-bottom: 2px solid #2a3f5f;
        }}
        .table-box td {{
            padding: 8px 12px;
            text-align: center;
            border-bottom: 1px solid #2a3f5f;
        }}
        .table-box tr:hover td {{ background: #243447; }}
        .up-text {{ color: #ef5350; }}
        .down-text {{ color: #26a69a; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #546e7a;
            font-size: 12px;
            border-top: 1px solid #2a3f5f;
            margin-top: 20px;
        }}
        @media (max-width: 768px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .container {{ padding: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>AI大模型辅助的金融交易策略 - 可视化分析面板</h1>
            <div class="subtitle">中船特气(688146) | 天地科技(600587) | 平安银行(000001)</div>
        </div>
        <div class="date" id="updateDate"></div>
    </div>

    <div class="nav">
        <div class="nav-item active" onclick="switchTab('task1')">TASK1 数据概览</div>
        <div class="nav-item" onclick="switchTab('task2')">TASK2 技术指标</div>
        <div class="nav-item" onclick="switchTab('task3')">TASK3 双均线策略</div>
        <div class="nav-item" onclick="switchTab('task4')">TASK4 海龟策略</div>
        <div class="nav-item" onclick="switchTab('summary')">策略对比汇总</div>
    </div>

    <div class="container">
        <!-- TASK1: 数据概览 -->
        <div class="tab-content active" id="task1">
            <div class="stock-selector">
                <button class="stock-btn active" onclick="switchStock('zc')">中船特气 688146</button>
                <button class="stock-btn" onclick="switchStock('td')">天地科技 600587</button>
                <button class="stock-btn" onclick="switchStock('pa')">平安银行 000001</button>
            </div>
            <div class="stats-grid" id="task1Stats"></div>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>K线图 & 成交量</h3>
                    <div class="chart tall" id="chartKline"></div>
                </div>
                <div class="chart-box">
                    <h3>收盘价走势</h3>
                    <div class="chart tall" id="chartClose"></div>
                </div>
            </div>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>日涨跌幅分布</h3>
                    <div class="chart" id="chartPctDist"></div>
                </div>
                <div class="chart-box">
                    <h3>成交量变化</h3>
                    <div class="chart" id="chartVolume"></div>
                </div>
            </div>
        </div>

        <!-- TASK2: 技术指标 -->
        <div class="tab-content" id="task2">
            <div class="stats-grid" id="task2Stats"></div>
            <div class="chart-grid full">
                <div class="chart-box">
                    <h3>RSI 相对强弱指标 (14日)</h3>
                    <div class="chart" id="chartRSI"></div>
                </div>
            </div>
            <div class="chart-grid full">
                <div class="chart-box">
                    <h3>MACD 指标 (12,26,9)</h3>
                    <div class="chart" id="chartMACD"></div>
                </div>
            </div>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>布林带 Bollinger Bands (20,2)</h3>
                    <div class="chart tall" id="chartBoll"></div>
                </div>
                <div class="chart-box">
                    <h3>KDJ 随机指标 (9,3,3)</h3>
                    <div class="chart tall" id="chartKDJ"></div>
                </div>
            </div>
        </div>

        <!-- TASK3: 双均线策略 -->
        <div class="tab-content" id="task3">
            <div class="stats-grid" id="task3Stats"></div>
            <div class="chart-grid full">
                <div class="chart-box">
                    <h3>双均线策略交易信号 (MA5/MA15)</h3>
                    <div class="chart tall" id="chartMA"></div>
                </div>
            </div>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>策略净值 vs 买入持有</h3>
                    <div class="chart tall" id="chartMANav"></div>
                </div>
                <div class="chart-box">
                    <h3>策略回撤对比</h3>
                    <div class="chart tall" id="chartMADD"></div>
                </div>
            </div>
            <div class="chart-grid full">
                <div class="table-box">
                    <h3 style="color:#4fc3f7;font-size:15px;margin-bottom:12px;">多股票多参数对比结果</h3>
                    <table id="maTable"></table>
                </div>
            </div>
        </div>

        <!-- TASK4: 海龟策略 -->
        <div class="tab-content" id="task4">
            <div class="stats-grid" id="task4Stats"></div>
            <div class="chart-grid full">
                <div class="chart-box">
                    <h3>海龟策略交易信号 (唐奇安通道20/10)</h3>
                    <div class="chart tall" id="chartTurtle"></div>
                </div>
            </div>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>策略净值 vs 买入持有</h3>
                    <div class="chart tall" id="chartTurtleNav"></div>
                </div>
                <div class="chart-box">
                    <h3>ATR 平均真实波幅</h3>
                    <div class="chart tall" id="chartATR"></div>
                </div>
            </div>
            <div class="chart-grid full">
                <div class="table-box">
                    <h3 style="color:#4fc3f7;font-size:15px;margin-bottom:12px;">海龟策略多股票多参数对比结果</h3>
                    <table id="turtleTable"></table>
                </div>
            </div>
        </div>

        <!-- 策略对比汇总 -->
        <div class="tab-content" id="summary">
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>各策略累计回报对比</h3>
                    <div class="chart tall" id="chartCompareReturn"></div>
                </div>
                <div class="chart-box">
                    <h3>各策略夏普比率对比</h3>
                    <div class="chart tall" id="chartCompareSharpe"></div>
                </div>
            </div>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>各策略最大回撤对比</h3>
                    <div class="chart tall" id="chartCompareMDD"></div>
                </div>
                <div class="chart-box">
                    <h3>各策略胜率对比</h3>
                    <div class="chart tall" id="chartCompareWinRate"></div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        AI大模型辅助的金融交易策略 | 童逸 | GitHub: AasChief/AI-quant | 数据自动更新
    </div>

    <script>
    const rawData = {json_data};
    let currentStock = 'zc';
    const charts = {{}};

    // 日期显示
    document.getElementById('updateDate').textContent = '更新时间: ' + new Date().toLocaleString('zh-CN');

    function switchTab(tabId) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        event.target.classList.add('active');
        setTimeout(() => {{
            Object.values(charts).forEach(c => c && c.resize());
        }}, 100);
    }}

    function switchStock(stock) {{
        currentStock = stock;
        document.querySelectorAll('.stock-btn').forEach(el => el.classList.remove('active'));
        event.target.classList.add('active');
        renderTask1();
    }}

    function getStockData() {{
        const map = {{ zc: 'zc_daily', td: 'td_daily', pa: 'pa_daily' }};
        return rawData[map[currentStock]] || [];
    }}

    function initChart(id) {{
        if (charts[id]) charts[id].dispose();
        const el = document.getElementById(id);
        if (!el) return null;
        charts[id] = echarts.init(el, 'dark');
        return charts[id];
    }}

    // ===== TASK1: 数据概览 =====
    function renderTask1() {{
        const data = getStockData();
        if (!data.length) return;
        const dates = data.map(d => d.trade_date);
        const closes = data.map(d => d.close);
        const volumes = data.map(d => d.vol);
        const pctChgs = data.map(d => d.pct_chg);
        const highs = data.map(d => d.high);
        const lows = data.map(d => d.low);
        const opens = data.map(d => d.open);

        // 统计卡片
        const upDays = pctChgs.filter(p => p > 0).length;
        const downDays = pctChgs.filter(p => p < 0).length;
        const maxPct = Math.max(...pctChgs);
        const minPct = Math.min(...pctChgs);
        const totalReturn = ((closes[closes.length-1] - closes[0]) / closes[0] * 100).toFixed(2);
        const stockNames = {{ zc: '中船特气', td: '天地科技', pa: '平安银行' }};

        document.getElementById('task1Stats').innerHTML = `
            <div class="stat-card"><div class="label">股票</div><div class="value neutral">${{stockNames[currentStock]}}</div></div>
            <div class="stat-card"><div class="label">交易日数</div><div class="value neutral">${{data.length}}</div></div>
            <div class="stat-card"><div class="label">期初收盘价</div><div class="value neutral">${{closes[0].toFixed(2)}}</div></div>
            <div class="stat-card"><div class="label">期末收盘价</div><div class="value neutral">${{closes[closes.length-1].toFixed(2)}}</div></div>
            <div class="stat-card"><div class="label">区间涨跌幅</div><div class="value ${{totalReturn > 0 ? 'up' : 'down'}}">${{totalReturn > 0 ? '+' : ''}}${{totalReturn}}%</div></div>
            <div class="stat-card"><div class="label">最高价</div><div class="value up">${{Math.max(...highs).toFixed(2)}}</div></div>
            <div class="stat-card"><div class="label">最低价</div><div class="value down">${{Math.min(...lows).toFixed(2)}}</div></div>
            <div class="stat-card"><div class="label">上涨/下跌天数</div><div class="value neutral">${{upDays}} / ${{downDays}}</div></div>
            <div class="stat-card"><div class="label">最大单日涨幅</div><div class="value up">+${{maxPct.toFixed(2)}}%</div></div>
            <div class="stat-card"><div class="label">最大单日跌幅</div><div class="value down">${{minPct.toFixed(2)}}%</div></div>
        `;

        // K线图
        const kChart = initChart('chartKline');
        if (kChart) {{
            kChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }} }},
                legend: {{ data: ['K线', '成交量'], top: 0 }},
                grid: [
                    {{ left: '8%', right: '3%', top: '8%', height: '55%' }},
                    {{ left: '8%', right: '3%', top: '70%', height: '20%' }}
                ],
                xAxis: [
                    {{ type: 'category', data: dates, gridIndex: 0, axisLabel: {{ show: false }} }},
                    {{ type: 'category', data: dates, gridIndex: 1 }}
                ],
                yAxis: [
                    {{ scale: true, gridIndex: 0, splitLine: {{ lineStyle: {{ color: '#2a3f5f' }} }} }},
                    {{ gridIndex: 1, splitLine: {{ show: false }} }}
                ],
                series: [
                    {{
                        name: 'K线', type: 'candlestick', data: data.map(d => [d.open, d.close, d.low, d.high]),
                        xAxisIndex: 0, yAxisIndex: 0,
                        itemStyle: {{ color: '#ef5350', color0: '#26a69a', borderColor: '#ef5350', borderColor0: '#26a69a' }}
                    }},
                    {{
                        name: '成交量', type: 'bar', data: volumes, xAxisIndex: 1, yAxisIndex: 1,
                        itemStyle: {{ color: '#2a3f5f' }}
                    }}
                ],
                dataZoom: [{{ type: 'inside', xAxisIndex: [0, 1], start: 0, end: 100 }}]
            }});
        }}

        // 收盘价走势
        const closeChart = initChart('chartClose');
        if (closeChart) {{
            closeChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value', scale: true }},
                series: [{{
                    name: '收盘价', type: 'line', data: closes, smooth: true,
                    lineStyle: {{ color: '#4fc3f7', width: 2 }},
                    areaStyle: {{ color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {{ offset: 0, color: 'rgba(79,195,247,0.3)' }},
                        {{ offset: 1, color: 'rgba(79,195,247,0.01)' }}
                    ]) }}
                }}],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        // 涨跌幅分布
        const pctChart = initChart('chartPctDist');
        if (pctChart) {{
            const bins = [-15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15, 20, 25];
            const counts = new Array(bins.length - 1).fill(0);
            pctChgs.forEach(p => {{
                for (let i = 0; i < bins.length - 1; i++) {{
                    if (p >= bins[i] && p < bins[i+1]) {{ counts[i]++; break; }}
                }}
            }});
            pctChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                xAxis: {{ type: 'category', data: bins.slice(0, -1).map((b, i) => `${{b}}~${{bins[i+1]}}%`) }},
                yAxis: {{ type: 'value' }},
                series: [{{
                    type: 'bar', data: counts,
                    itemStyle: {{ color: function(p) {{ return p.dataIndex < 4 ? '#26a69a' : '#ef5350'; }} }}
                }}]
            }});
        }}

        // 成交量
        const volChart = initChart('chartVolume');
        if (volChart) {{
            volChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value' }},
                series: [{{
                    type: 'bar', data: volumes,
                    itemStyle: {{ color: '#2a3f5f' }}
                }}],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}
    }}

    // ===== TASK2: 技术指标 =====
    function renderTask2() {{
        const data = rawData.indicators || [];
        if (!data.length) return;
        const dates = data.map(d => d.trade_date);

        // RSI
        const rsiChart = initChart('chartRSI');
        if (rsiChart) {{
            rsiChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['RSI(14)'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value', min: 0, max: 100 }},
                series: [{{
                    name: 'RSI(14)', type: 'line', data: data.map(d => d.rsi),
                    lineStyle: {{ color: '#ab47bc', width: 1.5 }}
                }}],
                visualMap: {{
                    show: false, pieces: [
                        {{ gt: 70, lte: 100, color: '#ef5350' }},
                        {{ gt: 30, lte: 70, color: '#ab47bc' }},
                        {{ gte: 0, lte: 30, color: '#26a69a' }}
                    ], outOfRange: {{ color: '#999' }}
                }},
                markLine: {{ data: [{{ yAxis: 70 }}, {{ yAxis: 30 }}, {{ yAxis: 50 }}] }},
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        // MACD
        const macdChart = initChart('chartMACD');
        if (macdChart) {{
            const histData = data.map(d => d.macd_hist);
            macdChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['DIF', 'DEA', 'MACD柱'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value' }},
                series: [
                    {{ name: 'DIF', type: 'line', data: data.map(d => d.macd_dif), lineStyle: {{ color: '#4fc3f7', width: 1.5 }} }},
                    {{ name: 'DEA', type: 'line', data: data.map(d => d.macd_dea), lineStyle: {{ color: '#ffa726', width: 1.5 }} }},
                    {{ name: 'MACD柱', type: 'bar', data: histData.map(v => ({{
                        value: v, itemStyle: {{ color: v >= 0 ? '#ef5350' : '#26a69a' }}
                    }})) }}
                ],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        // 布林带
        const bollChart = initChart('chartBoll');
        if (bollChart) {{
            bollChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['收盘价', '上轨', '中轨', '下轨'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value', scale: true }},
                series: [
                    {{ name: '收盘价', type: 'line', data: data.map(d => d.close), lineStyle: {{ color: '#4fc3f7', width: 1 }} }},
                    {{ name: '上轨', type: 'line', data: data.map(d => d.boll_upper), lineStyle: {{ color: '#ef5350', width: 1, type: 'dashed' }} }},
                    {{ name: '中轨', type: 'line', data: data.map(d => d.boll_mid), lineStyle: {{ color: '#ffa726', width: 1 }} }},
                    {{ name: '下轨', type: 'line', data: data.map(d => d.boll_lower), lineStyle: {{ color: '#26a69a', width: 1, type: 'dashed' }} }}
                ],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        // KDJ
        const kdjChart = initChart('chartKDJ');
        if (kdjChart) {{
            kdjChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['K', 'D', 'J'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value' }},
                series: [
                    {{ name: 'K', type: 'line', data: data.map(d => d.kdj_k), lineStyle: {{ color: '#4fc3f7', width: 1.5 }} }},
                    {{ name: 'D', type: 'line', data: data.map(d => d.kdj_d), lineStyle: {{ color: '#ffa726', width: 1.5 }} }},
                    {{ name: 'J', type: 'line', data: data.map(d => d.kdj_j), lineStyle: {{ color: '#ab47bc', width: 1 }} }}
                ],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        // 统计卡片
        const lastData = data[data.length - 1];
        document.getElementById('task2Stats').innerHTML = `
            <div class="stat-card"><div class="label">最新RSI(14)</div><div class="value ${{lastData.rsi > 70 ? 'up' : lastData.rsi < 30 ? 'down' : 'neutral'}}">${{lastData.rsi ? lastData.rsi.toFixed(2) : 'N/A'}}</div></div>
            <div class="stat-card"><div class="label">最新DIF</div><div class="value neutral">${{lastData.macd_dif ? lastData.macd_dif.toFixed(4) : 'N/A'}}</div></div>
            <div class="stat-card"><div class="label">最新DEA</div><div class="value neutral">${{lastData.macd_dea ? lastData.macd_dea.toFixed(4) : 'N/A'}}</div></div>
            <div class="stat-card"><div class="label">最新MACD柱</div><div class="value ${{lastData.macd_hist >= 0 ? 'up' : 'down'}}">${{lastData.macd_hist ? lastData.macd_hist.toFixed(4) : 'N/A'}}</div></div>
            <div class="stat-card"><div class="label">布林上轨</div><div class="value up">${{lastData.boll_upper ? lastData.boll_upper.toFixed(2) : 'N/A'}}</div></div>
            <div class="stat-card"><div class="label">布林下轨</div><div class="value down">${{lastData.boll_lower ? lastData.boll_lower.toFixed(2) : 'N/A'}}</div></div>
            <div class="stat-card"><div class="label">最新K值</div><div class="value neutral">${{lastData.kdj_k ? lastData.kdj_k.toFixed(2) : 'N/A'}}</div></div>
            <div class="stat-card"><div class="label">最新J值</div><div class="value neutral">${{lastData.kdj_j ? lastData.kdj_j.toFixed(2) : 'N/A'}}</div></div>
        `;
    }}

    // ===== TASK3: 双均线策略 =====
    function renderTask3() {{
        const data = rawData.ma_strategy || [];
        if (!data.length) return;
        const dates = data.map(d => d.trade_date);

        const maChart = initChart('chartMA');
        if (maChart) {{
            const buySignals = data.filter(d => d.signal === 1);
            const sellSignals = data.filter(d => d.signal === -1);
            maChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['收盘价', 'MA5', 'MA15', '买入', '卖出'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value', scale: true }},
                series: [
                    {{ name: '收盘价', type: 'line', data: data.map(d => d.close), lineStyle: {{ color: '#4fc3f7', width: 1 }} }},
                    {{ name: 'MA5', type: 'line', data: data.map(d => d.ma_short), lineStyle: {{ color: '#ffa726', width: 1 }} }},
                    {{ name: 'MA15', type: 'line', data: data.map(d => d.ma_long), lineStyle: {{ color: '#ab47bc', width: 1 }} }},
                    {{ name: '买入', type: 'scatter', data: buySignals.map(d => [d.trade_date, d.close]),
                       symbol: 'triangle', symbolSize: 12, itemStyle: {{ color: '#ef5350' }} }},
                    {{ name: '卖出', type: 'scatter', data: sellSignals.map(d => [d.trade_date, d.close]),
                       symbol: 'pin', symbolSize: 12, itemStyle: {{ color: '#26a69a' }} }}
                ],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        const navChart = initChart('chartMANav');
        if (navChart) {{
            navChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['策略净值', '买入持有'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value' }},
                series: [
                    {{ name: '策略净值', type: 'line', data: data.map(d => d.strategy_nav), lineStyle: {{ color: '#ef5350', width: 2 }} }},
                    {{ name: '买入持有', type: 'line', data: data.map(d => d.buy_hold_nav), lineStyle: {{ color: '#4fc3f7', width: 2 }} }}
                ],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        const ddChart = initChart('chartMADD');
        if (ddChart && data.length > 0) {{
            const nav = data.map(d => d.strategy_nav || 1);
            const bhNav = data.map(d => d.buy_hold_nav || 1);
            const peak = []; const bhPeak = [];
            let maxP = 0; let maxBhP = 0;
            nav.forEach(v => {{ maxP = Math.max(maxP, v); peak.push((v - maxP) / maxP * 100); }});
            bhNav.forEach(v => {{ maxBhP = Math.max(maxBhP, v); bhPeak.push((v - maxBhP) / maxBhP * 100); }});
            ddChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['策略回撤', '买入持有回撤'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value' }},
                series: [
                    {{ name: '策略回撤', type: 'line', data: peak, areaStyle: {{ color: 'rgba(239,83,80,0.3)' }}, lineStyle: {{ color: '#ef5350' }} }},
                    {{ name: '买入持有回撤', type: 'line', data: bhPeak, areaStyle: {{ color: 'rgba(79,195,247,0.2)' }}, lineStyle: {{ color: '#4fc3f7' }} }}
                ],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        // 对比表
        const maComp = rawData.ma_comparison || [];
        if (maComp.length) {{
            const tbl = document.getElementById('maTable');
            let html = '<thead><tr><th>股票</th><th>参数</th><th>策略回报(%)</th><th>买入持有(%)</th><th>夏普比率</th><th>最大回撤(%)</th><th>交易次数</th><th>胜率(%)</th></tr></thead><tbody>';
            maComp.forEach(r => {{
                html += `<tr><td>${{r.stock||r['股票']||''}}</td><td>${{r.config||r['参数']||''}}</td><td class="${{(r.strategy_cum_return||r['策略回报']) >= 0 ? 'up-text' : 'down-text'}}">${{r.strategy_cum_return||r['策略回报']||''}}</td><td>${{r.buy_hold_cum_return||r['买入持有']||''}}</td><td>${{r.strategy_sharpe||r['夏普比率']||''}}</td><td class="down-text">${{r.strategy_mdd||r['最大回撤']||''}}</td><td>${{r.n_trades||r['交易次数']||''}}</td><td>${{r.win_rate||r['胜率']||''}}</td></tr>`;
            }});
            html += '</tbody>';
            tbl.innerHTML = html;
        }}
    }}

    // ===== TASK4: 海龟策略 =====
    function renderTask4() {{
        const data = rawData.turtle_strategy || [];
        if (!data.length) return;
        const dates = data.map(d => d.trade_date);

        const turChart = initChart('chartTurtle');
        if (turChart) {{
            const buySignals = data.filter(d => d.signal === 1);
            const sellSignals = data.filter(d => d.signal === -1);
            turChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['收盘价', '上轨(20日)', '下轨(10日)', '买入', '卖出'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value', scale: true }},
                series: [
                    {{ name: '收盘价', type: 'line', data: data.map(d => d.close), lineStyle: {{ color: '#4fc3f7', width: 1 }} }},
                    {{ name: '上轨(20日)', type: 'line', data: data.map(d => d.upper_channel), lineStyle: {{ color: '#ef5350', width: 1, type: 'dashed' }} }},
                    {{ name: '下轨(10日)', type: 'line', data: data.map(d => d.lower_channel), lineStyle: {{ color: '#26a69a', width: 1, type: 'dashed' }} }},
                    {{ name: '买入', type: 'scatter', data: buySignals.map(d => [d.trade_date, d.close]),
                       symbol: 'triangle', symbolSize: 12, itemStyle: {{ color: '#ef5350' }} }},
                    {{ name: '卖出', type: 'scatter', data: sellSignals.map(d => [d.trade_date, d.close]),
                       symbol: 'pin', symbolSize: 12, itemStyle: {{ color: '#26a69a' }} }}
                ],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        const navChart = initChart('chartTurtleNav');
        if (navChart) {{
            navChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['策略净值', '买入持有'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value' }},
                series: [
                    {{ name: '策略净值', type: 'line', data: data.map(d => d.strategy_nav), lineStyle: {{ color: '#ef5350', width: 2 }} }},
                    {{ name: '买入持有', type: 'line', data: data.map(d => d.buy_hold_nav), lineStyle: {{ color: '#4fc3f7', width: 2 }} }}
                ],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        const atrChart = initChart('chartATR');
        if (atrChart) {{
            atrChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                legend: {{ data: ['ATR(14)'], top: 0 }},
                xAxis: {{ type: 'category', data: dates }},
                yAxis: {{ type: 'value' }},
                series: [{{
                    name: 'ATR(14)', type: 'line', data: data.map(d => d.atr),
                    lineStyle: {{ color: '#ab47bc', width: 1.5 }},
                    areaStyle: {{ color: 'rgba(171,71,188,0.15)' }}
                }}],
                dataZoom: [{{ type: 'inside', start: 0, end: 100 }}]
            }});
        }}

        // 统计卡片
        const lastNav = data[data.length - 1].strategy_nav || 1;
        const lastBhNav = data[data.length - 1].buy_hold_nav || 1;
        const totalReturn = ((lastNav - 1) * 100).toFixed(2);
        const bhReturn = ((lastBhNav - 1) * 100).toFixed(2);
        const buyCount = data.filter(d => d.signal === 1).length;
        const sellCount = data.filter(d => d.signal === -1).length;

        // 回撤计算
        const nav = data.map(d => d.strategy_nav || 1);
        let maxP = 0; let maxDD = 0;
        nav.forEach(v => {{ maxP = Math.max(maxP, v); maxDD = Math.min(maxDD, (v - maxP) / maxP * 100); }});

        document.getElementById('task4Stats').innerHTML = `
            <div class="stat-card"><div class="label">策略累计回报</div><div class="value ${{totalReturn > 0 ? 'up' : 'down'}}">${{totalReturn > 0 ? '+' : ''}}${{totalReturn}}%</div></div>
            <div class="stat-card"><div class="label">买入持有回报</div><div class="value ${{bhReturn > 0 ? 'up' : 'down'}}">${{bhReturn > 0 ? '+' : ''}}${{bhReturn}}%</div></div>
            <div class="stat-card"><div class="label">最大回撤</div><div class="value down">${{maxDD.toFixed(2)}}%</div></div>
            <div class="stat-card"><div class="label">买入次数</div><div class="value neutral">${{buyCount}}</div></div>
            <div class="stat-card"><div class="label">卖出次数</div><div class="value neutral">${{sellCount}}</div></div>
            <div class="stat-card"><div class="label">最新ATR</div><div class="value neutral">${{data[data.length-1].atr ? data[data.length-1].atr.toFixed(4) : 'N/A'}}</div></div>
        `;

        // 对比表
        const turComp = rawData.turtle_comparison || [];
        if (turComp.length) {{
            const tbl = document.getElementById('turtleTable');
            let html = '<thead><tr><th>股票</th><th>参数</th><th>策略回报(%)</th><th>买入持有(%)</th><th>夏普比率</th><th>最大回撤(%)</th><th>交易次数</th><th>胜率(%)</th></tr></thead><tbody>';
            turComp.forEach(r => {{
                html += `<tr><td>${{r.stock}}</td><td>${{r.config}}</td><td class="${{r.strategy_cum_return >= 0 ? 'up-text' : 'down-text'}}">${{r.strategy_cum_return.toFixed(2)}}</td><td>${{r.buy_hold_cum_return.toFixed(2)}}</td><td>${{r.strategy_sharpe.toFixed(2)}}</td><td class="down-text">${{r.strategy_mdd.toFixed(2)}}</td><td>${{r.n_trades}}</td><td>${{r.win_rate.toFixed(1)}}</td></tr>`;
            }});
            html += '</tbody>';
            tbl.innerHTML = html;
        }}
    }}

    // ===== 策略对比汇总 =====
    function renderSummary() {{
        const turComp = rawData.turtle_comparison || [];
        const maComp = rawData.ma_comparison || [];

        // 合并数据
        const allStrategies = [];
        maComp.forEach(r => allStrategies.push({{ name: `${{r.stock}}-MA${{r.config||r['参数']||''}}`, return: r.strategy_cum_return||r['策略回报']||0, sharpe: r.strategy_sharpe||r['夏普比率']||0, mdd: r.strategy_mdd||r['最大回撤']||0, winRate: r.win_rate||r['胜率']||0 }}));
        turComp.forEach(r => allStrategies.push({{ name: `${{r.stock}}-Turtle${{r.config}}`, return: r.strategy_cum_return, sharpe: r.strategy_sharpe, mdd: r.strategy_mdd, winRate: r.win_rate }}));

        const names = allStrategies.map(s => s.name);

        // 累计回报
        const retChart = initChart('chartCompareReturn');
        if (retChart) {{
            retChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                xAxis: {{ type: 'category', data: names, axisLabel: {{ rotate: 45, fontSize: 10 }} }},
                yAxis: {{ type: 'value' }},
                series: [{{
                    type: 'bar', data: allStrategies.map(s => ({{
                        value: s.return, itemStyle: {{ color: s.return >= 0 ? '#ef5350' : '#26a69a' }}
                    }}))
                }}]
            }});
        }}

        // 夏普比率
        const sharpeChart = initChart('chartCompareSharpe');
        if (sharpeChart) {{
            sharpeChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                xAxis: {{ type: 'category', data: names, axisLabel: {{ rotate: 45, fontSize: 10 }} }},
                yAxis: {{ type: 'value' }},
                series: [{{
                    type: 'bar', data: allStrategies.map(s => ({{
                        value: s.sharpe, itemStyle: {{ color: s.sharpe >= 0 ? '#ef5350' : '#26a69a' }}
                    }}))
                }}]
            }});
        }}

        // 最大回撤
        const mddChart = initChart('chartCompareMDD');
        if (mddChart) {{
            mddChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                xAxis: {{ type: 'category', data: names, axisLabel: {{ rotate: 45, fontSize: 10 }} }},
                yAxis: {{ type: 'value' }},
                series: [{{
                    type: 'bar', data: allStrategies.map(s => ({{
                        value: s.mdd, itemStyle: {{ color: '#26a69a' }}
                    }}))
                }}]
            }});
        }}

        // 胜率
        const wrChart = initChart('chartCompareWinRate');
        if (wrChart) {{
            wrChart.setOption({{
                backgroundColor: 'transparent',
                tooltip: {{ trigger: 'axis' }},
                xAxis: {{ type: 'category', data: names, axisLabel: {{ rotate: 45, fontSize: 10 }} }},
                yAxis: {{ type: 'value', max: 100 }},
                series: [{{
                    type: 'bar', data: allStrategies.map(s => ({{
                        value: s.winRate, itemStyle: {{ color: '#4fc3f7' }}
                    }}))
                }}]
            }});
        }}
    }}

    // 初始化
    window.addEventListener('load', () => {{
        renderTask1();
        renderTask2();
        renderTask3();
        renderTask4();
        renderSummary();
    }});

    window.addEventListener('resize', () => {{
        Object.values(charts).forEach(c => c && c.resize());
    }});
    </script>
</body>
</html>'''

    output_path = os.path.join(BASE_DIR, 'dashboard', 'index.html')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Dashboard已生成: {output_path}")
    return output_path


if __name__ == '__main__':
    generate_dashboard()
