# -*- coding: utf-8 -*-
"""
JoinQuant (聚宽) 平台策略代码
基于随机森林的机器学习交易策略
结合 TASK6 的因子体系和模型设计

作者: 童逸
说明: 将本代码完整复制到 JoinQuant 策略编辑器中运行
      策略类型: 股票策略
      回测周期: 日级别
      调仓频率: 每周一
      模型训练: 每月初
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from jqdata import *


# ============================================================
# 一、策略初始化
# ============================================================
def initialize(context):
    """策略初始化，设置参数和调度"""
    # --- 股票池 ---
    g.stocks = ['000001.XSHE', '600587.XSHG', '688146.XSHG']
    g.stock_names = {
        '000001.XSHE': '平安银行',
        '600587.XSHG': '天地科技',
        '688146.XSHG': '中船特气'
    }

    # --- 策略参数 ---
    g.lookback = 250           # 训练数据回看天数
    g.predict_horizon = 5      # 预测天数（未来5日涨跌方向）
    g.n_estimators = 100       # 随机森林树数量
    g.max_depth = 5            # 决策树最大深度
    g.min_samples_leaf = 10    # 叶节点最小样本数

    # --- 风控参数 ---
    g.stop_loss = 0.08         # 单股止损线 8%
    g.max_position = 0.35      # 单股最大仓位 35%
    g.min_confidence = 0.55    # 最低预测置信度

    # --- 模型存储 ---
    g.models = {}              # 各股票的模型
    g.scalers = {}             # 各股票的标准化器
    g.last_train_date = None   # 上次训练日期

    # --- 基准与交易成本 ---
    set_benchmark('000300.XSHG')  # 沪深300
    set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    set_slippage(FixedSlippage(0.02))

    # --- 调度任务 ---
    # 每月第一个交易日训练模型
    run_monthly(train_models, monthday=1, time='before_open')
    # 每周一开盘调仓
    run_weekly(rebalance, weekday=1, time='open')

    log.info('=== ML策略初始化完成 ===')
    log.info('股票池: %s' % g.stocks)
    log.info('参数: n_estimators=%d, max_depth=%d, predict_horizon=%d' % (
        g.n_estimators, g.max_depth, g.predict_horizon))


# ============================================================
# 二、技术指标计算（因子构建）
# ============================================================
def calc_rsi(close, period=14):
    """RSI 相对强弱指标"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD 指标"""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def calc_kdj(high, low, close, n=9, m1=3, m2=3):
    """KDJ 随机指标"""
    low_n = low.rolling(window=n, min_periods=n).min()
    high_n = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def compute_features(df):
    """计算全部技术因子，与TASK6保持一致"""
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

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
    df['ma_ratio'] = df['ma5'] / df['ma20']

    # 波动率因子
    df['bb_mid'] = c.rolling(20).mean()
    df['bb_std'] = c.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_pos'] = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['volatility_20'] = c.pct_change().rolling(20).std()

    # 量能因子
    df['vol_change'] = v.pct_change()
    df['vol_ma5'] = v.rolling(5).mean()
    df['vol_ratio'] = v / df['vol_ma5']

    # KDJ因子
    df['k'], df['d'], df['j'] = calc_kdj(h, l, c)

    # 应变量：未来N日收益率方向
    df['forward_ret'] = c.shift(-g.predict_horizon) / c - 1
    df['target'] = (df['forward_ret'] > 0).astype(int)

    return df


def get_feature_columns():
    """返回特征列名列表"""
    return [
        'rsi', 'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
        'dif', 'dea', 'macd_hist', 'ma_ratio',
        'bb_pos', 'volatility_20',
        'vol_change', 'vol_ratio',
        'k', 'd', 'j'
    ]


# ============================================================
# 三、模型训练
# ============================================================
def train_models(context):
    """每月初训练随机森林模型"""
    log.info('=== 开始训练模型 ===')
    today = context.current_dt.date()
    g.last_train_date = today

    feature_cols = get_feature_columns()

    for stock in g.stocks:
        # 获取历史数据
        df = get_price(
            stock,
            count=g.lookback + 60,  # 多取一些保证指标有效
            end_date=context.current_dt,
            frequency='daily',
            fields=['open', 'high', 'low', 'close', 'volume'],
            skip_paused=True,
            fq='pre',
            panel=False
        )

        if df is None or len(df) < 100:
            log.warning('%s 数据不足，跳过训练' % g.stock_names[stock])
            continue

        # 计算因子
        df = compute_features(df)
        df = df.dropna(subset=feature_cols + ['target'])

        if len(df) < 50:
            log.warning('%s 有效样本不足，跳过训练' % g.stock_names[stock])
            continue

        # 准备训练数据
        X = df[feature_cols].values
        y = df['target'].values

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 训练随机森林
        model = RandomForestClassifier(
            n_estimators=g.n_estimators,
            max_depth=g.max_depth,
            min_samples_leaf=g.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled, y)

        # 存储模型
        g.models[stock] = model
        g.scalers[stock] = scaler

        # 训练集准确率
        train_acc = model.score(X_scaled, y)
        log.info('%s 模型训练完成: 样本=%d, 训练准确率=%.4f' % (
            g.stock_names[stock], len(df), train_acc))

    log.info('=== 模型训练完成 ===')


# ============================================================
# 四、调仓执行
# ============================================================
def rebalance(context):
    """每周一根据模型预测调仓"""
    if not g.models:
        log.warning('模型尚未训练，跳过调仓')
        return

    feature_cols = get_feature_columns()
    total_weight = 0.0

    for stock in g.stocks:
        if stock not in g.models:
            continue

        # 获取最新数据计算因子
        df = get_price(
            stock,
            count=80,
            end_date=context.current_dt,
            frequency='daily',
            fields=['open', 'high', 'low', 'close', 'volume'],
            skip_paused=True,
            fq='pre',
            panel=False
        )

        if df is None or len(df) < 70:
            continue

        df = compute_features(df)
        latest = df.iloc[-1]

        # 检查因子完整性
        features = latest[feature_cols].values
        if np.isnan(features).any():
            log.warning('%s 因子含缺失值，跳过' % g.stock_names[stock])
            continue

        # 预测
        features_scaled = g.scalers[stock].transform([features])
        proba = g.models[stock].predict_proba(features_scaled)[0]
        pred_class = int(np.argmax(proba))
        confidence = max(proba)

        # 止损检查
        current_price = latest['close']
        position = context.portfolio.positions.get(stock)
        stop_loss_triggered = False

        if position and position.total_amount > 0:
            cost = position.avg_cost
            if current_price < cost * (1 - g.stop_loss):
                log.warning('%s 触发止损 (%.2f -> %.2f, 亏损%.1f%%)' % (
                    g.stock_names[stock], cost, current_price,
                    (current_price / cost - 1) * 100))
                stop_loss_triggered = True

        # 交易决策
        if stop_loss_triggered:
            target_weight = 0.0
            log.info('%s: 止损清仓' % g.stock_names[stock])
        elif pred_class == 1 and confidence >= g.min_confidence:
            target_weight = g.max_position * min(1.0, confidence / 0.7)
            log.info('%s: 预测上涨 (置信度=%.3f) -> 目标仓位=%.1f%%' % (
                g.stock_names[stock], confidence, target_weight * 100))
        else:
            target_weight = 0.0
            if pred_class == 0:
                log.info('%s: 预测下跌 (置信度=%.3f) -> 清仓' % (
                    g.stock_names[stock], confidence))
            else:
                log.info('%s: 置信度不足 (%.3f < %.3f) -> 观望' % (
                    g.stock_names[stock], confidence, g.min_confidence))

        # 执行调仓
        order_target_value(stock, context.portfolio.total_value * target_weight)
        total_weight += target_weight

    # 空仓时买入货币基金
    if total_weight < 0.01:
        log.info('全部空仓，持有现金等待机会')
    else:
        log.info('总仓位: %.1f%%' % (total_weight * 100))


# ============================================================
# 五、盘前处理
# ============================================================
def before_trading_start(context):
    """每日开盘前记录持仓状态"""
    total_value = context.portfolio.total_value
    positions_value = context.portfolio.positions_value

    if positions_value > 0:
        for stock in g.stocks:
            pos = context.portfolio.positions.get(stock)
            if pos and pos.total_amount > 0:
                pnl = (pos.price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0
                # 仅在有持仓时记录
                pass


# ============================================================
# 策略说明
# ============================================================
# 1. 本策略基于随机森林分类模型，预测股票未来5日涨跌方向
# 2. 因子体系包含16个技术指标：动量(RSI/收益率)、趋势(MACD/均线比)、
#    波动率(布林带位置/标准差)、量能(量比/成交量变化)、超买超卖(KDJ)
# 3. 每月初重新训练模型，使用过去250个交易日数据
# 4. 每周一根据模型预测调仓，预测上涨则建仓，预测下跌则清仓
# 5. 风控措施：单股止损8%、最大仓位35%、最低置信度阈值55%
#
# 参数调优建议：
# - n_estimators: 50 / 100 / 200（树越多越稳定但训练慢）
# - max_depth: 3 / 5 / 10（越深越容易过拟合）
# - predict_horizon: 3 / 5 / 10（预测周期影响信号频率）
# - min_confidence: 0.50 / 0.55 / 0.60（越高越保守）
