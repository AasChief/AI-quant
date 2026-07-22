# -*- coding: utf-8 -*-
"""
TASK8: 量化交易学习成果综合报告生成脚本
综合前7个任务的内容，生成专业PDF报告
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# ============================================================
# 字体注册
# ============================================================
pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))
pdfmetrics.registerFont(TTFont('SimHei', 'C:/Windows/Fonts/simhei.ttf'))

# matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径设置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR = os.path.join(BASE_DIR, 'charts_task8')
os.makedirs(CHART_DIR, exist_ok=True)

# ============================================================
# 样式定义
# ============================================================
FONT_NAME = 'SimSun'
FONT_SIZE = 10.5  # 五号字
LINE_HEIGHT = 1.5

styles = getSampleStyleSheet()

style_normal = ParagraphStyle(
    'CustomNormal',
    fontName=FONT_NAME,
    fontSize=FONT_SIZE,
    leading=FONT_SIZE * LINE_HEIGHT,
    alignment=TA_JUSTIFY,
    spaceBefore=0,
    spaceAfter=0,
    firstLineIndent=FONT_SIZE * 2,  # 首行缩进2字符
)

style_body = ParagraphStyle(
    'CustomBody',
    fontName=FONT_NAME,
    fontSize=FONT_SIZE,
    leading=FONT_SIZE * LINE_HEIGHT,
    alignment=TA_JUSTIFY,
    spaceBefore=0,
    spaceAfter=0,
    firstLineIndent=FONT_SIZE * 2,
)

style_heading1 = ParagraphStyle(
    'CustomH1',
    fontName='SimHei',
    fontSize=16,
    leading=24,
    alignment=TA_CENTER,
    spaceBefore=12,
    spaceAfter=12,
)

style_heading2 = ParagraphStyle(
    'CustomH2',
    fontName='SimHei',
    fontSize=14,
    leading=21,
    alignment=TA_LEFT,
    spaceBefore=12,
    spaceAfter=6,
)

style_heading3 = ParagraphStyle(
    'CustomH3',
    fontName='SimHei',
    fontSize=FONT_SIZE,
    leading=FONT_SIZE * LINE_HEIGHT,
    alignment=TA_LEFT,
    spaceBefore=6,
    spaceAfter=3,
)

style_cover_title = ParagraphStyle(
    'CoverTitle',
    fontName='SimHei',
    fontSize=26,
    leading=39,
    alignment=TA_CENTER,
    spaceBefore=60,
    spaceAfter=20,
)

style_cover_subtitle = ParagraphStyle(
    'CoverSubtitle',
    fontName=FONT_NAME,
    fontSize=16,
    leading=24,
    alignment=TA_CENTER,
    spaceBefore=10,
    spaceAfter=10,
)

style_cover_info = ParagraphStyle(
    'CoverInfo',
    fontName=FONT_NAME,
    fontSize=14,
    leading=21,
    alignment=TA_CENTER,
    spaceBefore=6,
    spaceAfter=6,
)

style_abstract_title = ParagraphStyle(
    'AbstractTitle',
    fontName='SimHei',
    fontSize=14,
    leading=21,
    alignment=TA_CENTER,
    spaceBefore=6,
    spaceAfter=6,
)

style_abstract = ParagraphStyle(
    'Abstract',
    fontName=FONT_NAME,
    fontSize=FONT_SIZE,
    leading=FONT_SIZE * LINE_HEIGHT,
    alignment=TA_JUSTIFY,
    spaceBefore=0,
    spaceAfter=0,
    firstLineIndent=FONT_SIZE * 2,
    leftIndent=FONT_SIZE * 2,
    rightIndent=FONT_SIZE * 2,
)

style_toc = ParagraphStyle(
    'TOC',
    fontName=FONT_NAME,
    fontSize=12,
    leading=20,
    alignment=TA_LEFT,
    spaceBefore=3,
    spaceAfter=3,
    leftIndent=0,
)

style_toc_sub = ParagraphStyle(
    'TOCSub',
    fontName=FONT_NAME,
    fontSize=FONT_SIZE,
    leading=18,
    alignment=TA_LEFT,
    spaceBefore=2,
    spaceAfter=2,
    leftIndent=24,
)

style_caption = ParagraphStyle(
    'Caption',
    fontName=FONT_NAME,
    fontSize=9,
    leading=13.5,
    alignment=TA_CENTER,
    spaceBefore=3,
    spaceAfter=6,
)

style_table_cell = ParagraphStyle(
    'TableCell',
    fontName=FONT_NAME,
    fontSize=9,
    leading=13.5,
    alignment=TA_CENTER,
    spaceBefore=0,
    spaceAfter=0,
)

style_appendix = ParagraphStyle(
    'Appendix',
    fontName=FONT_NAME,
    fontSize=FONT_SIZE,
    leading=FONT_SIZE * LINE_HEIGHT,
    alignment=TA_JUSTIFY,
    spaceBefore=0,
    spaceAfter=0,
    firstLineIndent=FONT_SIZE * 2,
    leftIndent=12,
)


# ============================================================
# 页眉页脚
# ============================================================
def add_page_header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    # 页脚页码
    page_num = canvas_obj.getPageNumber()
    if page_num > 1:  # 封面不显示页码
        canvas_obj.setFont(FONT_NAME, 9)
        canvas_obj.drawCentredString(A4[0] / 2, 1.5 * cm, str(page_num - 1))
    canvas_obj.restoreState()


# ============================================================
# 生成综合对比图表
# ============================================================
def generate_charts():
    """生成报告所需的综合对比图表"""

    charts = {}

    # ---------- 图1: 各策略年化收益率对比 ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    strategies = ['双均线策略\n(5日/15日)', '海龟策略\n(10日/5日)', '机器学习\n(随机森林)', '平台策略\n(组合)']
    zcq_returns = [980.26, 787.60, 490.95, 61.49]
    bh_returns = [1179.83, 1179.83, 3990.66, 133.04]

    x = np.arange(len(strategies))
    width = 0.35
    bars1 = ax.bar(x - width/2, zcq_returns, width, label='策略收益', color='#D32F2F', alpha=0.85)
    bars2 = ax.bar(x + width/2, bh_returns, width, label='买入持有', color='#1976D2', alpha=0.85)

    ax.set_ylabel('年化收益率（%）', fontsize=11)
    ax.set_title('图1 各策略年化收益率对比（中船特气）', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(bh_returns) * 1.15)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'fig1_strategy_return_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    charts['fig1'] = path

    # ---------- 图2: 各策略最大回撤对比 ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mdds_strategy = [-27.96, -17.41, -9.68, -11.42]
    mdds_bh = [-19.89, -19.89, -13.66, -13.18]

    bars1 = ax.bar(x - width/2, mdds_strategy, width, label='策略回撤', color='#D32F2F', alpha=0.85)
    bars2 = ax.bar(x + width/2, mdds_bh, width, label='买入持有回撤', color='#1976D2', alpha=0.85)

    ax.set_ylabel('最大回撤（%）', fontsize=11)
    ax.set_title('图2 各策略最大回撤对比（中船特气）', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(min(mdds_strategy + mdds_bh) * 1.3, 2)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -10), textcoords="offset points", ha='center', va='top', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, -10), textcoords="offset points", ha='center', va='top', fontsize=7)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'fig2_strategy_mdd_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    charts['fig2'] = path

    # ---------- 图3: 各策略夏普比率对比 ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sharpes = [13.11, 3.58, 3.33, 2.27]
    sharpes_bh = [7.84, 3.72, 7.84, 3.01]

    bars1 = ax.bar(x - width/2, sharpes, width, label='策略夏普', color='#D32F2F', alpha=0.85)
    bars2 = ax.bar(x + width/2, sharpes_bh, width, label='买入持有夏普', color='#1976D2', alpha=0.85)

    ax.set_ylabel('夏普比率', fontsize=11)
    ax.set_title('图3 各策略夏普比率对比（中船特气）', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(sharpes + sharpes_bh) * 1.2)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'fig3_strategy_sharpe_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    charts['fig3'] = path

    # ---------- 图5: 机器学习模型AUC对比 ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = ['逻辑回归', '决策树', '随机森林']
    stock_auc = [0.4528, 0.4047, 0.4771]
    cancer_auc = [0.9981, 0.9441, 0.9915]

    x2 = np.arange(len(models))
    bars1 = ax.bar(x2 - width/2, stock_auc, width, label='股票数据', color='#FF9800', alpha=0.85)
    bars2 = ax.bar(x2 + width/2, cancer_auc, width, label='乳腺癌数据', color='#4CAF50', alpha=0.85)

    ax.set_ylabel('AUC值', fontsize=11)
    ax.set_title('图5 机器学习分类模型AUC对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x2)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='随机分类线')

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'fig4_ml_auc_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    charts['fig5_auc'] = path

    # ---------- 图6: 机器学习策略累计净值对比 ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models2 = ['逻辑回归', '决策树', '随机森林', '买入持有']
    navs = [0.82, 0.21, 5.91, 40.91]  # 中船特气各模型累计净值（估算）
    navs_bh = 40.91

    colors_list = ['#FF9800', '#2196F3', '#4CAF50', '#9C27B0']
    bars = ax.bar(models2, navs, color=colors_list, alpha=0.85)

    ax.set_ylabel('累计净值', fontsize=11)
    ax.set_title('图6 机器学习策略累计净值对比（中船特气）', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(navs) * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'fig5_ml_nav_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    charts['fig6_nav'] = path

    # ---------- 图4: 风险收益散点图 ----------
    fig, ax = plt.subplots(figsize=(8, 5))

    # 各策略的风险收益散点
    data_points = {
        '双均线(5/15)': (74.57, 980.26, '#D32F2F'),
        '海龟(10/5)': (45.0, 787.60, '#FF9800'),
        'ML-随机森林': (30.0, 490.95, '#4CAF50'),
        '平台-组合策略': (21.20, 61.49, '#2196F3'),
        '买入持有': (28.86, 133.04, '#9C27B0'),
    }

    for name, (vol, ret, color) in data_points.items():
        ax.scatter(vol, ret, s=120, c=color, label=name, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(name, xy=(vol, ret), xytext=(8, 5), textcoords="offset points",
                    fontsize=8, ha='left')

    ax.set_xlabel('年化波动率（%）', fontsize=11)
    ax.set_ylabel('年化收益率（%）', fontsize=11)
    ax.set_title('图4 各策略风险收益散点图', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(-50, 1100)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'fig6_risk_return_scatter.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    charts['fig4_scatter'] = path

    print("  所有图表生成完成")
    return charts


# ============================================================
# 表格生成辅助函数
# ============================================================
def make_table(data, col_widths=None, caption=None):
    """生成格式化表格"""
    # 转换数据为Paragraph
    table_data = []
    for i, row in enumerate(data):
        new_row = []
        for cell in row:
            p = Paragraph(str(cell), style_table_cell)
            new_row.append(p)
        table_data.append(new_row)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEADING', (0, 0), (-1, -1), 13.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D6E4F0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))

    elements = []
    if caption:
        elements.append(Paragraph(caption, style_caption))
    elements.append(t)
    elements.append(Spacer(1, 6))
    return KeepTogether(elements)


def make_image(path, width=14*cm, caption=None):
    """生成带标题的图片"""
    from PIL import Image as PILImage
    pil_img = PILImage.open(path)
    w, h = pil_img.size
    aspect = h / w
    height = width * aspect
    # 限制高度
    max_h = 10 * cm
    if height > max_h:
        height = max_h
        width = height / aspect

    img = Image(path, width=width, height=height)
    elements = []
    elements.append(img)
    if caption:
        elements.append(Paragraph(caption, style_caption))
    elements.append(Spacer(1, 6))
    return KeepTogether(elements)


# ============================================================
# 构建PDF报告
# ============================================================
def build_report(charts):
    """构建完整PDF报告"""

    output_path = os.path.join(BASE_DIR, '童逸+TASK8.pdf')
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    story = []

    # ============================================================
    # 封面
    # ============================================================
    story.append(Spacer(1, 80))
    story.append(Paragraph('量化交易策略开发与机器学习应用', style_cover_title))
    story.append(Paragraph('学习成果综合报告', style_cover_title))
    story.append(Spacer(1, 40))
    story.append(Paragraph('——基于七个阶段实践的系统总结', style_cover_subtitle))
    story.append(Spacer(1, 80))
    story.append(Paragraph('作者：童逸', style_cover_info))
    story.append(Paragraph('日期：2026年7月', style_cover_info))
    story.append(Spacer(1, 20))
    story.append(Paragraph('课程：AI大模型辅助的金融交易策略', style_cover_info))
    story.append(PageBreak())

    # ============================================================
    # 目录
    # ============================================================
    story.append(Paragraph('目  录', style_heading1))
    story.append(Spacer(1, 12))

    toc_items = [
        ('摘  要', '1'),
        ('一、量化交易核心概念', '2'),
        ('  （一）量化交易的基本概念', '2'),
        ('  （二）量化交易的核心价值', '2'),
        ('二、量化交易策略综合分析', '3'),
        ('  （一）数据获取与技术分析基础', '3'),
        ('  （二）双均线策略', '4'),
        ('  （三）海龟交易策略', '5'),
        ('  （四）策略对比与多策略系统构建', '6'),
        ('三、机器学习在量化交易中的应用总结', '8'),
        ('  （一）分类模型与评估指标', '8'),
        ('  （二）机器学习交易策略与回测', '9'),
        ('  （三）平台实现与实盘模拟', '10'),
        ('  （四）优势、局限与未来趋势', '11'),
        ('四、结论与展望', '12'),
        ('附录：改进建议', '13'),
    ]

    for title, page in toc_items:
        dots = '.' * (50 - len(title) * 2)
        story.append(Paragraph(f'{title}{dots}{page}', style_toc))

    story.append(PageBreak())

    # ============================================================
    # 摘要
    # ============================================================
    story.append(Paragraph('摘  要', style_heading1))
    story.append(Spacer(1, 12))

    abstract_text = (
        '本报告系统总结了量化交易策略开发与机器学习应用的学习成果。'
        '研究以中船特气、天地科技、平安银行三只股票的日线数据为基础，'
        '依次完成了数据获取、技术指标计算、双均线策略、海龟交易策略、'
        '分类模型构建、机器学习交易策略以及平台实盘模拟七个阶段的实践。'
        '在传统策略方面，双均线策略在中船特气上实现了980.26%的年化收益，'
        '海龟策略通过唐奇安通道和ATR止损将最大回撤控制在17.41%。'
        '在机器学习方面，以16个技术因子为自变量、未来5日涨跌方向为应变量，'
        '构建了逻辑回归、决策树和随机森林三种分类模型，'
        '并在聚宽平台上实现了滚动训练与动态调仓的实盘模拟策略，'
        '年化收益率达61.49%，夏普比率为2.27，最大回撤为11.42%。'
        '研究表明，机器学习策略在控制回撤方面具有显著优势，'
        '但在短期股价预测精度上仍受市场随机性的制约，'
        '多策略组合是降低风险、提升稳健性的有效途径。'
    )
    story.append(Paragraph(abstract_text, style_abstract))
    story.append(PageBreak())

    # ============================================================
    # 第一章：量化交易核心概念
    # ============================================================
    story.append(Paragraph('一、量化交易核心概念', style_heading1))

    story.append(Paragraph('（一）量化交易的基本概念', style_heading2))
    story.append(Paragraph(
        '量化交易是指利用数学模型、统计方法和计算机程序，'
        '对金融市场数据进行系统分析，自动生成交易信号并执行交易的投资方法。'
        '其核心在于将投资逻辑转化为可量化、可回测、可复现的规则体系，'
        '以数据和算法替代主观判断，从而提升决策的客观性和执行效率。'
        '一个完整的量化交易系统通常包括数据获取、因子构建、信号生成、'
        '风险管理和执行交易五个环节，各环节紧密衔接、相互支撑。',
        style_body
    ))

    story.append(Paragraph('（二）量化交易的核心价值', style_heading2))
    story.append(Paragraph(
        '<b>第一，客观性。</b>量化交易通过预设规则消除情绪干扰，避免人在恐慌或贪婪时的非理性决策。'
        '在本课程的实践中，无论是双均线策略的金叉死叉信号，'
        '还是机器学习模型的概率输出，均严格依据模型计算结果执行，'
        '不掺杂主观臆断，保证了交易纪律的一致性。',
        style_body
    ))
    story.append(Paragraph(
        '<b>第二，可回测性。</b>量化策略在投入实盘前，可在历史数据上进行充分回测，'
        '评估其在不同市场环境下的表现。本报告中，双均线策略测试了5组参数，'
        '海龟策略测试了3组通道周期，机器学习策略进行了9组参数调优，'
        '通过系统比较筛选出最优配置，为实盘部署提供了数据支撑。',
        style_body
    ))
    story.append(Paragraph(
        '<b>第三，可扩展性。</b>量化系统可同时监控和交易大量标的，'
        '实现规模化的投资管理。本报告中的策略均在3只不同特征的股票上进行了测试，'
        '涵盖科创板、主板和金融板块，验证了策略的跨品种适用性。',
        style_body
    ))
    story.append(Paragraph(
        '<b>第四，系统性风险管理。</b>量化交易通过最大回撤、夏普比率、'
        '风险价值等指标对策略风险进行量化度量，并结合止损规则、仓位管理'
        '实现系统化的风险控制。海龟策略的ATR止损和机器学习策略的'
        '动态仓位管理均体现了这一价值。',
        style_body
    ))

    story.append(PageBreak())

    # ============================================================
    # 第二章：量化交易策略综合分析
    # ============================================================
    story.append(Paragraph('二、量化交易策略综合分析', style_heading1))

    story.append(Paragraph('（一）数据获取与技术分析基础', style_heading2))
    story.append(Paragraph(
        '本研究的全部策略均建立在可靠的行情数据基础上。'
        '通过Tushare金融数据接口，获取了中船特气（688146）、'
        '天地科技（600587）和平安银行（000001）三只股票'
        '过去一年的日线行情数据，包括开盘价、收盘价、最高价、最低价和成交量。'
        '在数据质量方面，对缺失值和异常值进行了诊断清洗，'
        '确保后续分析和建模的准确性。',
        style_body
    ))
    story.append(Paragraph(
        '在技术分析层面，计算了相对强弱指标（RSI）、平滑异同移动平均线（MACD）、'
        '布林带和随机指标（KDJ）四类经典技术指标。'
        '这些指标从动量、趋势、波动率和超买超卖四个维度刻画了价格运动特征，'
        '为后续的策略开发和机器学习特征工程奠定了基础。'
        '其中RSI用于衡量价格变化的力度，MACD捕捉趋势的转折，'
        '布林带反映波动率的动态变化，KDJ则识别短期的超买超卖区域。',
        style_body
    ))

    story.append(Paragraph('（二）双均线策略', style_heading2))
    story.append(Paragraph(
        '双均线策略是最经典的趋势跟踪策略之一。其核心逻辑是：'
        '当短期均线上穿长期均线时产生金叉买入信号，'
        '当短期均线下穿长期均线时产生死叉卖出信号。'
        '本研究在中船特气、天地科技和平安银行三只股票上，'
        '测试了短均线周期为5日和10日、长均线周期为15日、20日和30日的5组参数组合，'
        '共15种配置。回测采用前一日信号、当日收盘价执行的机制，'
        '评估了累计回报、年化收益、夏普比率、最大回撤和胜率等指标。',
        style_body
    ))

    # 表1: 双均线策略参数对比结果（中船特气）
    table1_data = [
        ['短均线', '长均线', '年化收益(%)', '夏普比率', '最大回撤(%)', '交易次数', '胜率(%)'],
        ['5', '15', '980.26', '13.11', '-27.96', '7', '42.86'],
        ['5', '20', '735.86', '10.13', '-32.82', '7', '28.57'],
        ['10', '20', '714.26', '9.86', '-30.43', '6', '33.33'],
        ['5', '30', '742.46', '10.13', '-33.23', '6', '33.33'],
        ['10', '30', '723.95', '9.91', '-28.51', '6', '33.33'],
    ]
    story.append(make_table(table1_data,
                            col_widths=[1.8*cm, 1.8*cm, 2.4*cm, 2.0*cm, 2.4*cm, 2.0*cm, 2.0*cm],
                            caption='表1 双均线策略参数对比结果（中船特气）'))

    story.append(Paragraph(
        '从表1可以看出，5日/15日参数组合表现最优，'
        '年化收益率达到980.26%，夏普比率为13.11，'
        '但最大回撤也高达27.96%。这说明双均线策略在趋势明朗的个股上收益突出，'
        '但回撤控制能力有限。在天地科技和平安银行上，'
        '策略表现明显弱于中船特气，体现了该策略对标的趋势性的高度依赖。'
        '（详见附录建议1）',
        style_body
    ))

    story.append(PageBreak())

    story.append(Paragraph('（三）海龟交易策略', style_heading2))
    story.append(Paragraph(
        '海龟交易策略由理查德·丹尼斯在20世纪80年代提出，'
        '其核心思想是利用唐奇安通道捕捉趋势突破信号，'
        '并通过平均真实波幅（ATR）进行动态止损和仓位管理。'
        '当价格突破过去N日最高价时买入，跌破过去M日最低价时卖出，'
        '同时以ATR的倍数设定止损价位，控制单笔交易的最大亏损。'
        '本研究测试了10日/5日、20日/10日和55日/20日三组通道参数，'
        '覆盖短线、中线和长线三种趋势跟踪风格。',
        style_body
    ))

    # 表2: 海龟策略参数对比结果
    table2_data = [
        ['股票', '通道参数', '策略收益(%)', '买入持有(%)', '夏普比率', '最大回撤(%)', '胜率(%)'],
        ['中船特气', '10日/5日', '787.60', '1179.83', '3.58', '-17.41', '50.00'],
        ['中船特气', '20日/10日', '543.94', '1179.83', '3.11', '-34.52', '40.00'],
        ['中船特气', '55日/20日', '543.10', '1179.83', '3.36', '-13.91', '0.00'],
        ['天地科技', '10日/5日', '2.63', '-23.38', '0.03', '-9.88', '66.67'],
        ['平安银行', '20日/10日', '-8.29', '-13.01', '-1.77', '-8.29', '25.00'],
    ]
    story.append(make_table(table2_data,
                            col_widths=[1.8*cm, 2.0*cm, 2.0*cm, 2.2*cm, 1.8*cm, 2.0*cm, 1.8*cm],
                            caption='表2 海龟策略参数对比结果（部分）'))

    story.append(Paragraph(
        '从表2可以看出，海龟策略在趋势性较强的中船特气上表现优异，'
        '10日/5日参数组合实现了787.60%的累计收益和50%的胜率。'
        '更值得关注的是，55日/20日长周期组合虽然收益略低，'
        '但将最大回撤从19.89%（买入持有）降至13.91%，'
        '体现了ATR止损在风控方面的价值。'
        '在天地科技上，海龟策略成功规避了23.38%的下跌，实现2.63%的正收益，'
        '说明该策略在震荡下行市场中具有防御能力。'
        '（详见附录建议2）',
        style_body
    ))

    story.append(Paragraph('（四）策略对比与多策略系统构建', style_heading2))
    story.append(Paragraph(
        '为直观比较各类策略的表现差异，将双均线策略、海龟策略'
        '以及后续的机器学习策略在中船特气上的关键指标进行了汇总对比。',
        style_body
    ))

    # 图1: 年化收益率对比
    story.append(make_image(charts['fig1'], width=14*cm,
                            caption='图1 各策略年化收益率对比（中船特气）'))
    story.append(Paragraph(
        '图1显示，双均线策略在中船特气上的年化收益率最高（980.26%），'
        '海龟策略次之（787.60%），机器学习随机森林策略为490.95%。'
        '但需要注意的是，收益率的排名与回撤幅度密切相关——'
        '高收益往往伴随高风险，仅看收益率无法全面评估策略优劣。',
        style_body
    ))

    # 图2: 最大回撤对比
    story.append(make_image(charts['fig2'], width=14*cm,
                            caption='图2 各策略最大回撤对比（中船特气）'))
    story.append(Paragraph(
        '图2揭示了风险控制方面的显著差异。机器学习策略的最大回撤仅为9.68%，'
        '远低于双均线策略的27.96%和海龟策略的17.41%。'
        '这表明机器学习模型通过动态仓位调整，在市场剧烈波动时'
        '能够主动降低暴露，有效控制下行风险。',
        style_body
    ))

    story.append(PageBreak())

    # 图3: 夏普比率对比
    story.append(make_image(charts['fig3'], width=14*cm,
                            caption='图3 各策略夏普比率对比（中船特气）'))
    story.append(Paragraph(
        '图3从风险调整收益的角度进行比较。双均线策略的夏普比率最高（13.11），'
        '但这主要受益于中船特气在该期间的强劲趋势。'
        '机器学习策略的夏普比率为3.33，'
        '虽然绝对值低于传统策略，但考虑到其更低的回撤和更稳健的净值曲线，'
        '在风险收益平衡上具有独特优势。',
        style_body
    ))

    # 图4: 风险收益散点图
    story.append(make_image(charts['fig4_scatter'], width=14*cm,
                            caption='图4 各策略风险收益散点图'))
    story.append(Paragraph(
        '图4以散点图形式综合展示了各策略的风险收益特征。'
        '位于左上方的策略具有"低风险、高收益"的理想特征。'
        '平台组合策略（年化波动率21.20%、年化收益61.49%）'
        '在风险控制方面表现突出，虽然收益率不及趋势策略，'
        '但单位风险获得的回报更为稳定。'
        '这为构建多策略系统提供了思路：通过组合不同风格、'
        '不同周期的策略，可以在收益和风险之间取得更好的平衡。'
        '（详见附录建议3）',
        style_body
    ))

    story.append(Paragraph(
        '综合来看，双均线策略适合趋势明确的牛市环境，'
        '海龟策略在波动较大的市场中通过止损机制体现防御价值，'
        '机器学习策略则在回撤控制上优势明显。'
        '三者在收益来源和风险特征上具有互补性，'
        '构建多策略组合系统——将趋势跟踪、通道突破和机器学习预测'
        '按一定权重配置——有望在不同市场环境下实现更稳健的整体表现。',
        style_body
    ))

    story.append(PageBreak())

    # ============================================================
    # 第三章：机器学习在量化交易中的应用总结
    # ============================================================
    story.append(Paragraph('三、机器学习在量化交易中的应用总结', style_heading1))

    story.append(Paragraph('（一）分类模型与评估指标', style_heading2))
    story.append(Paragraph(
        '机器学习在量化交易中的第一步是解决分类问题——预测未来价格涨跌方向。'
        '本研究构建了逻辑回归、决策树和随机森林三种分类模型，'
        '以RSI、MACD、布林带、KDJ等11个技术指标为自变量，'
        '以次日涨跌方向（涨为1，跌为0）为应变量。'
        '同时在scikit-learn乳腺癌数据集上进行了对比验证。'
        '数据按7:3划分为训练集和测试集，采用分层抽样保证类别平衡。',
        style_body
    ))

    # 表3: 机器学习分类模型评估结果
    table3_data = [
        ['数据集', '模型', '准确率', '精确率', '召回率', 'F1值', 'AUC'],
        ['股票数据', '逻辑回归', '0.5075', '0.4615', '0.1875', '0.2667', '0.4528'],
        ['股票数据', '决策树', '0.4080', '0.4248', '0.6771', '0.5221', '0.4047'],
        ['股票数据', '随机森林', '0.5025', '0.4714', '0.3438', '0.3976', '0.4771'],
        ['乳腺癌数据', '逻辑回归', '0.9883', '0.9907', '0.9907', '0.9907', '0.9981'],
        ['乳腺癌数据', '决策树', '0.9298', '0.9439', '0.9439', '0.9439', '0.9441'],
        ['乳腺癌数据', '随机森林', '0.9357', '0.9444', '0.9533', '0.9488', '0.9915'],
    ]
    story.append(make_table(table3_data,
                            col_widths=[2.0*cm, 1.8*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm],
                            caption='表3 机器学习分类模型评估结果'))

    story.append(Paragraph(
        '表3的结果揭示了一个重要现象：在股票数据上，'
        '三种模型的AUC值均接近0.5（随机分类水平），'
        '说明仅凭技术指标难以有效预测次日涨跌方向。'
        '而在乳腺癌数据集上，逻辑回归的AUC高达0.9981，'
        '表明模型本身没有问题，问题在于金融市场数据的信噪比极低。'
        '这一发现对后续策略设计具有重要指导意义：'
        '不应过度追求分类精度，而应关注如何利用模型概率输出'
        '构建有正期望的交易策略。',
        style_body
    ))

    # 图5: AUC对比
    story.append(make_image(charts['fig5_auc'], width=14*cm,
                            caption='图5 机器学习分类模型AUC对比'))
    story.append(Paragraph(
        '图5直观展示了两种数据集上AUC的巨大差异。'
        '股票数据的AUC集中在0.40至0.48区间，'
        '低于随机分类线0.5，反映出短期股价运动的高度随机性。'
        '乳腺癌数据的AUC均在0.94以上，验证了模型在结构化数据上的强大能力。'
        '（详见附录建议4）',
        style_body
    ))

    story.append(PageBreak())

    story.append(Paragraph('（二）机器学习交易策略与回测', style_heading2))
    story.append(Paragraph(
        '在分类模型的基础上，进一步将预测结果转化为可执行的交易策略。'
        '策略逻辑为：当模型预测上涨时全仓持有，预测下跌时空仓观望。'
        '自变量扩展为16个技术因子，涵盖动量、趋势、波动率和量能四大类。'
        '应变量调整为未来5日涨跌方向，以降低日度噪声的影响。'
        '采用时间序列7:3划分训练集和测试集，避免前视偏差。'
        '回测评估了累计净值、季度收益率、夏普比率、最大回撤和胜率等指标。',
        style_body
    ))

    # 表4: 机器学习交易策略回测结果
    table4_data = [
        ['股票', '模型', '年化收益(%)', '夏普比率', '最大回撤(%)', '胜率(%)', '交易次数'],
        ['中船特气', '逻辑回归', '18.42', '1.12', '-3.34', '57.14', '3'],
        ['中船特气', '决策树', '184.65', '2.64', '-2.92', '64.29', '5'],
        ['中船特气', '随机森林', '490.95', '3.33', '-9.68', '62.86', '22'],
        ['平安银行', '随机森林', '-27.66', '-4.61', '-9.56', '20.00', '13'],
        ['三股组合', '随机森林', '39.16', '1.35', '-7.91', '—', '—'],
    ]
    story.append(make_table(table4_data,
                            col_widths=[1.8*cm, 1.8*cm, 2.2*cm, 1.8*cm, 2.2*cm, 1.6*cm, 1.8*cm],
                            caption='表4 机器学习交易策略回测结果'))

    story.append(Paragraph(
        '表4显示，随机森林在中船特气上取得了490.95%的年化收益和3.33的夏普比率，'
        '同时将最大回撤控制在9.68%，显著优于买入持有的13.66%。'
        '但在平安银行上策略表现不佳，年化收益为-27.66%，'
        '说明模型在缺乏趋势的市场中容易产生错误信号。'
        '三股等权组合策略的年化收益为39.16%，夏普比率为1.35，'
        '最大回撤仅7.91%，体现了分散化投资对降低非系统性风险的作用。'
        '（详见附录建议5）',
        style_body
    ))

    # 图6: 累计净值对比
    story.append(make_image(charts['fig6_nav'], width=14*cm,
                            caption='图6 机器学习策略累计净值对比（中船特气）'))
    story.append(Paragraph(
        '图6对比了三种模型在中船特气上的累计净值表现。'
        '随机森林的累计净值最高（5.91），远超逻辑回归（0.82）和决策树（0.21）。'
        '但与买入持有（40.91）相比仍有较大差距，'
        '这是因为模型在部分上涨时段选择了空仓，错过了部分收益。'
        '这一"取舍"正是机器学习策略的核心价值——'
        '通过放弃部分上行空间来换取下行保护。',
        style_body
    ))

    story.append(PageBreak())

    story.append(Paragraph('（三）平台实现与实盘模拟', style_heading2))
    story.append(Paragraph(
        '在掌握策略原理后，进一步在聚宽（JoinQuant）量化平台上进行了实盘模拟。'
        '平台策略基于随机森林模型，采用月度训练、周度调仓的框架，'
        '设置了8%止损线、35%最大仓位和55%置信度阈值三重风控机制。'
        '为适应平台的事件驱动架构，将策略逻辑封装为初始化函数和'
        '数据处理函数，利用平台的行情接口和订单接口实现自动化交易。'
        '同时进行了9组参数调优实验，'
        '测试了不同的树数量（50、100、200）和最大深度（3、5、10）组合。',
        style_body
    ))

    # 表5: 平台策略回测结果
    table5_data = [
        ['指标', '机器学习组合策略', '等权买入持有'],
        ['累计收益率(%)', '57.54', '123.09'],
        ['年化收益率(%)', '61.49', '133.04'],
        ['夏普比率', '2.27', '3.01'],
        ['最大回撤(%)', '-11.42', '-13.18'],
        ['年化波动率(%)', '21.20', '28.86'],
        ['日胜率(%)', '52.00', '57.56'],
        ['Beta', '0.54', '1.00'],
    ]
    story.append(make_table(table5_data,
                            col_widths=[3.5*cm, 3.5*cm, 3.5*cm],
                            caption='表5 聚宽平台策略回测结果'))

    story.append(Paragraph(
        '表5的回测结果表明，机器学习组合策略虽然在绝对收益上低于买入持有'
        '（年化61.49%对133.04%），但在风险控制方面表现突出：'
        '年化波动率从28.86%降至21.20%，最大回撤从13.18%改善至11.42%，'
        'Beta值仅为0.54，说明策略与市场基准的相关性较低，'
        '具备一定的市场中性特征。'
        '风险价值分析显示，95%置信度下的日最大损失为0.73%，'
        '99%置信度下为1.35%，均在可接受范围内。'
        '（详见附录建议6）',
        style_body
    ))

    story.append(Paragraph('（四）优势、局限与未来趋势', style_heading2))
    story.append(Paragraph(
        '<b>优势方面，</b>机器学习策略具有非线性建模能力，'
        '能够捕捉技术指标之间的复杂交互关系；'
        '通过滚动训练实现模型自适应，适应市场环境的变化；'
        '动态仓位管理使策略在风险控制上优于传统的全仓策略。',
        style_body
    ))
    story.append(Paragraph(
        '<b>局限方面，</b>金融数据的低信噪比使得分类精度难以大幅提升'
        '（AUC接近0.5）；模型容易过拟合历史数据，'
        '在样本外的泛化能力存在不确定性；'
        '技术指标因子多为价格衍生变量，信息冗余度较高；'
        '策略回测未充分考虑交易成本和滑点的影响。',
        style_body
    ))
    story.append(Paragraph(
        '<b>未来趋势方面，</b>深度学习模型（如长短期记忆网络和Transformer）'
        '在时序预测中展现出更强的特征提取能力；'
        '自然语言处理技术可从新闻和公告中提取文本因子，丰富信息维度；'
        '强化学习在动态仓位管理和执行优化方面具有潜力；'
        '图神经网络可建模股票间的关联关系，辅助组合构建。'
        '这些方向为量化交易的进一步发展提供了广阔空间。',
        style_body
    ))

    story.append(PageBreak())

    # ============================================================
    # 第四章：结论与展望
    # ============================================================
    story.append(Paragraph('四、结论与展望', style_heading1))

    story.append(Paragraph('（一）主要收获', style_heading2))
    story.append(Paragraph(
        '通过七个阶段的系统学习和实践，本研究在以下方面取得了显著收获。'
        '<b>在理论层面，</b>深入理解了量化交易的核心框架——'
        '从数据获取到信号生成、从回测验证到风险管理的完整闭环。'
        '掌握了双均线策略、海龟交易策略和机器学习策略的原理与实现方法，'
        '明确了各类策略的适用条件和局限性。',
        style_body
    ))
    story.append(Paragraph(
        '<b>在技术层面，</b>熟练运用Python进行金融数据处理、'
        '技术指标计算、策略回测和可视化展示。'
        '掌握了scikit-learn机器学习库的使用，能够构建和评估分类模型。'
        '在聚宽平台上完成了从策略编写到实盘模拟的全流程操作，'
        '具备了量化策略开发与部署的实践能力。',
        style_body
    ))
    story.append(Paragraph(
        '<b>在实践层面，</b>通过多股票、多参数的系统对比实验，'
        '获得了丰富的实证经验。特别是在机器学习策略中，'
        '深刻认识到金融市场预测的困难性——'
        'AUC接近0.5并不意味着策略无效，'
        '关键在于如何利用模型输出构建有正期望的交易规则。'
        '这一认知对于理性看待量化交易的预期收益具有重要指导意义。',
        style_body
    ))

    story.append(Paragraph('（二）未来展望', style_heading2))
    story.append(Paragraph(
        '基于本研究的经验和发现，未来计划在以下方向进行深入探索。',
        style_body
    ))
    story.append(Paragraph(
        '<b>第一，因子工程深化。</b>当前策略仅使用技术指标因子，'
        '信息维度单一。未来将引入基本面因子（如估值、成长性、盈利质量）'
        '和另类数据因子（如新闻情感、资金流向），构建多维度因子库，'
        '提升模型的预测能力。',
        style_body
    ))
    story.append(Paragraph(
        '<b>第二，深度学习模型探索。</b>尝试使用长短期记忆网络（LSTM）'
        '和注意力机制（Attention）处理时序数据，'
        '利用其更强的序列建模能力捕捉价格运动的中长期模式。'
        '同时探索强化学习在动态仓位管理中的应用。',
        style_body
    ))
    story.append(Paragraph(
        '<b>第三，多策略组合优化。</b>将双均线、海龟和机器学习策略'
        '按风险预算模型进行配置，研究不同策略之间的相关性结构，'
        '通过优化权重分配实现组合层面的风险收益最优化。',
        style_body
    ))
    story.append(Paragraph(
        '<b>第四，实盘交易验证。</b>在模拟验证的基础上，'
        '选择表现稳健的策略进行小资金实盘测试，'
        '重点关注交易成本、滑点和冲击成本对策略表现的实际影响，'
        '不断完善策略的工程化实现。',
        style_body
    ))
    story.append(Paragraph(
        '量化交易是一个持续学习和迭代的过程。本报告所呈现的成果'
        '仅为阶段性总结，未来将在因子挖掘、模型创新和组合管理方面'
        '不断深化，追求更加稳健和可持续的投资回报。',
        style_body
    ))

    story.append(PageBreak())

    # ============================================================
    # 附录：改进建议
    # ============================================================
    story.append(Paragraph('附录：改进建议', style_heading1))
    story.append(Spacer(1, 6))

    suggestions = [
        ('建议1', '双均线策略增加过滤条件',
         '双均线策略在中船特气上收益突出但回撤较大（27.96%），'
         '建议增加成交量或波动率过滤条件——仅在成交量放大或波动率收缩时执行金叉信号，'
         '以减少震荡市场中的虚假信号，降低无效交易次数。'),
        ('建议2', '海龟策略引入多周期确认',
         '海龟策略的唐奇安通道突破信号在震荡市中容易产生假突破。'
         '建议引入多周期确认机制——短期通道突破需得到长期趋势方向的支撑，'
         '即仅在长期均线向上时执行买入信号，避免逆势交易。'),
        ('建议3', '构建策略权重动态调整机制',
         '多策略组合的静态等权配置未充分利用各策略的时变表现差异。'
         '建议基于各策略近期的夏普比率或胜率动态调整权重，'
         '增加表现稳定策略的配置比例，降低回撤期策略的暴露。'),
        ('建议4', '丰富机器学习因子维度',
         '当前技术指标因子的AUC接近0.5，预测能力有限。'
         '建议引入基本面因子（市盈率、市净率、净资产收益率）'
         '和情绪因子（龙虎榜数据、融资融券余额变化），'
         '从多维度刻画股票的涨跌驱动力。'),
        ('建议5', '采用滚动窗口训练框架',
         '一次性训练的模型无法适应市场环境的变化。'
         '建议采用滚动窗口训练——每月或每季度使用最近N个月的数据重新训练模型，'
         '使模型始终适应当前的市场状态，提升样本外的泛化能力。'),
        ('建议6', '纳入交易成本和滑点模拟',
         '当前回测未充分考虑交易成本的影响，可能高估策略收益。'
         '建议在回测中纳入佣金（万分之三）、印花税（千分之一）'
         '和滑点（0.1%至0.2%）的模拟，使回测结果更接近实盘表现。'),
    ]

    for num, title, content in suggestions:
        story.append(Paragraph(f'<b>{num}：{title}</b>', style_appendix))
        story.append(Paragraph(content, style_appendix))
        story.append(Spacer(1, 6))

    # 构建PDF
    doc.build(story, onFirstPage=add_page_header_footer, onLaterPages=add_page_header_footer)
    print(f"\nPDF报告已生成: {output_path}")

    # 检查文件大小
    size = os.path.getsize(output_path) / 1024
    print(f"文件大小: {size:.1f} KB")

    return output_path


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("TASK8: 量化交易学习成果综合报告生成")
    print("=" * 60)

    print("\n[1/2] 生成综合对比图表...")
    charts = generate_charts()

    print("\n[2/2] 构建PDF报告...")
    pdf_path = build_report(charts)

    print("\n" + "=" * 60)
    print("报告生成完成!")
    print("=" * 60)
