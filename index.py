from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안함
import matplotlib.pyplot as plt
import io
import base64
import math

app = Flask(__name__)

def format_number(num):
    """숫자를 보기 좋게 포맷팅"""
    if abs(num) >= 100000000:  # 1억 이상
        return f'{num/100000000:.1f}억원'
    elif abs(num) >= 10000:    # 1만 이상
        return f'{num/10000:.1f}만원'
    else:
        return f'{num:,.0f}원'

def format_volume(num):
    """발송량 포맷팅 (소수점 내림 처리)"""
    num = math.floor(num)  # 소수점 내림
    
    if num >= 1000000:
        return f'{math.floor(num/1000000)}M'
    elif num >= 1000:
        return f'{math.floor(num/1000)}K'
    return str(num)

def create_graph(params):
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 파라미터 추출
    base_cost = float(params['base_cost'])
    revenue_rate = float(params['revenue_rate'])
    volumes = [
        int(params['volume_1']),
        int(params['volume_2']),
        int(params['volume_3'])
    ]
    
    # 비용 파라미터
    server_cost = float(params['server_cost'])
    additional_costs = float(params['additional_costs'])
    
    # 요금제 파라미터
    basic_fee = float(params['basic_fee'])
    standard_fee = float(params['standard_fee'])
    premium_fee = float(params['premium_fee'])
    
    # 사용자 수
    basic_users = int(params['basic_users'])
    standard_users = int(params['standard_users'])
    premium_users = int(params['premium_users'])
    
    # 월 구독 수익
    monthly_subscription = (
        basic_fee * basic_users +
        standard_fee * standard_users +
        premium_fee * premium_users
    )
    
    # 결과 저장
    results = []
    for volume in volumes:
        # 매출 계산
        message_revenue = volume * revenue_rate
        total_revenue = message_revenue + monthly_subscription
        
        # 비용 계산
        message_cost = volume * base_cost
        total_cost = message_cost + server_cost + additional_costs
        
        # GP 계산
        gp = total_revenue - total_cost
        gp_margin = (gp / total_revenue) * 100 if total_revenue > 0 else 0
        
        results.append([
            volume, message_revenue, monthly_subscription,
            total_revenue, message_cost, server_cost + additional_costs,
            total_cost, gp, gp_margin
        ])

    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=[
        "발송량(건)", "메시지매출(원)", "구독매출(원)",
        "총매출(원)", "메시지비용(원)", "고정비용(원)",
        "총비용(원)", "GP(원)", "GP마진(%)"
    ])

    # 그래프 생성 (2x3 레이아웃)
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. 매출 구조 분석 (파이 차트)
    ax1 = fig.add_subplot(gs[0, 0])
    total_message_revenue = df["메시지매출(원)"].iloc[-1]
    subscription_revenue = monthly_subscription
    revenue_data = [total_message_revenue, subscription_revenue]
    labels = ['메시지 매출', '구독 매출']
    ax1.pie(revenue_data, labels=labels, autopct='%1.1f%%', 
            colors=['lightblue', 'lightgreen'])
    ax1.set_title('매출 구조 비율', pad=20, fontsize=12, fontweight='bold')

    # 2. GP 추이 분석
    ax2 = fig.add_subplot(gs[0, 1])
    color1, color2 = '#3498db', '#e74c3c'
    ln1 = ax2.plot(volumes, df["GP(원)"] / 10000, 
                   color=color1, marker='o', linewidth=2, label='GP')
    ax2.set_ylabel('GP (만원)', color=color1, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color1)
    
    ax2_2 = ax2.twinx()
    ln2 = ax2_2.plot(volumes, df["GP마진(%)"], 
                     color=color2, marker='s', linestyle='--', label='GP 마진율')
    ax2_2.set_ylabel('GP 마진율 (%)', color=color2, fontsize=10)
    ax2_2.tick_params(axis='y', labelcolor=color2)
    
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='upper left')
    ax2.set_title('발송량별 GP 추이', pad=20, fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)

    # 3. 비용 구조 분석
    ax3 = fig.add_subplot(gs[0, 2])
    width = 0.35
    x = np.arange(len(volumes))
    ax3.bar(x, df["메시지비용(원)"] / 10000, width, label='메시지 비용')
    ax3.bar(x, df["고정비용(원)"] / 10000, width, 
            bottom=df["메시지비용(원)"] / 10000, label='고정 비용')
    ax3.set_xticks(x)
    ax3.set_xticklabels([format_volume(v) for v in volumes])
    ax3.set_ylabel('비용 (만원)')
    ax3.set_title('비용 구조 분석', pad=20, fontsize=12, fontweight='bold')
    ax3.legend()

    # 4. 수익성 지표 대시보드
    ax4 = fig.add_subplot(gs[1, 0])
    metrics = {
        '평균 GP 마진율': f"{df['GP마진(%)'].mean():.1f}%",
        '최대 GP': format_number(df['GP(원)'].max()),
        '최소 GP': format_number(df['GP(원)'].min()),
        '손익분기점': format_volume(-(server_cost + additional_costs)/(revenue_rate-base_cost)) + '건'
    }
    
    # 수익성 지표 시각화 개선
    y_pos = np.arange(len(metrics))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (metric, value) in enumerate(metrics.items()):
        # 배경 박스 생성
        ax4.add_patch(plt.Rectangle(
            (0, i-0.3), 1, 0.6,
            color=colors[i],
            alpha=0.2
        ))
        # 지표 이름
        ax4.text(0.05, i, metric,
                ha='left', va='center',
                fontsize=11, fontweight='bold',
                color='#2c3e50')
        # 지표 값
        ax4.text(0.95, i, value,
                ha='right', va='center',
                fontsize=12, fontweight='bold',
                color=colors[i])
    
    ax4.set_ylim(-0.5, len(metrics)-0.5)
    ax4.set_xlim(0, 1)
    ax4.axis('off')
    ax4.set_title('주요 수익성 지표', pad=20, fontsize=12, fontweight='bold')

    # x축 레이블 포맷팅 수정
    ax2.set_xticklabels([format_volume(v) for v in volumes])
    ax3.set_xticklabels([format_volume(v) for v in volumes])
    
    # y축 레이블 포맷팅 수정
    def format_yaxis(x, p):
        return format_number(x * 10000).replace('원', '')
    
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_yaxis))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_yaxis))
    
    # 파이 차트 값 포맷팅
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f'{pct:.1f}%\n({format_number(val)})'
        return my_autopct

    # 매출 구조 파이 차트 업데이트
    ax1.clear()
    revenue_data = [total_message_revenue, subscription_revenue]
    ax1.pie(revenue_data, labels=labels, 
            autopct=make_autopct(revenue_data),
            colors=['lightblue', 'lightgreen'])
    ax1.set_title('매출 구조 비율', pad=20, fontsize=12, fontweight='bold')

    # 요금제별 수익 파이 차트 업데이트
    ax5 = fig.add_subplot(gs[1, 1])
    fees = [basic_fee * basic_users, 
            standard_fee * standard_users, 
            premium_fee * premium_users]
    ax5.pie(fees, labels=['Basic', 'Standard', 'Premium'], 
            autopct=make_autopct(fees),
            colors=['#3498db', '#2ecc71', '#e74c3c'])
    ax5.set_title('요금제별 수익 기여도', pad=20, fontsize=12, fontweight='bold')

    # 손익분기점 그래프 업데이트
    ax6 = fig.add_subplot(gs[1, 2])
    x_range = np.linspace(0, max(volumes), 100)
    revenue = x_range * revenue_rate + monthly_subscription
    cost = x_range * base_cost + server_cost + additional_costs
    ax6.plot(x_range, revenue/10000, label='총 매출', color='#2ecc71')
    ax6.plot(x_range, cost/10000, label='총 비용', color='#e74c3c')
    ax6.axvline(x=-(server_cost + additional_costs)/(revenue_rate-base_cost), 
                color='gray', linestyle='--', alpha=0.5)
    ax6.set_title('손익분기점 분석', pad=20, fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.3)

    # 전체 타이틀 추가
    fig.suptitle(f'알마니 수익성 종합 분석 리포트\n(원가: {base_cost}원, 매출단가: {revenue_rate}원)', 
                 fontsize=16, fontweight='bold', y=0.95)

    # 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        params = {
            'base_cost': request.form.get('base_cost', '8.5'),
            'revenue_rate': request.form.get('revenue_rate', '12.0'),
            'volume_1': request.form.get('volume_1', '100000'),
            'volume_2': request.form.get('volume_2', '500000'),
            'volume_3': request.form.get('volume_3', '1000000'),
            'server_cost': request.form.get('server_cost', '1000000'),
            'additional_costs': request.form.get('additional_costs', '500000'),
            'basic_fee': request.form.get('basic_fee', '4900'),
            'standard_fee': request.form.get('standard_fee', '19000'),
            'premium_fee': request.form.get('premium_fee', '49000'),
            'basic_users': request.form.get('basic_users', '10'),
            'standard_users': request.form.get('standard_users', '5'),
            'premium_users': request.form.get('premium_users', '2')
        }
        graph_image = create_graph(params)
        return render_template('index.html', graph_image=graph_image, params=params)
    return render_template('index.html')

@app.route('/breakeven', methods=['GET', 'POST'])
def breakeven_analysis():
    if request.method == 'POST':
        params = {
            'base_cost': float(request.form.get('base_cost', '8.5')),
            'revenue_rate': float(request.form.get('revenue_rate', '12.0')),
            'server_cost': float(request.form.get('server_cost', '1000000')),
            'additional_costs': float(request.form.get('additional_costs', '500000')),
            'basic_fee': float(request.form.get('basic_fee', '4900')),
            'standard_fee': float(request.form.get('standard_fee', '19000')),
            'premium_fee': float(request.form.get('premium_fee', '49000')),
            'basic_users': int(request.form.get('basic_users', '10')),
            'standard_users': int(request.form.get('standard_users', '5')),
            'premium_users': int(request.form.get('premium_users', '2'))
        }
        
        graph_image = create_breakeven_analysis(params)
        return render_template('breakeven.html', graph_image=graph_image, params=params)
    return render_template('breakeven.html')

def create_breakeven_analysis(params):
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 기본 파라미터 계산
    base_cost = params['base_cost']  # 발송 원가
    revenue_rate = params['revenue_rate']  # 발송 매출단가
    server_cost = params['server_cost']  # 서버 비용
    additional_costs = params['additional_costs']  # 부대비용
    fixed_costs = server_cost + additional_costs  # 총 고정비용
    
    # 구독 수익 계산
    monthly_subscription = (
        params['basic_fee'] * params['basic_users'] +
        params['standard_fee'] * params['standard_users'] +
        params['premium_fee'] * params['premium_users']
    )
    
    # 2. 손익분기점 계산 수정
    # 총매출 = 발송매출 + 구독수익
    # 총비용 = 발송원가 + 서버비용 + 부대비용
    # 손익분기: 발송매출 + 구독수익 = 발송원가 + 서버비용 + 부대비용
    # (발송량 × 매출단가) + 구독수익 = (발송량 × 원가) + 서버비용 + 부대비용
    # 발송량 × (매출단가 - 원가) = (서버비용 + 부대비용) - 구독수익
    
    # 발송량에 따른 수익과 비용의 차이 계산
    margin_per_volume = revenue_rate - base_cost  # 건당 마진
    required_margin = fixed_costs - monthly_subscription  # 필요 마진
    
    if margin_per_volume <= 0:
        breakeven_volume = float('inf')  # 손익분기점 도달 불가
    else:
        breakeven_volume = max(0, required_margin / margin_per_volume)
    
    # 3. 그래프 생성
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 4. 손익분기점 그래프 (상단)
    ax1 = fig.add_subplot(gs[0, :])
    
    # 발송량 범위 설정
    if breakeven_volume == float('inf'):
        max_volume = 1000000  # 기본 최대 발송량 설정
    else:
        max_volume = max(breakeven_volume * 2, 100000)  # 최소 10만건 이상 표시
    
    volume_range = np.linspace(0, max_volume, 100)
    
    # 매출, 비용, 이익 계산
    total_revenue = volume_range * revenue_rate + monthly_subscription  # 총 매출
    total_cost = volume_range * base_cost + fixed_costs  # 총 비용
    profit = total_revenue - total_cost  # 순이익
    
    # 그래프 그리기
    ax1.plot(volume_range, total_revenue/10000, label='총 매출', color='#2ecc71', linewidth=2)
    ax1.plot(volume_range, total_cost/10000, label='총 비용', color='#e74c3c', linewidth=2)
    ax1.plot(volume_range, profit/10000, label='순이익', color='#3498db', linewidth=2)
    
    # 기준선
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 손익분기점 표시
    if breakeven_volume != float('inf'):
        ax1.axvline(x=breakeven_volume, color='gray', linestyle='--', alpha=0.5)
        ax1.plot([breakeven_volume], [0], 'ko', markersize=10)
        
        # 손익분기점에서의 매출 계산
        breakeven_revenue = breakeven_volume * revenue_rate + monthly_subscription
        
        # 손익분기점 주석
        bbox_props = dict(
            boxstyle='round,pad=0.5',
            fc='yellow',
            alpha=0.5,
            edgecolor='gray'
        )
        ax1.annotate(
            f'손익분기점\n{format_volume(breakeven_volume)}건\n' +
            f'(월 매출: {format_number(breakeven_revenue)})',
            xy=(breakeven_volume, 0),
            xytext=(30, 30),
            textcoords='offset points',
            ha='left',
            va='bottom',
            bbox=bbox_props,
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=0.2',
                color='gray'
            ),
            fontsize=10,
            fontweight='bold'
        )
    else:
        ax1.text(
            0.05, 0.95,
            '현재 설정으로는 손익분기점에 도달할 수 없습니다.\n' +
            f'발송 마진: {revenue_rate - base_cost}원/건',
            transform=ax1.transAxes,
            bbox=dict(facecolor='red', alpha=0.2),
            fontsize=12,
            verticalalignment='top'
        )
    
    # 그래프 설정
    ax1.set_title('손익분기점 분석', pad=20, fontsize=14, fontweight='bold')
    ax1.set_xlabel('월간 발송량 (건)', fontsize=12)
    ax1.set_ylabel('금액 (만원)', fontsize=12)
    
    # 범례 위치 조정
    ax1.legend(
        fontsize=12,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 2. 수익성 지표
    ax2 = fig.add_subplot(gs[1, 0])
    metrics = {
        '손익분기 발송량': format_volume(breakeven_volume) + '건',
        '손익분기 매출액': format_number(revenue_rate * breakeven_volume + monthly_subscription),
        '고정비용': format_number(fixed_costs),
        '월 구독수익': format_number(monthly_subscription)
    }
    
    y_pos = np.arange(len(metrics))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (metric, value) in enumerate(metrics.items()):
        ax2.add_patch(plt.Rectangle(
            (0, i-0.3), 1, 0.6,
            color=colors[i],
            alpha=0.2
        ))
        ax2.text(0.05, i, metric,
                ha='left', va='center',
                fontsize=11, fontweight='bold',
                color='#2c3e50')
        ax2.text(0.95, i, value,
                ha='right', va='center',
                fontsize=12, fontweight='bold',
                color=colors[i])
    
    ax2.set_ylim(-0.5, len(metrics)-0.5)
    ax2.set_xlim(0, 1)
    ax2.axis('off')
    ax2.set_title('주요 지표', pad=20, fontsize=14, fontweight='bold')
    
    # 3. 민감도 분석
    ax3 = fig.add_subplot(gs[1, 1])
    
    # 매출단가 범위 설정 (현재 매출단가 기준으로 앞뒤 2원씩)
    base_rate = round(revenue_rate)  # 현재 매출단가를 반올림
    rate_changes = np.array([-2, -1, 0, 1, 2])
    actual_rates = np.array([base_rate + change for change in rate_changes])
    breakeven_changes = []
    
    for rate in actual_rates:
        if rate <= base_cost:  # 매출단가가 원가 이하면 손익분기점 없음
            new_breakeven = float('inf')
        else:
            new_breakeven = -fixed_costs / (rate - base_cost)
        breakeven_changes.append(new_breakeven)
    
    # 무한대 값 처리
    breakeven_changes = np.array(breakeven_changes)
    mask = np.isfinite(breakeven_changes)
    
    if np.any(mask):  # 유한한 값이 있는 경우만 그래프 그리기
        ax3.plot(actual_rates[mask], breakeven_changes[mask], 
                marker='o', linewidth=2, color='#3498db')
        
        # y축 방향 뒤집기
        ax3.invert_yaxis()
        
        # x축 레이블 수정
        ax3.set_xticks(actual_rates)
        ax3.set_xticklabels([f'{int(rate)}원' for rate in actual_rates], rotation=0)
        
        ax3.set_title('매출단가별 손익분기점', pad=20, fontsize=14, fontweight='bold')
        ax3.set_xlabel('매출단가', fontsize=12)
        ax3.set_ylabel('손익분기 발송량 (건)', fontsize=12)
        
        # 격자 추가
        ax3.grid(True, linestyle='--', alpha=0.3)
        
        # y축 레이블 포맷팅
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_volume(x)))
        
        # 각 포인트에 값 표시
        for x, y in zip(actual_rates[mask], breakeven_changes[mask]):
            ax3.annotate(
                format_volume(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, -15),
                ha='center',
                fontsize=9
            )
    
    # 전체 타이틀
    fig.suptitle(
        f'알마니 손익분기점 상세 분석\n' + 
        f'(원가: {base_cost}원, 매출단가: {int(revenue_rate)}원)', 
        fontsize=16, fontweight='bold', y=0.95
    )
    
    # 그래프 여백 조정
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True, port=5001)
