from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안함
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

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
            total_revenue, total_cost, gp, gp_margin
        ])

    # 데이터프레임 생성
    df = pd.DataFrame(results, columns=[
        "발송량(건)", "메시지매출(원)", "구독매출(원)",
        "총매출(원)", "총비용(원)", "GP(원)", "GP마진(%)"
    ])

    # 그래프 생성
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. 매출 구조 분석
    ax1.bar(range(len(volumes)), df["메시지매출(원)"] / 10000, label='메시지 매출')
    ax1.bar(range(len(volumes)), df["구독매출(원)"] / 10000, bottom=df["메시지매출(원)"] / 10000, label='구독 매출')
    ax1.set_title('매출 구조 분석')
    ax1.set_ylabel('매출 (만원)')
    ax1.set_xticks(range(len(volumes)))
    ax1.set_xticklabels([f'{v:,}건' for v in volumes])
    ax1.legend()
    
    # 2. GP 분석
    color = 'tab:blue'
    ax2.set_xlabel('발송량 (건)')
    ax2.set_ylabel('GP (만원)', color=color)
    line1 = ax2.plot(volumes, df["GP(원)"] / 10000, color=color, marker='o', label='GP')
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax2_2 = ax2.twinx()
    color = 'tab:red'
    ax2_2.set_ylabel('GP 마진율 (%)', color=color)
    line2 = ax2_2.plot(volumes, df["GP마진(%)"], color=color, marker='s', linestyle='--', label='GP 마진율')
    ax2_2.tick_params(axis='y', labelcolor=color)
    ax2.set_title('GP 분석')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    # 3. 비용 구조 분석
    fixed_costs = server_cost + additional_costs
    ax3.bar(range(len(volumes)), df["총비용(원)"] / 10000, label='메시지 비용')
    ax3.bar(range(len(volumes)), [fixed_costs / 10000] * len(volumes), label='고정 비용')
    ax3.set_title('비용 구조 분석')
    ax3.set_ylabel('비용 (만원)')
    ax3.set_xticks(range(len(volumes)))
    ax3.set_xticklabels([f'{v:,}건' for v in volumes])
    ax3.legend()

    plt.tight_layout()
    
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

if __name__ == '__main__':
    app.run(debug=True, port=5001)