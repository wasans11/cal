import streamlit as st
import joblib
import pandas as pd
import catboost

@st.cache_resource
def load_model():
    return joblib.load('m0627.pkl')

def calculate_cumulative_rainfall_factor(recent_rain_level, current_rainfall):
    """
    최근 강수 상태와 현재 강수량을 종합하여 토양 습도 추정
    recent_rain_level: 0-5 단계 (건조 → 습윤)
    """
    # 최근 강수 상태별 기본 습윤도 (토양 수분 잔존 효과)
    base_moisture = {
        0: 0.0,    # 매우 건조 (3일+ 무강수)
        1: 0.1,    # 건조 (2-3일 무강수)  
        2: 0.3,    # 보통 (1-2일 전 소량 강수)
        3: 0.5,    # 습윤 (최근 24시간 내 강수)
        4: 0.7,    # 매우 습윤 (지속적 강수)
        5: 0.9     # 포화 (연속 강수/홍수 주의보급)
    }
    
    # 현재 강수량에 따른 추가 습윤 효과
    current_effect = 0
    if current_rainfall >= 10:
        current_effect = 0.9
    elif current_rainfall >= 5:
        current_effect = 0.7
    elif current_rainfall >= 1:
        current_effect = 0.4
    elif current_rainfall > 0:
        current_effect = 0.2
    
    # 종합 습윤도 (최대값 사용 - 둘 중 하나라도 높으면 습함)
    total_moisture = max(base_moisture[recent_rain_level], current_effect)
    
    return total_moisture

def apply_smart_rainfall_adjustment(base_risk, rainfall_mm, recent_rain_level):
    """스마트 강수 조정 - 누적 효과 고려"""
    moisture_factor = calculate_cumulative_rainfall_factor(recent_rain_level, rainfall_mm)
    
    # 습윤도에 따른 위험도 감소 (지수적 감소)
    risk_multiplier = 1 - (moisture_factor * 0.85)  # 최대 85% 감소
    
    return base_risk * risk_multiplier, moisture_factor

def get_risk_level(risk):
    """위험도 레벨 및 색상 반환"""
    if risk >= 80:
        return "🚨 극도로 높음", "darkred"
    elif risk >= 65:
        return "🔥 매우 높음", "red"
    elif risk >= 45:
        return "⚠️ 높음", "orange"
    elif risk >= 25:
        return "🔶 보통", "gold"
    elif risk >= 10:
        return "💚 낮음", "green"
    else:
        return "✅ 매우 낮음", "blue"

# 모델 로드
model = load_model()

st.title("🔥 산불 위험도 예측 시스템 v2.0")
st.caption("스마트폰 날씨앱에 있는 데이터를 넣어주세요")

# 기상 정보 입력
st.subheader("🌤️ 현재 기상 정보")
col1, col2 = st.columns(2)

with col1:
    기온 = st.number_input("기온 (°C)", value=25.0, step=0.1)
    풍속 = st.number_input("풍속 (m/s)", value=2.0, step=0.1)
    이슬점온도 = st.number_input("이슬점온도 (°C)", value=15.0, step=0.1)
    월 = st.selectbox("월", list(range(1,13)), index=4)

with col2:
    강수량 = st.number_input("현재 강수량 (mm)", value=0.0, step=0.1, min_value=0.0)
    습도 = st.number_input("습도 (%)", value=50.0, step=1.0, min_value=0.0, max_value=100.0)
    기압 = st.number_input("기압 (hPa)", value=1013.25, step=0.1)
    시간 = st.selectbox("시간", list(range(24)), index=12)

# 핵심 개선: 최근 강수 상태 입력
st.subheader("💧 최근 강수 상태 (핵심 보정 요소)")
recent_rain_level = st.radio(
    "지난 3일간 날씨를 생각해보세요:",
    options=[0, 1, 2, 3, 4, 5],
    format_func=lambda x: {
        0: "☀️ 매우 건조 - 3일 이상 비 없음, 흙이 바싹 마름",
        1: "🌤️ 건조 - 2-3일 전 약간의 비, 표면만 약간 습함",
        2: "⛅ 보통 - 1-2일 전 소량 강수, 지면이 촉촉함",
        3: "🌧️ 습윤 - 24시간 내 비, 땅이 충분히 젖음",
        4: "🌧️ 매우 습윤 - 이틀간 지속적 강수, 땅이 질척함",
        5: "⛈️ 포화 - 연속 강수로 물웅덩이/배수 문제 발생"
    }[x],
    index=1,
    help="최근 강수 이력이 토양 습도에 미치는 누적 효과를 반영합니다."
)

# 풍향 입력
풍향_선택 = st.selectbox("풍향", ['북','북동','동','남동','남','남서','서','북서'])
풍향_맵 = {'북':0,'북동':1,'동':2,'남동':3,'남':4,'남서':5,'서':6,'북서':7}
풍향 = 풍향_맵[풍향_선택]

# 예측 실행
if st.button("🔥 화재 위험도 예측", type="primary"):
    # 데이터 준비
    X = pd.DataFrame([[
        기온, 강수량, 풍속, 습도, 이슬점온도, 기압, 월, 시간, 풍향
    ]], columns=['기온','강수량','풍속','습도','이슬점온도','기압','월','시간','풍향'])
    
    for c in ['월','시간','풍향']:
        X[c] = X[c].astype(str)
    
    try:
        pool = catboost.Pool(X, cat_features=[6,7,8])
        proba = model.predict_proba(pool)[0][1]
        
        base_risk = proba * 100
        adjusted_risk, moisture_factor = apply_smart_rainfall_adjustment(
            base_risk, 강수량, recent_rain_level
        )
        
        # 결과 표시
        st.subheader("📊 예측 결과")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("기본 모델 예측", f"{base_risk:.1f}%")
        with col2:
            st.metric("토양 습윤도", f"{moisture_factor:.1%}")
        with col3:
            st.metric("최종 위험도", f"{adjusted_risk:.1f}%")
        
        # 조정 효과 시각화
        reduction = base_risk - adjusted_risk
        if reduction > 0:
            st.success(f"💧 토양 습윤도로 인해 위험도 {reduction:.1f}%p 감소")
        
        # 위험도 레벨
        level, color = get_risk_level(adjusted_risk)
        st.markdown(f"### 🎯 종합 위험도: <span style='color:{color}; font-weight:bold'>{level}</span>", 
                   unsafe_allow_html=True)
        
        # 프로그레스 바
        progress_val = min(adjusted_risk / 100, 1.0)
        st.progress(progress_val)
        
        # 상세 분석
        with st.expander("📈 상세 분석"):
            st.markdown(f"""
            **기상 조건 분석:**
            - 기온 {기온}°C, 습도 {습도}%, 풍속 {풍속}m/s
            - 현재 강수량: {강수량}mm
            
            **토양 상태 평가:**
            - 최근 강수 상태: {recent_rain_level}/5 단계
            - 추정 토양 습윤도: {moisture_factor:.1%}
            - 화재 억제 효과: {reduction:.1f}%p
            
            **위험 요인:**
            {'- 건조한 토양으로 인한 높은 착화 위험' if moisture_factor < 0.3 else '- 습한 토양으로 화재 위험 크게 감소'}
            """)
            
    except Exception as e:
        st.error(f"예측 중 오류: {str(e)}")

# 사이드바 가이드
with st.sidebar:
    st.header("💡 사용 가이드")
    
    st.subheader("최근 강수 상태 판단법")
    st.markdown("""
    **3일간 날씨 회상:**
    - 0-1: 계속 맑았음
    - 2: 하루 정도 비 옴  
    - 3: 어제/오늘 비 옴
    - 4-5: 계속 비가 옴
    """)
    
    st.subheader("개선된 특징")
    st.markdown("""
    ✅ 토양 습도 누적 효과 반영  
    ✅ 직관적 최근 날씨 입력  
    ✅ 오탐 위험 대폭 감소  
    ✅ 실용성 중심 설계  
    """)
    
    st.subheader("한계점")
    st.markdown("""
    ⚠️ 사용자 기억에 의존  
    ⚠️ 지역별 차이 미반영  
    ⚠️ 정밀 기상 데이터 부재  
    """)

st.markdown("---")
st.markdown("""
## 📊 모델 성능 정보

**🔹 훈련 데이터 성능:**  
화재 탐지율 98% | 정밀도 98%

**🔹 실제 테스트 성능:**  
• Test1 (화재 2%): 탐지율 100%, 정밀도 5%  
• Test2 (화재 5%): 탐지율 84%, 정밀도 12%

**사용 모델:** CatBoost | **최적화:** Optuna | **해석성:** SHAP | **특징:** 고탐지율

**하이퍼파라미터:** depth=8, learning_rate=0.1, l2_leaf_reg=0.8, iterations=358  
**강수량 조정:** 임계값 기반 위험도 감소 적용
""")
st.caption("💬 **개선 사항**: 최근 강수 누적 효과를 반영하여 오탐을 줄이고 실용성을 높였습니다.")
