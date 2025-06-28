import streamlit as st
import joblib
import pandas as pd
import catboost

@st.cache_resource
def load_model():
    return joblib.load('m0627.pkl')

def calculate_cumulative_rainfall_factor(recent_rain_level, current_rainfall, humidity):
    """
    최근 강수 상태와 현재 강수량으로 토양 습도 추정 (습도는 대기 상태만 반영)
    """
    # 최근 강수 상태별 기본 토양 습윤도
    base_moisture = {
        0: 0.0,    # 매우 건조
        1: 0.1,    # 건조
        2: 0.3,    # 보통
        3: 0.5,    # 습윤
        4: 0.7,    # 매우 습윤
        5: 0.9     # 포화
    }
    
    # 현재 강수량에 따른 즉시 효과
    if current_rainfall >= 10:
        rain_effect = 0.9
    elif current_rainfall >= 5:
        rain_effect = 0.7
    elif current_rainfall >= 1:
        rain_effect = 0.4
    elif current_rainfall > 0:
        rain_effect = 0.2
    else:
        rain_effect = 0.0
    
    # 토양 습윤도는 강수량과 최근 강수 상태 중 높은 값 사용
    soil_moisture = max(base_moisture[recent_rain_level], rain_effect)
    
    return soil_moisture

def apply_smart_rainfall_adjustment(base_risk, rainfall_mm, recent_rain_level, humidity):
    """토양 습윤도 기반 위험도 조정 + 대기 습도 별도 적용"""
    # 토양 습윤도 계산 (강수만 반영)
    soil_moisture = calculate_cumulative_rainfall_factor(recent_rain_level, rainfall_mm, humidity)
    
    # 대기 습도 효과 (별도 적용, 약한 효과)
    humidity_factor = max(0, (humidity - 60) / 40) * 0.2  # 최대 20% 감소
    
    # 토양 습윤도 효과 (강한 효과)
    soil_factor = soil_moisture * 0.7  # 최대 70% 감소
    
    # 총 위험도 감소 (둘 중 큰 값 적용)
    total_reduction = max(soil_factor, humidity_factor)
    risk_multiplier = 1 - total_reduction
    
    return base_risk * risk_multiplier, soil_moisture

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

st.title("🔥 산불 위험도 예측 시스템")

# 기상 정보 입력
st.subheader("🌤️ 기상 정보")
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

# 최근 강수 상태
st.subheader("💧 최근 강수 상태")
recent_rain_level = st.radio(
    "지난 3일간 강수 상황:",
    options=[0, 1, 2, 3, 4, 5],
    format_func=lambda x: {
        0: "☀️ 매우 건조 - 3일 이상 비 없음",
        1: "🌤️ 건조 - 2-3일 전 약간의 비",
        2: "⛅ 보통 - 1-2일 전 소량 강수",
        3: "🌧️ 습윤 - 24시간 내 비",
        4: "🌧️ 매우 습윤 - 지속적 강수",
        5: "⛈️ 포화 - 연속 강수"
    }[x],
    index=1
)

# 풍향 입력
풍향_선택 = st.selectbox("풍향", ['북','북동','동','남동','남','남서','서','북서'])
풍향_맵 = {'북':0,'북동':1,'동':2,'남동':3,'남':4,'남서':5,'서':6,'북서':7}
풍향 = 풍향_맵[풍향_선택]

# 예측 실행
if st.button("🔥 화재 위험도 예측", type="primary"):
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
            base_risk, 강수량, recent_rain_level, 습도
        )
        
        # 결과 표시
        st.subheader("📊 예측 결과")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("기본 예측", f"{base_risk:.1f}%")
        with col2:
            st.metric("토양 습윤도", f"{moisture_factor:.1%}")
        with col3:
            st.metric("최종 위험도", f"{adjusted_risk:.1f}%")
        
        # 조정 효과
        reduction = base_risk - adjusted_risk
        if reduction > 0:
            st.success(f"💧 습윤 효과로 위험도 {reduction:.1f}%p 감소")
        
        # 위험도 레벨
        level, color = get_risk_level(adjusted_risk)
        st.markdown(f"### 🎯 종합 위험도: <span style='color:{color}; font-weight:bold'>{level}</span>", 
                   unsafe_allow_html=True)
        
        # 프로그레스 바
        st.progress(min(adjusted_risk / 100, 1.0))
        
        # 상세 분석
        with st.expander("📈 상세 분석"):
            st.markdown(f"""
            **기상 조건:**
            - 기온: {기온}°C, 습도: {습도}%, 풍속: {풍속}m/s
            - 강수량: {강수량}mm
            
            **습윤도 평가:**
            - 최근 강수: {recent_rain_level}/5 단계
            - 토양 습윤도: {moisture_factor:.1%}
            - 위험도 감소: {reduction:.1f}%p
            """)
            
    except Exception as e:
        st.error(f"예측 오류: {str(e)}")
