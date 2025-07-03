import streamlit as st ; import joblib ; import pandas as pd ; import catboost

st.markdown("""
<style>
    .stNumberInput > div > div > input {
        height: 2.5rem;
    }
    .stSelectbox > div > div > div {
        height: 2.5rem;
    }
    .stRadio > div {
        gap: 0.5rem;
    }
    div[data-testid="column"] {
        padding: 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(): return joblib.load('m0627.pkl')

def adjust_risk(base_risk, rainfall_mm, recent_rain_level):
    rainfall_reduction = min(0.99, rainfall_mm * 0.2)
    recent_rain_reduction = {0:0.0, 1:0.05, 2:0.1, 3:0.3, 4:0.5, 5:0.7}[recent_rain_level]
    total_reduction = min(rainfall_reduction + recent_rain_reduction, 0.99)
    adjusted_risk = base_risk * (1 - total_reduction)
    return adjusted_risk, total_reduction

def get_risk_level(risk):
    if risk >= 85: return "🚨 극도로 높음", "darkred"
    elif risk >= 65: return "🔥 높음", "red"
    elif risk >= 50: return "⚠️ 보통", "orange"
    elif risk >= 30: return "🔶 낮음", "gold"
    elif risk >= 20: return "💚 매우낮음", "green"
    else: return "✅ 극도로 낮음", "blue"

model = load_model()
st.title("🔥 산불 위험도 예측 시스템")
st.caption("스마트폰 날씨앱 데이터를 입력하세요")
st.subheader("🌤️ 기상 정보")
col1, col2, col3 = st.columns([1.2, 1.2, 2], gap="large")

with col1:
    기온 = st.number_input("기온 (°C)", value=25.0, step=1.0, key="temp")
    풍속 = st.number_input("풍속 (m/s)", value=2.0, step=1.0, key="wind_speed")
    이슬점온도 = st.number_input("이슬점온도 (°C)", value=15.0, step=1.0, key="dew_point")
    월 = st.selectbox("월", list(range(1,13)), index=4, key="month")
with col2:
    강수량 = st.number_input("현재 강수량 (mm)", value=0.0, step=1.0, min_value=0.0, key="rainfall")
    습도 = st.number_input("습도 (%)", value=50.0, step=1.0, min_value=0.0, max_value=100.0, key="humidity")
    기압 = st.number_input("기압 (hPa)", value=1013.0, step=1.0, key="pressure")
    시간 = st.selectbox("시간", list(range(24)), index=12, key="hour")
with col3:
    풍향 = st.selectbox("풍향", ['북','북동','동','남동','남','남서','서','북서'], key="wind_dir")
    st.markdown("**최근 3일간 눈/비/지면 상태:**")
    recent_rain_level = st.radio("", options=[0, 1, 2, 3, 4, 5], format_func=lambda x: {0: "☀️ 매우 건조 - 3일 이상 비·눈 없음", 1: "🌤️ 건조 - 2-3일 전 약간의 비", 2: "⛅ 보통 - 1-2일 전 비", 3: "🌧️ 습윤 - 24시간 내 비", 4: "❄️ 매우 습윤 - 최근 많은 눈비", 5: "💧 포화 - 강우나 폭설"}[x], index=1, key="rain_level")

if st.button("🔥 화재 위험도 예측", type="primary"):
    X = pd.DataFrame([[기온, 강수량, 풍속, 습도, 이슬점온도, 기압, 월, 시간, 풍향]], columns=['기온','강수량','풍속','습도','이슬점온도','기압','월','시간','풍향'])
    for c in ['월','시간','풍향']: X[c] = X[c].astype(str)
    try:
        pool = catboost.Pool(X, cat_features=[6, 7, 8])
        proba = model.predict_proba(pool)[0][1]
        base_risk = proba * 100
        adjusted_risk, total_reduction = adjust_risk(base_risk, 강수량, recent_rain_level)
        st.subheader("📊 예측 결과")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("기본 예측", f"{base_risk:.1f}%")
    
        with col3: st.metric("최종 위험도", f"{adjusted_risk:.1f}%")
        reduction = base_risk - adjusted_risk
        
        level, color = get_risk_level(adjusted_risk)
        st.markdown(f"### 🎯 종합 위험도: <span style='color:{color}; font-weight:bold'>{level}</span>", unsafe_allow_html=True)
        st.progress(min(adjusted_risk / 100, 1.0))
        st.markdown("""
        **모델 성능:**  
        - Training (Class 1: Precision 0.99, Recall 0.98)  
        - Test1 (Class 1: Precision 0.05, Recall 1.00)  
        - Test2 (Class 1: Precision 0.12, Recall 0.84)  
        
        **하이퍼파라미터:**  
        - depth: 8, learning_rate: 0.09846, l2_leaf_reg: 0.8032, iterations: 358
        """)
    except Exception as e: st.error(f"예측 오류: {str(e)}")
