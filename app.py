import streamlit as st; import joblib; import pandas as pd; import catboost

st.markdown("""<style>
.stNumberInput > div > div > input, .stSelectbox > div > div > div { height: 2.5rem; }
.stRadio > div { gap: 0.5rem; }
div[data-testid="column"] { padding: 0 1rem; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model(): return joblib.load('m0627.pkl')

def adjust_risk(base_risk, rain_mm, rain_level):
    rain_red = min(0.99, rain_mm * 0.2)
    recent_red = [0.001, 0.15, 0.35, 0.7, 0.9][rain_level]
    total_red = min(rain_red + recent_red, 0.99)
    return base_risk * (1 - total_red), total_red

def get_risk_level(risk):
    levels = [(85, "🚨 매우 높음", "darkred"), (65, "🔥 높음", "red"), (50, "⚠️ 보통", "orange"), (30, "🔶 낮음", "gold")]
    for threshold, label, color in levels:
        if risk >= threshold: return label, color
    return "💚 매우 낮음", "green"

model = load_model()
st.title("🔥 산불 위험도 예측 시스템")
st.caption("스마트폰 날씨앱 데이터를 입력하세요")
st.subheader("🌤️ 기상 정보")

col1, col2, col3 = st.columns([1.2, 1.2, 2], gap="large")
with col1:
    temp = st.number_input("기온 (°C)", value=0.0, step=1.0)
    wind = st.number_input("풍속 (m/s)", value=0.0, step=1.0)
    dew = st.number_input("이슬점온도 (°C)", value=0.0, step=1.0)
    month = st.selectbox("월", list(range(1,13)), index=0)
with col2:
    rain = st.number_input("현재 강수량 (mm)", value=0.0, step=1.0, min_value=0.0)
    humid = st.number_input("습도 (%)", value=0.0, step=1.0, min_value=0.0, max_value=100.0)
    press = st.number_input("기압 (hPa)", value=0.0, step=1.0)
    hour = st.selectbox("시간", list(range(24)), index=0)
with col3:
    wind_dir = st.selectbox("풍향", ['북','북동','동','남동','남','남서','서','북서'])
    st.markdown("**최근 3일간 눈/비/지면 상태:**")
    rain_opts = ["🌤️ 건조 - 2-3일 최근 비/눈 없음(바닥이 마른 상태)", "⛅ 보통 - 1-2일 전 비/눈(바닥이 조금 젖어있음)", "🌧️ 습윤 - 24시간 내 비/눈(바닥이 많이 젖어있음)", "❄️ 매우 습윤 - 최근 많은 비/눈(곳곳에 물웅덩이/잔설 존재)", "💧 포화 - 지금도 비/눈이 오고있음"]
    rain_level = st.radio("", options=list(range(5)), format_func=lambda x: rain_opts[x], index=1)

if st.button("🔥 화재 위험도 예측", type="primary"):
    X = pd.DataFrame([[temp, rain, wind, humid, dew, press, month, hour, wind_dir]], columns=['기온','강수량','풍속','습도','이슬점온도','기압','월','시간','풍향'])
    for c in ['월','시간','풍향']: X[c] = X[c].astype(str)
    try:
        pool = catboost.Pool(X, cat_features=[6, 7, 8])
        proba = model.predict_proba(pool)[0][1]
        base_risk = proba * 100
        adj_risk, _ = adjust_risk(base_risk, rain, rain_level)
        
        st.subheader("📊 예측 결과")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("기본 예측", f"{base_risk:.1f}%")
        with col3: st.metric("최종 위험도", f"{adj_risk:.1f}%")
        
        level, color = get_risk_level(adj_risk)
        st.markdown(f"### 🎯 종합 위험도: <span style='color:{color}; font-weight:bold'>{level}</span>", unsafe_allow_html=True)
        st.progress(min(adj_risk / 100, 1.0))
        
        st.markdown("""
        **모델 정보 : CatBoost + optuna**
        - Training (Class 1: Precision 0.99, Recall 0.98) 3대1비율
        - Test1 (Class 1: Precision 0.05, Recall 1.00) 화재비율 2% 가정
        - Test2 (Class 1: Precision 0.12, Recall 0.84) 화재비율 5% 가정
        - 문턱값 : 0.65
        **하이퍼파라미터:**
        - depth: 8, learning_rate: 0.09846, l2_leaf_reg: 0.8032, iterations: 358
        """)
    except Exception as e: st.error(f"예측 오류: {str(e)}")
