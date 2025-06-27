import streamlit as st
import joblib
import pandas as pd
import catboost

@st.cache_resource
def load_model():
    return joblib.load('m0627.pkl')

model = load_model()

st.title("🔥 산불 위험도 예측 시스템")

기온 = st.number_input("기온 (°C)", value=25.0)
강수량 = st.number_input("강수량 (mm)", value=0.0)
풍속 = st.number_input("풍속 (m/s)", value=2.0)
습도 = st.number_input("습도 (%)", value=50.0)
이슬점온도 = st.number_input("이슬점온도 (°C)", value=15.0)
기압 = st.number_input("기압 (hPa)", value=1013.25)
월 = st.selectbox("월", list(range(1,13)), index=0)
시간 = st.selectbox("시간", list(range(24)), index=12)
풍향_선택 = st.selectbox("풍향", ['북','북동','동','남동','남','남서','서','북서'])

풍향_맵 = {'북':0,'북동':1,'동':2,'남동':3,'남':4,'남서':5,'서':6,'북서':7}
풍향 = 풍향_맵[풍향_선택]

if st.button("🔥 화재 위험도 예측"):
    X = pd.DataFrame([[
        기온, 강수량, 풍속, 습도, 이슬점온도, 기압, 월, 시간, 풍향
    ]], columns=['기온','강수량','풍속','습도','이슬점온도','기압','월','시간','풍향'])

    for c in ['월','시간','풍향']:
        X[c] = X[c].astype(str)

    pool = catboost.Pool(X, cat_features=[6,7,8])

    proba = model.predict_proba(pool)[0][1]
    st.write(f"화재 발생 확률: {proba:.4f}")

    risk = proba * 100
    st.write(f"위험도 (0~100): {risk:.1f}")

    if risk >= 85:
        level = "🚨 매우 높음"
        color = "red"
    elif risk >= 65:
        level = "⚠️ 높음"
        color = "orange"
    elif risk >= 45:
        level = "🔶 보통"
        color = "yellow"
    elif risk >= 25:
        level = "💚 낮음"
        color = "green"
    else:
        level = "✅ 매우 낮음"
        color = "blue"

    st.markdown(f"### 위험도: <span style='color:{color}'>{level} ({risk:.1f}%)</span>", unsafe_allow_html=True)

    # 계산기 하단 설명 추가
    st.markdown("""
    ---
    <h4>📊 모델 성능 정보</h4>
    
    <div style="margin-bottom: 15px;">
        <strong>🔹 훈련 데이터 성능:</strong><br>
        정확도 87.0% | 화재 탐지율 91.3% | 정밀도 91.4%
    </div>
    
    <div style="margin-bottom: 15px;">
        <strong>🔹 실제 테스트 성능:</strong><br>
        • Test1 (화재 2%): 탐지율 90.0%, 정밀도 5.0%<br>
        • Test2 (화재 5%): 탐지율 84.0%, 정밀도 13.6%
    </div>
    
    <div class="performance-grid" style="display:flex; gap:20px; margin-bottom: 15px;">
        <div>
            <strong>사용 모델</strong><br>
            CatBoost
        </div>
        <div>
            <strong>최적화</strong><br>
            Optuna
        </div>
        <div>
            <strong>해석성</strong><br>
            SHAP
        </div>
        <div>
            <strong>특징</strong><br>
            고탐지율
        </div>
    </div>
    
    <div style="font-size: 12px; opacity: 0.9;">
        <strong>하이퍼파라미터:</strong> depth=4, learning_rate=0.017, l2_leaf_reg=0.105, iterations=379<br>
    </div>
    """, unsafe_allow_html=True)
