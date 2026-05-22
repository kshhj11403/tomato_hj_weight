import joblib
import numpy as np
import pandas as pd  # sklearn 모델 종류에 따라 DataFrame이 필요할 수 있어 추가했습니다.
import streamlit as st

# --- 1. 페이지 제목 및 안내 ---
st.title("🌱 신체 정보가 아닌 스마트팜 환경 기반 착과율 예측")
st.write("스마트팜 내부 환경 데이터를 입력하면 예상 착과율을 예측합니다.")

# 사이드바 제목 (기존 코드의 형식을 유지)
st.sidebar.header("머신러닝 모델 설계 실습 (랜덤포레스트)")

# --- 2. 모델 로드 ---
# [수정] 요청하신 단일 모델 파일명(tomato_model.pkl)으로 변경했습니다.
tomato_model = joblib.load("tomato_model.pkl")

# --- 3. 재배 방식 선택 (기존의 성별 선택에 대응) ---
cultivation_type = st.radio("재배 방식 선택", ["토경재배 (흙)", "수경재배 (양액)"])

# --- 4. 조건별 입력 데이터 처리 (슬라이더 활용) ---
if cultivation_type == "토경재배 (흙)":
    # 기존: 키, 허리둘레, 엉덩이둘레 -> 환경 변수로 변경
    temp = st.slider("내부온도 (°C)", 10.0, 40.0, 25.0)
    humidity = st.slider("내부습도 (%)", 30.0, 100.0, 60.0)
    soil_temp = st.slider("지온 (°C)", 10.0, 35.0, 18.0)

    # 입력 데이터 정렬 (모델이 학습할 때 사용한 feature 순서대로 맞춰야 합니다)
    # 1) 만약 모델이 numpy 배열을 받는다면:
    X = np.array([[temp, humidity, soil_temp]])

    # 2) 만약 모델이 DataFrame(컬럼명 포함)을 요구한다면 아래 주석을 해제하세요.
    # X = pd.DataFrame([[temp, humidity, soil_temp]], columns=['내부온도', '내부습도', '지온'])

    model = tomato_model

else:
    # 수경재배는 지온 대신 다른 핵심 변수(예: 양액 온도나 EC 등)를 넣는 형태로 구성할 수 있습니다.
    temp = st.slider("내부온도 (°C)", 10.0, 40.0, 25.0)
    humidity = st.slider("내부습도 (%)", 30.0, 100.0, 60.0)
    ec_level = st.slider("급액 EC (dS/m)", 1.0, 5.0, 2.5)  # 예시 변수

    X = np.array([[temp, humidity, ec_level]])
    # DataFrame 형태 요구 시:
    # X = pd.DataFrame([[temp, humidity, ec_level]], columns=['내부온도', '내부습도', '급액EC'])

    model = tomato_model

# --- 5. 예측 버튼 및 결과 출력 ---
if st.button("착과율 예측하기"):
    prediction = model.predict(X)

    # 예측 결과 시각화 (출력 포맷 유지)
    st.success(f"예측 착과율 : {prediction[0]:.1f} %")