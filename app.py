import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="서울 기온 예측기", page_icon="🌡️", layout="wide"
)


@st.cache_data
def load_and_preprocess_data():
  # 데이터 로드
  url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
  df = pd.read_csv(url, encoding="utf-8")

  # 날짜 열을 datetime 형식으로 변환
  df["날짜"] = pd.to_datetime(df["날짜"])
  df["연도"] = df["날짜"].dt.year

  # 1. 수업 기준 기간(2025년까지) 필터링
  df = df[df["연도"] <= 2025]

  # 2. 연도별 관측일 수 및 평균기온 계산
  annual_stats = (
      df.groupby("연도")
      .agg(관측일수=("평균기온", "count"), 연평균기온=("평균기온", "mean"))
      .reset_index()
  )

  # 3. 관측일이 300일 미만인 해 제외
  filtered_df = annual_stats[annual_stats["관측일수"] >= 300].dropna(
      subset=["연평균기온"]
  )

  return filtered_df


# 데이터 불러오기
try:
  df_annual = load_and_preprocess_data()
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()

# UI 구성
st.title("🌡️ 서울 기온 예측기")
st.markdown(
    "서울의 역사적 기온 데이터를 바탕으로 연도별 평균기온의 변화를 확인하고, 회귀 분석을"
    " 통해 미래 기온을 예측해보세요."
)

# 데이터 분석 정보 요약
num_years = len(df_annual)
start_year = int(df_annual["연도"].min())
end_year = int(df_annual["연도"].max())

col1, col2, col3 = st.columns(3)
with col1:
  st.metric(label="분석에 사용된 해의 개수", value=f"{num_years}개")
with col2:
  st.metric(label="시작 연도", value=f"{start_year}년")
with col3:
  st.metric(label="끝 연도", value=f"{end_year}년")

st.markdown("---")

# 회귀 분석 모델 학습
X = df_annual[["연도"]]
y = df_annual["연평균기온"]

model = LinearRegression()
model.fit(X, y)

# 상관계수 계산
correlation = df_annual["연도"].corr(df_annual["연평균기온"])

# 회귀 직선용 예측 값 생성
df_annual["예측기온"] = model.predict(X)

# 시각화 (Plotly)
fig = px.scatter(
    df_annual,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 산점도 및 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    opacity=0.7,
)

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=df_annual["연도"],
        y=df_annual["예측기온"],
        mode="lines",
        name="회귀 직선",
        line=dict(color="red", width=2),
    )
)

fig.update_layout(
    xaxis_title="연도", yaxis_title="연평균기온 (°C)", template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# 상관계수 안내
st.info(
    f"📊 **연도와 연평균기온 간의 상관계수 ($r$):** `{correlation:.4f}`"
    " (1에 가까울수록 강한 양의 상관관계가 있음을 의미합니다.)"
)

st.markdown("---")

# 연도 슬라이더로 예측 기온 확인
st.subheader("🔮 특정 연도 예상 기온 확인")
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 선택된 연도의 예상 기온 계산
predicted_temp = model.predict([[selected_year]])[0]

# 크게 표시
st.markdown(
    f"<h3 style='text-align: center; color: #ff4b4b;'>{selected_year}년 서울의"
    f" 예상 연평균기온</h3>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<h1 style='text-align: center; font-size: 60px;'>{predicted_temp:.2f}"
    " °C</h1>",
    unsafe_allow_html=True,
)
