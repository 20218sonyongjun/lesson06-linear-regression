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

# 1. 전체 기간 회귀 분석 모델 학습
X_all = df_annual[["연도"]]
y_all = df_annual["연평균기온"]
model_all = LinearRegression()
model_all.fit(X_all, y_all)

slope_all = model_all.coef_[0]  # 1년당 기온 변화
slope_all_100y = slope_all * 100  # 100년당 기온 변화

# 2. 최근 20년 기간 회귀 분석 모델 학습 (예: 2006년 ~ 2025년)
recent_year_threshold = end_year - 20
df_recent = df_annual[df_annual["연도"] >= recent_year_threshold]
X_recent = df_recent[["연도"]]
y_recent = df_recent["연평균기온"]
model_recent = LinearRegression()
model_recent.fit(X_recent, y_recent)

slope_recent = model_recent.coef_[0]  # 1년당 기온 변화
slope_recent_100y = slope_recent * 100  # 100년당 기온 변화

# --- 기울기 비교 화면 표시 ---
st.subheader("📈 기온 상승 속도 비교 (100년 기준)")
col_a, col_b = st.columns(2)

with col_a:
  st.markdown(
      f"<div style='padding: 20px; border-radius: 10px; background-color:"
      " #f0f2f6; text-align: center;'><h3>전체 기간 ({start_year}~{end_year}년)</h3>"
      f"<p style='font-size: 36px; font-weight: bold; color:"
      f" #1f77b4;'>+{slope_all_100y:.2f} °C / 100년</p><p>(연간"
      f" +{slope_all:.4f}°C)</p></div>",
      unsafe_allow_html=True,
  )

with col_b:
  st.markdown(
      f"<div style='padding: 20px; border-radius: 10px; background-color:"
      " #f0f2f6; text-align: center;'><h3>최근 20년"
      f" ({recent_year_threshold}~{end_year}년)</h3><p style='font-size: 36px;"
      f" font-weight: bold; color: #ff7f0e;'>+{slope_recent_100y:.2f} °C /"
      f" 100년</p><p>(연간 +{slope_recent:.4f}°C)</p></div>",
      unsafe_allow_html=True,
  )

st.markdown("")

# 상관계수 계산
correlation = df_annual["연도"].corr(df_annual["연평균기온"])

# 전체 기간 회귀 직선용 예측 값 생성
df_annual["예측기온"] = model_all.predict(X_all)

# 시각화 (Plotly)
fig = px.scatter(
    df_annual,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 산점도 및 전체 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    opacity=0.7,
)

# 전체 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=df_annual["연도"],
        y=df_annual["예측기온"],
        mode="lines",
        name="전체 회귀 직선",
        line=dict(color="red", width=2),
    )
)

fig.update_layout(
    xaxis_title="연도", yaxis_title="연평균기온 (°C)", template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# 상관계수 안내
st.info(
    f"📊 **전체 기간 연도와 연평균기온 간의 상관계수 ($r$):** `{correlation:.4f}`"
)

st.markdown("---")

# 연도 슬라이더로 예측 기온 확인 (전체 기간 학습 모델 기준)
st.subheader("🔮 특정 연도 예상 기온 확인")
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1,
)

# 선택된 연도의 예상 기온 계산 (전체 기간 모델 기준)
predicted_temp = model_all.predict([[selected_year]])[0]

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
