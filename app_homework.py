import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우에서 한글 지원

# 데이터 확인 및 전처리
# 데이터 불러오기 (pandas 활용)
df = pd.read_csv("data/Maternal Health Risk Data Set.csv")

# step01.
# 결측치, 이상치 확인 및 처리
# 이 예제에서는 직접적인 결측치나 이상치를 처리하지 않음.
# 하지만 필요한 경우 결측치를 처리하는 코드를 추가할 수 있습니다.
# 예를 들어: df = df.dropna()  또는 df.fillna(0) 등

# step02. 데이터 타입 변환 및 정리
# Label Encoding(레이블 인코딩)은 문자열(범주형 데이터)을 숫자로 변환하는 방법입니다.
le = LabelEncoder()
df["RiskLevel"] = le.fit_transform(df["RiskLevel"])

# step03. 특성과 타겟변수 분리
# 특성과 타겟 분리
# X (입력 데이터, Feature Set):
# Age (나이), SystolicBP (수축기 혈압), DiastolicBP (이완기 혈압),
# BS (혈당), BodyTemp (체온), HeartRate (심박수)
# y (타겟 데이터, Label): RiskLevel (위험 수준)
X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = df['RiskLevel']

# step04. 데이터 정규화(Scaling, 표준화)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# step05. 데이터셋 분할(train-test Split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# step06. 머신러닝 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 특성 중요도 출력
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
print(feature_importances)

# step07. 예측 및 성능평가(Model Prediction & Evaluation)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=le.classes_)

# Streamlit UI 디자인
st.set_page_config(page_title="Maternal Health Risk Dashboard", layout="wide")

# 사이드바 메뉴
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Go to", ["Home", "Data Analysis", "EDA", "Model Performance"])

# 홈화면
def home():
    st.title("Maternal Health Risk Dashboard")
    st.markdown("""
    - **Age**: 나이
    - **SystolicBP**: 수축기 혈압
    - **DiastolicBP**: 이완기 혈압
    - **BS**: 혈당
    - **BodyTemp**: 체온
    - **HeartRate**: 심박수
    - **RiskLevel**: 위험 수준 (Low risk, Mid risk, High risk)
    """)

# 데이터 분석 화면
def data_analysis():
    st.title("데이터 분석")

    st.markdown("""
    ### 데이터 분석 개요
    임산부의 건강에 영향을 미치는 주요 변수들에 대해 분석을 진행하였고, 각 변수들이 위험 수준과 어떻게 상관관계를 가지는지 분석하였습니다.
    
    주요 분석 결과:
    - **위험 수준에 따른 나이 및 혈압**: 나이가 많을수록, 특히 수축기 혈압(Systolic BP)이 높을수록 위험 수준이 높아지는 경향이 있습니다.
    - **심박수와 체온**: 심박수와 체온이 급격히 상승하는 경우도 위험 수준이 높다는 패턴이 관찰되었습니다.
    - **특성 중요도**: 모델 학습 후, 각 변수들이 위험 수준을 예측하는데 얼마나 중요한지에 대한 정보가 제공되었습니다.

    ### 주요 변수들
    - **나이 (Age)**: 나이가 증가할수록 위험 수준이 증가하는 경향이 있습니다.
    - **수축기 혈압 (Systolic BP)**: 수축기 혈압이 높을수록 위험 수준이 높습니다.
    - **이완기 혈압 (Diastolic BP)**: 이완기 혈압 역시 위험 수준과 긍정적인 상관관계를 가집니다.
    - **체온 (Body Temp)**: 체온이 급격히 상승하는 경우 위험 수준이 높습니다.

    ### 데이터 분석 결과 요약:
    - 나이와 혈압은 건강 위험 수준을 예측하는 중요한 변수로, 특정 수준 이상일 때 위험도가 급격히 증가합니다.
    - 심박수와 체온 역시 중요한 변수로, 임산부의 건강 상태를 반영하는 중요한 지표로 작용합니다.
    """)

    st.markdown("### 특성 중요도 분석")
    st.write("각 특성의 중요도를 기반으로 모델이 예측을 어떻게 진행했는지 확인해보겠습니다.")
    st.table(feature_importances)

# EDA(데이터시각화화면)
def eda():
    st.title("데이터 시각화")
    chart_tabs = st.tabs(["histgram", "boxplot", "hitmap"])

    # 히스토그램 (Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate 분포)
    with chart_tabs[0]:
        st.subheader("연령, 혈압, 혈당, 체온, 심박수 분포")
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        columns = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
        for i, col in enumerate(columns):
            ax = axes[i // 2, i % 2]
            sns.histplot(df[col], bins=20, kde=True, ax=ax)
            ax.set_title(col)
        
        # 그래프 간 간격을 자동으로 조정
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        st.pyplot(fig)

    # 박스플롯 (위험 수준별 각 변수)
    with chart_tabs[1]:
        st.subheader("위험 수준에 따른 변수 분포")
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for i, col in enumerate(columns):
            ax = axes[i // 2, i % 2]
            sns.boxplot(data=df, x="RiskLevel", y=col, ax=ax)
            ax.set_title(f"{col}에 따른 위험 수준 분포")
        
        # 그래프 간 간격을 자동으로 조정
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        st.pyplot(fig)

    # 변수 간 상관관계 분석 (히트맵)
    with chart_tabs[2]:
        st.subheader("상관관계 히트맵")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        
        # 레이아웃 자동 맞춤
        plt.tight_layout()
        st.pyplot(fig)

# 모델 성능 평가 화면
def model_performance():
    st.title("모델 성능 평가")
    st.write(f'### 모델 정확도: {accuracy:.2f}')
    st.text(classification_rep)

# 메뉴 선택에 따른 화면 전환
if menu == "Home":
    home()
elif menu == "Data Analysis":
    data_analysis()
elif menu == "EDA":
    eda()
elif menu == "Model Performance":
    model_performance()

