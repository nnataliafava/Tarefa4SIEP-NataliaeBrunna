import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== CONFIGURAÇÃO DA PÁGINA =====================
st.set_page_config(page_title="Dashboard SIEP", layout="wide")

# ===================== LOGO + TÍTULO BONITO =====================
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://raw.githubusercontent.com/nnataliafava/tarefa4-siep-brunna/main/images.png' width='250'/>
        <h1 style='font-size: 40px; margin-top: 10px;'>📊 Previsão de Reclamações com Modelos Supervisionados</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ===================== UPLOAD =====================
st.sidebar.header("📁 Upload dos Dados")
file = st.sidebar.file_uploader("Selecione o arquivo .csv", type="csv")

if file:
    df = pd.read_csv(file, sep=";")
    st.subheader("🔍 Visualização Inicial")
    st.dataframe(df.head())

    # ===================== FILTROS INTERATIVOS =====================
    if "Education" in df.columns and "Marital_Status" in df.columns:
        educ = st.sidebar.multiselect("Educação", df["Education"].unique(), df["Education"].unique())
        status = st.sidebar.multiselect("Estado Civil", df["Marital_Status"].unique(), df["Marital_Status"].unique())
        df = df[df["Education"].isin(educ) & df["Marital_Status"].isin(status)]
        st.write(f"Base com {df.shape[0]} registros após filtros.")
    else:
        st.warning("⚠️ Colunas 'Education' e 'Marital_Status' não encontradas. Verifique os nomes no CSV.")

    # ===================== PREPARAÇÃO DOS DADOS =====================
    df.dropna(inplace=True)
    target = "Complain"
    y = df[target]
    X = df.drop(columns=[target, "ID", "Z_CostContact", "Z_Revenue", "Dt_Customer"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)

    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=10)
    X_rfe = rfe.fit_transform(X, y)
    selected = X.columns[rfe.support_]
    st.sidebar.success("✅ Variáveis selecionadas: " + ", ".join(selected))

    # SMOTE + Split
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X[selected], y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===================== TREINAMENTO =====================
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ===================== MÉTRICAS =====================
    st.subheader("📈 Avaliação do Modelo")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        st.metric("AUC", f"{roc_auc_score(y_test, y_prob):.2f}")
    with col2:
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    # Matriz de Confusão
    st.subheader("📊 Matriz de Confusão")
    fig1, ax1 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax1)
    st.pyplot(fig1)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    st.subheader("🚀 Curva ROC")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.legend()
    st.pyplot(fig2)

    # Coeficientes interpretáveis
    st.subheader("📌 Interpretação dos Coeficientes")
    coef_df = pd.DataFrame({
        "Variável": selected,
        "Coeficiente": model.coef_[0],
        "Odds Ratio": np.exp(model.coef_[0])
    }).sort_values(by="Coeficiente", key=abs, ascending=False)
    st.dataframe(coef_df)

else:
    st.info("👈 Faça o upload do arquivo CSV para começar.")
