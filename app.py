import streamlit as st
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Prediccion de abandono escolar", layout="wide")

@st.cache_resource
def preparar():
    datos = fetch_ucirepo(id=697)
    X = datos.data.features
    y = datos.data.targets["Target"]
    cols = list(X.columns)
    le = LabelEncoder(); y_cod = le.fit_transform(y)
    scaler = StandardScaler(); X_esc = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_esc, y_cod, test_size=0.20,
                                              stratify=y_cod, random_state=42)
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_bal, y_bal)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        eval_metric="mlogloss", random_state=42).fit(X_bal, y_bal)
    return rf, xgb, scaler, le, cols

rf, xgb, scaler, le, cols = preparar()

st.sidebar.title("Configuracion")
modelo_sel = st.sidebar.selectbox("Modelo de prediccion", ["XGBoost", "Random Forest"])
st.sidebar.caption("Sistema de prediccion temprana del abandono escolar (TFM)")

st.title("Prediccion temprana del abandono escolar")
st.write("Introduce los datos del estudiante y pulsa Predecir.")
st.subheader("Datos del estudiante")
c1, c2, c3 = st.columns(3)
with c1:
    u2 = st.number_input("Unidades aprobadas 2o semestre", 0, 26, 5)
    u1 = st.number_input("Unidades aprobadas 1er semestre", 0, 26, 5)
with c2:
    matricula = st.selectbox("Matricula al dia", ["Si", "No"])
    edad = st.number_input("Edad en la matricula", 17, 70, 20)
with c3:
    beca = st.selectbox("Becario", ["Si", "No"])
    nota2 = st.number_input("Calificacion media 2o semestre", 0.0, 20.0, 12.0)

if st.button("Predecir", type="primary"):
    x = np.zeros((1, len(cols)))
    def poner(n, v):
        if n in cols: x[0, cols.index(n)] = v
    poner("Curricular units 2nd sem (approved)", u2)
    poner("Curricular units 1st sem (approved)", u1)
    poner("Tuition fees up to date", 1 if matricula == "Si" else 0)
    poner("Age at enrollment", edad)
    poner("Scholarship holder", 1 if beca == "Si" else 0)
    poner("Curricular units 2nd sem (grade)", nota2)
    x_esc = scaler.transform(x)
    proba = (xgb if modelo_sel == "XGBoost" else rf).predict_proba(x_esc)[0]
    clase = le.classes_[int(np.argmax(proba))]
    colores = {"Dropout": "ROJO", "Enrolled": "AMARILLO", "Graduate": "VERDE"}
    st.subheader("Resultado")
    st.markdown("## " + clase + "  (" + colores.get(clase, "") + ")")
    st.table(pd.DataFrame({"Clase": le.classes_,
                           "Probabilidad": [f"{p*100:.1f}%" for p in proba]}))
