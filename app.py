import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

st.set_page_config(page_title="Sistema de Prediccion de Abandono Escolar", layout="wide")

# ---------- Carga, preprocesamiento y entrenamiento (una sola vez) ----------
@st.cache_resource
def preparar_todo():
    datos = fetch_ucirepo(id=697)
    X = datos.data.features
    y = datos.data.targets["Target"]
    cols = list(X.columns)

    le = LabelEncoder()
    y_cod = le.fit_transform(y)
    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_esc, y_cod, test_size=0.20, stratify=y_cod, random_state=42)
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_bal, y_bal)
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        eval_metric="mlogloss", random_state=42).fit(X_bal, y_bal)
    # MLP (red neuronal con scikit-learn, para evitar TensorFlow en el despliegue)
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                        alpha=0.001, max_iter=300, random_state=42).fit(X_bal, y_bal)

    # Metricas reales por modelo sobre el conjunto de prueba
    idx_dp = list(le.classes_).index("Dropout")
    def metricas(modelo):
        pred = modelo.predict(X_te)
        proba = modelo.predict_proba(X_te)
        return {
            "F1-score macro": round(f1_score(y_te, pred, average="macro"), 2),
            "Recall Dropout": round(recall_score(y_te, pred, labels=[idx_dp], average="macro"), 2),
            "Precision Dropout": round(precision_score(y_te, pred, labels=[idx_dp], average="macro"), 2),
            "AUC-ROC": round(roc_auc_score(y_te, proba, multi_class="ovr", average="macro"), 2),
        }
    met = {"Random Forest": metricas(rf), "XGBoost": metricas(xgb), "MLP": metricas(mlp)}

    # Datos para SHAP y para graficos
    return {"X": X, "y": y, "cols": cols, "le": le, "scaler": scaler,
            "rf": rf, "xgb": xgb, "mlp": mlp, "met": met,
            "X_te": X_te, "y_te": y_te, "idx_dp": idx_dp}

D = preparar_todo()
cols, le, scaler = D["cols"], D["le"], D["scaler"]
colores = {"Dropout": "#c0392b", "Enrolled": "#f39c12", "Graduate": "#27ae60"}

# ---------- Barra lateral ----------
st.sidebar.title("Sistema de Prediccion de Abandono Escolar")
st.sidebar.markdown("---")
st.sidebar.subheader("Navegacion")
seccion = st.sidebar.radio("", ["Exploracion de datos", "Prediccion",
                                "Explicabilidad", "Comparativa de modelos"])
st.sidebar.markdown("---")
st.sidebar.subheader("Configuracion del modelo")
modelo_sel = st.sidebar.selectbox("Modelo de prediccion",
                                  ["Random Forest", "XGBoost", "MLP — Red Neuronal"])
def modelo_obj():
    if modelo_sel.startswith("Random"): return D["rf"], "Random Forest"
    if modelo_sel.startswith("XGBoost"): return D["xgb"], "XGBoost"
    return D["mlp"], "MLP"

# ===================== SECCION 1: EXPLORACION DE DATOS =====================
if seccion == "Exploracion de datos":
    st.title("Exploracion de datos")
    st.write("Conjunto de datos: Realinho et al. (2022). Registros:", D["X"].shape[0],
             "| Variables:", D["X"].shape[1])
    conteo = D["y"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(conteo.index, conteo.values, color=["#27ae60", "#c0392b", "#f39c12"])
    ax.set_ylabel("Numero de estudiantes")
    ax.set_title("Distribucion de clases")
    for i, v in enumerate(conteo.values):
        ax.text(i, v + 20, str(v), ha="center")
    st.pyplot(fig)
    st.write("Estadisticas descriptivas de variables academicas:")
    st.dataframe(D["X"][["Curricular units 2nd sem (approved)",
                         "Curricular units 1st sem (approved)",
                         "Age at enrollment"]].describe().round(2))

# ===================== SECCION 2: PREDICCION =====================
elif seccion == "Prediccion":
    st.title("Modulo de Prediccion")
    st.write("Introduce los datos del estudiante para obtener la prediccion de riesgo de abandono.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Datos del estudiante")
        edad = st.number_input("Edad en la matricula", 17, 70, 19)
        genero = st.selectbox("Genero", ["Masculino", "Femenino"])
        beca = st.selectbox("Becario", ["No", "Si"])
        u2 = st.number_input("Unidades aprobadas (2o semestre)", 0, 26, 3)
        n2 = st.number_input("Calificacion media (2o semestre)", 0.0, 20.0, 9.1)
        u1 = st.number_input("Unidades aprobadas (1er semestre)", 0, 26, 4)
        n1 = st.number_input("Calificacion media (1er semestre)", 0.0, 20.0, 10.2)
        predecir = st.button("Obtener prediccion", type="primary")

    if predecir:
        x = np.zeros((1, len(cols)))
        def poner(n, v):
            if n in cols: x[0, cols.index(n)] = v
        poner("Age at enrollment", edad)
        poner("Gender", 1 if genero == "Masculino" else 0)
        poner("Scholarship holder", 1 if beca == "Si" else 0)
        poner("Curricular units 2nd sem (approved)", u2)
        poner("Curricular units 2nd sem (grade)", n2)
        poner("Curricular units 1st sem (approved)", u1)
        poner("Curricular units 1st sem (grade)", n1)
        x_esc = scaler.transform(x)
        modelo, nombre = modelo_obj()
        proba = modelo.predict_proba(x_esc)[0]
        clase = le.classes_[int(np.argmax(proba))]

        with c2:
            if clase == "Dropout":
                st.error("⚠️ RIESGO DE ABANDONO DETECTADO")
            st.markdown(f"### Clase predicha: :{'red' if clase=='Dropout' else ('orange' if clase=='Enrolled' else 'green')}[{clase}]")
            for cl, p in zip(le.classes_, proba):
                st.write(f"P({cl}): {p:.2f}")
                st.progress(float(p))
            st.caption(f"Modelo: {modelo_sel}")

        st.markdown("---")
        st.subheader("Explicacion SHAP de la prediccion")
        try:
            if nombre in ("Random Forest", "XGBoost"):
                expl = shap.TreeExplainer(modelo)
                sv = expl.shap_values(x_esc)
                vals = sv[0, :, D["idx_dp"]] if np.array(sv).ndim == 3 else sv[D["idx_dp"]][0]
            else:
                fondo = shap.kmeans(D["X_te"], 30)
                expl = shap.KernelExplainer(lambda d: modelo.predict_proba(d)[:, D["idx_dp"]], fondo)
                vals = expl.shap_values(x_esc, nsamples=100)[0]
            serie = pd.Series(vals, index=cols).sort_values(key=abs, ascending=False).head(6)
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            colores_b = ["#c0392b" if v > 0 else "#2980b9" for v in serie.values]
            ax2.barh(range(len(serie)), serie.values, color=colores_b)
            ax2.set_yticks(range(len(serie)))
            ax2.set_yticklabels(serie.index, fontsize=8)
            ax2.invert_yaxis()
            ax2.axvline(0, color="gray", linewidth=0.8)
            ax2.set_xlabel("Valor SHAP (impacto en Dropout)")
            st.pyplot(fig2)
            st.caption("Rojo: variables que aumentan el riesgo de Dropout. Azul: variables que lo reducen.")
        except Exception as e:
            st.info("La explicacion SHAP no pudo generarse para este modelo.")

        st.markdown("---")
        m = D["met"][nombre]
        cols_m = st.columns(4)
        cols_m[0].metric("F1-score macro", m["F1-score macro"])
        cols_m[1].metric("Recall Dropout", m["Recall Dropout"])
        cols_m[2].metric("Precision Dropout", m["Precision Dropout"])
        cols_m[3].metric("AUC-ROC", m["AUC-ROC"])

# ===================== SECCION 3: EXPLICABILIDAD =====================
elif seccion == "Explicabilidad":
    st.title("Explicabilidad global (SHAP)")
    st.write("Importancia global de las variables para el modelo XGBoost sobre el conjunto de prueba.")
    try:
        expl = shap.TreeExplainer(D["xgb"])
        sv = expl.shap_values(D["X_te"])
        sv_dp = sv[:, :, D["idx_dp"]] if np.array(sv).ndim == 3 else sv[D["idx_dp"]]
        imp = pd.Series(np.abs(sv_dp).mean(axis=0), index=cols).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(range(len(imp)), imp.values, color="#2980b9")
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Importancia media (|SHAP|)")
        st.pyplot(fig)
    except Exception:
        st.info("No se pudo generar el grafico SHAP global.")

# ===================== SECCION 4: COMPARATIVA DE MODELOS =====================
elif seccion == "Comparativa de modelos":
    st.title("Comparativa de modelos")
    st.write("Metricas de evaluacion sobre el conjunto de prueba (885 registros).")
    tabla = pd.DataFrame(D["met"]).T
    st.dataframe(tabla)
    st.write("Matriz de confusion:")
    modelo, nombre = modelo_obj()
    cm = confusion_matrix(D["y_te"], modelo.predict(D["X_te"]))
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(le.classes_, fontsize=8); ax.set_yticklabels(le.classes_, fontsize=8)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
    ax.set_title(nombre)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)
