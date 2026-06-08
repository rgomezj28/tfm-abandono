import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

st.set_page_config(page_title="Sistema de Predicción de Abandono Escolar",
                   layout="wide")

COL = {"Dropout": "#c0392b", "Enrolled": "#f39c12", "Graduate": "#27ae60"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 110})

# Traduccion de nombres de variables a castellano (solo para mostrar)
TRAD = {
    "Curricular units 2nd sem (approved)": "Unidades aprobadas (2º semestre)",
    "Curricular units 1st sem (approved)": "Unidades aprobadas (1er semestre)",
    "Curricular units 2nd sem (grade)": "Calificación media (2º semestre)",
    "Curricular units 1st sem (grade)": "Calificación media (1er semestre)",
    "Curricular units 2nd sem (enrolled)": "Unidades matriculadas (2º semestre)",
    "Curricular units 1st sem (enrolled)": "Unidades matriculadas (1er semestre)",
    "Curricular units 2nd sem (evaluations)": "Evaluaciones (2º semestre)",
    "Curricular units 1st sem (evaluations)": "Evaluaciones (1er semestre)",
    "Curricular units 2nd sem (credited)": "Unidades convalidadas (2º semestre)",
    "Curricular units 1st sem (credited)": "Unidades convalidadas (1er semestre)",
    "Curricular units 2nd sem (without evaluations)": "Unidades sin evaluar (2º semestre)",
    "Curricular units 1st sem (without evaluations)": "Unidades sin evaluar (1er semestre)",
    "Tuition fees up to date": "Matrícula al día",
    "Course": "Titulación",
    "Age at enrollment": "Edad en la matrícula",
    "Scholarship holder": "Becario",
    "Gender": "Género",
    "Unemployment rate": "Tasa de desempleo",
    "Inflation rate": "Tasa de inflación",
    "GDP": "PIB",
    "Admission grade": "Nota de admisión",
    "Displaced": "Desplazado",
    "Debtor": "Deudor",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Mother's qualification": "Nivel educativo de la madre",
    "Father's qualification": "Nivel educativo del padre",
    "Previous qualification (grade)": "Nota de cualificación previa",
    "Previous qualification": "Cualificación previa",
    "Application mode": "Modo de solicitud",
    "Application order": "Orden de solicitud",
    "Marital Status": "Estado civil",
    "Nacionality": "Nacionalidad",
    "International": "Internacional",
    "Daytime/evening attendance": "Asistencia diurna/nocturna",
    "Educational special needs": "Necesidades educativas especiales",
}
TRAD_CLASE = {"Dropout": "Abandono", "Enrolled": "Matriculado", "Graduate": "Graduado"}

def trad(nombre):
    return TRAD.get(nombre, nombre)

def trad_clase(c):
    return TRAD_CLASE.get(c, c)


# ------------------------------------------------------------------
# Carga de datos, preprocesamiento y entrenamiento (una sola vez)
# ------------------------------------------------------------------
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

    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_bal, y_bal)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                        eval_metric="mlogloss", random_state=42).fit(X_bal, y_bal)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                        alpha=0.001, max_iter=300, random_state=42).fit(X_bal, y_bal)

    idx_dp = list(le.classes_).index("Dropout")

    def metricas(modelo):
        pred = modelo.predict(X_te)
        proba = modelo.predict_proba(X_te)
        return {
            "F1-score macro": round(f1_score(y_te, pred, average="macro"), 2),
            "Recall Abandono": round(recall_score(y_te, pred, labels=[idx_dp],
                                                   average="macro"), 2),
            "Precisión Abandono": round(precision_score(y_te, pred, labels=[idx_dp],
                                                        average="macro"), 2),
            "AUC-ROC": round(roc_auc_score(y_te, proba, multi_class="ovr",
                                           average="macro"), 2),
        }

    met = {"Random Forest": metricas(rf),
           "XGBoost": metricas(xgb),
           "MLP": metricas(mlp)}

    return {"X": X, "y": y, "cols": cols, "le": le, "scaler": scaler,
            "rf": rf, "xgb": xgb, "mlp": mlp, "met": met,
            "X_te": X_te, "y_te": y_te, "idx_dp": idx_dp}

D = preparar_todo()
cols, le, scaler = D["cols"], D["le"], D["scaler"]


def modelo_obj(sel):
    if sel.startswith("Random"):
        return D["rf"], "Random Forest"
    if sel.startswith("XGBoost"):
        return D["xgb"], "XGBoost"
    return D["mlp"], "MLP"


def vector_estudiante(valores):
    x = np.zeros((1, len(cols)))
    for n, v in valores.items():
        if n in cols:
            x[0, cols.index(n)] = v
    return scaler.transform(x)


def shap_local(modelo, nombre, x_esc):
    if nombre in ("Random Forest", "XGBoost"):
        expl = shap.TreeExplainer(modelo)
        sv = expl.shap_values(x_esc)
        vals = sv[0, :, D["idx_dp"]] if np.array(sv).ndim == 3 else sv[D["idx_dp"]][0]
    else:
        fondo = shap.kmeans(D["X_te"], 30)
        expl = shap.KernelExplainer(
            lambda d: modelo.predict_proba(d)[:, D["idx_dp"]], fondo)
        vals = expl.shap_values(x_esc, nsamples=100)[0]
    serie = pd.Series(vals, index=[trad(c) for c in cols])
    return serie.sort_values(key=abs, ascending=False).head(6)


# ------------------------------------------------------------------
# Barra lateral
# ------------------------------------------------------------------
st.sidebar.title("Sistema de Predicción de Abandono Escolar")
st.sidebar.markdown("---")
st.sidebar.subheader("Navegación")
seccion = st.sidebar.radio(
    "Sección",
    ["Exploración de datos", "Predicción", "Explicabilidad", "Comparativa de modelos"],
    label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.subheader("Configuración del modelo")
modelo_sel = st.sidebar.selectbox(
    "Modelo de predicción",
    ["Random Forest", "XGBoost", "MLP — Red Neuronal"])
st.sidebar.markdown("---")
st.sidebar.caption("TFM — Predicción temprana del abandono escolar")


# ==================================================================
# SECCIÓN 1: EXPLORACIÓN DE DATOS
# ==================================================================
if seccion == "Exploración de datos":
    st.title("Exploración de datos")
    st.write(f"Conjunto de datos de Realinho et al. (2022). "
             f"Registros: {D['X'].shape[0]} | Variables: {D['X'].shape[1]}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribución de clases")
        conteo = D["y"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 2.8))
        ax.bar([trad_clase(k) for k in conteo.index], conteo.values,
               color=[COL.get(k, "#888") for k in conteo.index])
        ax.set_ylabel("Estudiantes")
        for i, v in enumerate(conteo.values):
            ax.text(i, v + 20, str(v), ha="center", fontsize=7)
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.subheader("Edad en la matrícula")
        fig, ax = plt.subplots(figsize=(4, 2.8))
        ax.hist(D["X"]["Age at enrollment"], bins=25,
                color="#2980b9", edgecolor="white")
        ax.set_xlabel("Edad")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig, use_container_width=True)

    st.subheader("Estadísticas de variables académicas")
    tabla = D["X"][["Curricular units 1st sem (approved)",
                    "Curricular units 2nd sem (approved)",
                    "Curricular units 1st sem (grade)",
                    "Curricular units 2nd sem (grade)",
                    "Age at enrollment"]].describe().round(2)
    tabla.columns = [trad(c) for c in tabla.columns]
    st.dataframe(tabla, use_container_width=True)


# ==================================================================
# SECCIÓN 2: PREDICCIÓN
# ==================================================================
elif seccion == "Predicción":
    st.title("Módulo de Predicción")
    st.write("Introduce los datos del estudiante para obtener la predicción "
             "de riesgo de abandono.")

    modo = st.radio("Modo de predicción",
                    ["Individual", "Carga de CSV (lote)"], horizontal=True)

    if modo == "Individual":
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Datos del estudiante")
            edad = st.number_input("Edad en la matrícula", 17, 70, 25)
            genero = st.selectbox("Género", ["Masculino", "Femenino"])
            beca = st.selectbox("Becario", ["No", "Sí"])
            matricula = st.selectbox("Matrícula al día", ["No", "Sí"])
            u2 = st.number_input("Unidades aprobadas (2º semestre)", 0, 26, 1)
            n2 = st.number_input("Calificación media (2º semestre)", 0.0, 20.0, 8.0)
            u1 = st.number_input("Unidades aprobadas (1er semestre)", 0, 26, 2)
            n1 = st.number_input("Calificación media (1er semestre)", 0.0, 20.0, 9.0)
            predecir = st.button("Obtener predicción", type="primary")

        if predecir:
            x_esc = vector_estudiante({
                "Age at enrollment": edad,
                "Gender": 1 if genero == "Masculino" else 0,
                "Scholarship holder": 1 if beca == "Sí" else 0,
                "Tuition fees up to date": 1 if matricula == "Sí" else 0,
                "Curricular units 2nd sem (approved)": u2,
                "Curricular units 2nd sem (grade)": n2,
                "Curricular units 1st sem (approved)": u1,
                "Curricular units 1st sem (grade)": n1,
            })
            modelo, nombre = modelo_obj(modelo_sel)
            proba = modelo.predict_proba(x_esc)[0]
            clase = le.classes_[int(np.argmax(proba))]

            with c2:
                st.subheader("Resultado")
                if clase == "Dropout":
                    st.markdown(
                        "<div style='background:#FADBD8;border:2px solid #c0392b;"
                        "border-radius:8px;padding:14px;margin-bottom:8px'>"
                        "<h4 style='color:#c0392b;margin:0'>"
                        "&#9888;&#65039; RIESGO DE ABANDONO DETECTADO</h4></div>",
                        unsafe_allow_html=True)
                elif clase == "Enrolled":
                    st.markdown(
                        "<div style='background:#FCF3CF;border:2px solid #f39c12;"
                        "border-radius:8px;padding:14px;margin-bottom:8px'>"
                        "<h4 style='color:#b9770e;margin:0'>"
                        "SEGUIMIENTO RECOMENDADO</h4></div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<div style='background:#D5F5E3;border:2px solid #27ae60;"
                        "border-radius:8px;padding:14px;margin-bottom:8px'>"
                        "<h4 style='color:#1e8449;margin:0'>"
                        "SIN RIESGO DETECTADO</h4></div>",
                        unsafe_allow_html=True)
                st.markdown(f"**Clase predicha: {trad_clase(clase)}**")
                for cl, p in zip(le.classes_, proba):
                    st.write(f"P({trad_clase(cl)}): {p:.2f}")
                    st.progress(float(p))
                st.caption(f"Modelo: {modelo_sel}")

            st.markdown("---")
            st.subheader("Explicación SHAP de la predicción")
            try:
                serie = shap_local(modelo, nombre, x_esc)
                fig2, ax2 = plt.subplots(figsize=(7, 2.6))
                colores_b = ["#c0392b" if v > 0 else "#2980b9" for v in serie.values]
                ax2.barh(range(len(serie)), serie.values, color=colores_b)
                ax2.set_yticks(range(len(serie)))
                ax2.set_yticklabels(serie.index, fontsize=7)
                ax2.invert_yaxis()
                ax2.axvline(0, color="gray", linewidth=0.8)
                ax2.set_xlabel("Valor SHAP (impacto en Abandono)")
                st.pyplot(fig2, use_container_width=True)
                st.caption("Rojo: variables que aumentan el riesgo de abandono. "
                           "Azul: variables que lo reducen.")
            except Exception:
                st.info("La explicación SHAP no pudo generarse para este modelo.")

            st.markdown("---")
            m = D["met"][nombre]
            cm = st.columns(4)
            cm[0].metric("F1-score macro", m["F1-score macro"])
            cm[1].metric("Recall Abandono", m["Recall Abandono"])
            cm[2].metric("Precisión Abandono", m["Precisión Abandono"])
            cm[3].metric("AUC-ROC", m["AUC-ROC"])

    else:
        st.subheader("Predicción en lote")
        st.write("Sube un CSV con las mismas columnas del dataset original "
                 "para predecir varios estudiantes a la vez.")
        archivo = st.file_uploader("Archivo CSV", type=["csv"])
        if archivo is not None:
            try:
                df = pd.read_csv(archivo)
                df_cols = df.reindex(columns=cols, fill_value=0)
                x_esc = scaler.transform(df_cols.values)
                modelo, nombre = modelo_obj(modelo_sel)
                pred = modelo.predict(x_esc)
                df_res = df.copy()
                df_res["Predicción"] = [trad_clase(le.classes_[p]) for p in pred]
                st.dataframe(df_res, use_container_width=True)
                st.download_button("Descargar resultados",
                                   df_res.to_csv(index=False).encode("utf-8"),
                                   "predicciones.csv", "text/csv")
            except Exception:
                st.error("No se pudo procesar el archivo. Revisa el formato.")


# ==================================================================
# SECCIÓN 3: EXPLICABILIDAD
# ==================================================================
elif seccion == "Explicabilidad":
    st.title("Explicabilidad global (SHAP)")
    st.write("Importancia global de las variables para el modelo XGBoost "
             "sobre el conjunto de prueba.")
    try:
        expl = shap.TreeExplainer(D["xgb"])
        sv = expl.shap_values(D["X_te"])
        sv_dp = sv[:, :, D["idx_dp"]] if np.array(sv).ndim == 3 else sv[D["idx_dp"]]
        imp = pd.Series(np.abs(sv_dp).mean(axis=0),
                        index=[trad(c) for c in cols]).sort_values(ascending=False).head(10)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.barh(range(len(imp)), imp.values, color="#2980b9")
            ax.set_yticks(range(len(imp)))
            ax.set_yticklabels(imp.index, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Importancia media (|SHAP|)")
            st.pyplot(fig, use_container_width=True)
        with c2:
            st.write("Variables más influyentes:")
            st.dataframe(imp.round(3).rename("Importancia"),
                         use_container_width=True)
    except Exception:
        st.info("No se pudo generar el gráfico SHAP global.")


# ==================================================================
# SECCIÓN 4: COMPARATIVA DE MODELOS
# ==================================================================
elif seccion == "Comparativa de modelos":
    st.title("Comparativa de modelos")
    st.write("Métricas de evaluación sobre el conjunto de prueba (885 registros).")
    st.dataframe(pd.DataFrame(D["met"]).T, use_container_width=True)

    st.subheader("Matrices de confusión")
    c = st.columns(3)
    for col, (nombre, modelo) in zip(
            c, [("Random Forest", D["rf"]), ("XGBoost", D["xgb"]), ("MLP", D["mlp"])]):
        cm = confusion_matrix(D["y_te"], modelo.predict(D["X_te"]))
        fig, ax = plt.subplots(figsize=(2.8, 2.6))
        ax.imshow(cm, cmap="Blues")
        etiquetas = [trad_clase(c_) for c_ in le.classes_]
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(etiquetas, fontsize=6, rotation=45, ha="right")
        ax.set_yticklabels(etiquetas, fontsize=6)
        ax.set_title(nombre, fontsize=8)
        ax.set_xlabel("Predicción", fontsize=7)
        ax.set_ylabel("Real", fontsize=7)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
        col.pyplot(fig, use_container_width=True)
