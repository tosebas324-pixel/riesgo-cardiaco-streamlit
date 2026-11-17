"""Prototipo académico: predicción del nivel de riesgo cardíaco (Bajo / Medio / Alto)."""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent / "artefactos"
ORDEN = ["Bajo", "Medio", "Alto"]
COLORES = {"Bajo": "#4C9F70", "Medio": "#E0B040", "Alto": "#C8553D"}
INTEGRANTES = ["Sebastián Andrés Muñoz Lemus"]   # <-- agrega a todos los integrantes
FUENTE = "Heart Failure Prediction Dataset (fedesoriano, 2021), Kaggle, 918 registros."

SEXO = {"Masculino": "male", "Femenino": "female"}
DOLOR = {
    "Asintomático": "asymptomatic",
    "Angina atípica": "atypical_angina",
    "Dolor no anginoso": "non_anginal",
    "Angina típica": "typical_angina",
}
ECG = {"Normal": "normal", "Anomalía ST-T": "st_t_abnormality", "Hipertrofia ventricular izq. (LVH)": "lvh"}
GLUCOSA = {"No (≤ 120 mg/dl)": 0, "Sí (> 120 mg/dl)": 1}

RANGOS = {"age": (18, 100), "resting_bp": (80, 220), "cholesterol": (100, 603), "max_hr": (60, 220)}

st.set_page_config(page_title="Riesgo cardíaco – prototipo académico", page_icon="🫀", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "modelo_riesgo.joblib")


@st.cache_data
def cargar_metricas():
    return json.loads((BASE / "metricas.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_eda():
    return pd.read_csv(BASE / "datos_eda.csv")


@st.cache_data
def cargar_muestra():
    return pd.read_csv(BASE / "muestra_prueba.csv")


def validar(d):
    """Devuelve la lista de errores de tipo, rango o categoría."""
    errores = []
    for campo, (lo, hi) in RANGOS.items():
        v = d[campo]
        if campo == "cholesterol" and (v is None or (isinstance(v, float) and np.isnan(v))):
            continue  # colesterol no disponible: se imputa con la mediana del entrenamiento
        if not isinstance(v, (int, float, np.integer, np.floating)) or np.isnan(v):
            errores.append(f"{campo}: debe ser un número.")
        elif not lo <= v <= hi:
            errores.append(f"{campo}: {v} fuera del rango permitido ({lo}–{hi}).")
    if d["sex"] not in SEXO.values():
        errores.append("Sexo: categoría no válida.")
    if d["chest_pain_type"] not in DOLOR.values():
        errores.append("Tipo de dolor torácico: categoría no válida.")
    if d["resting_ecg"] not in ECG.values():
        errores.append("ECG en reposo: categoría no válida.")
    if d["fasting_bs"] not in (0, 1):
        errores.append("Glucosa en ayunas: valor no válido.")
    return errores


def fig_matriz(m):
    cm = np.array(m["matriz"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), m["orden"]); ax.set_yticks(range(3), m["orden"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


st.title("🫀 Predicción del nivel de riesgo cardíaco")
st.warning("**Prototipo académico.** No es un dispositivo médico y no sustituye la decisión de un profesional de la salud. "
           "El nivel de riesgo es una etiqueta construida por el grupo, no un riesgo clínico validado.")

modelo = cargar_modelo()
metricas = cargar_metricas()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Proyecto", "Exploración de datos", "Predicción", "Modelo y métricas", "Verificación"])

with tab1:
    st.subheader("Proyecto")
    st.write("**Objetivo:** clasificar a un paciente en riesgo **Bajo, Medio o Alto** a partir de ocho variables "
             "demográficas y clínicas básicas, con énfasis en no confundir un paciente de riesgo Alto con uno Bajo.")
    st.write("**Curso:** Minería de Datos – Universidad de la Costa (Barranquilla, Colombia). Docente: José Escorcia Gutiérrez.")
    st.write("**Integrantes:** " + ", ".join(INTEGRANTES))
    st.write("**Fuente de los datos:** " + FUENTE)
    st.write("**Variable objetivo:** `nivel_riesgo`, construida combinando el diagnóstico con tres alertas de la prueba "
             "de esfuerzo (angina inducida, pendiente ST no ascendente y Oldpeak ≥ 1). Esas variables se excluyen de los "
             "predictores para evitar fuga de información.")

with tab2:
    st.subheader("Exploración de datos")
    eda = cargar_eda()
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        conteo = eda["nivel_riesgo"].value_counts().reindex(ORDEN)
        ax.bar(ORDEN, conteo.values, color=[COLORES[c] for c in ORDEN])
        for i, v in enumerate(conteo.values):
            ax.text(i, v, int(v), ha="center", va="bottom")
        ax.set_ylabel("Número de pacientes"); ax.set_title("Distribución del nivel de riesgo")
        fig.tight_layout(); st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        bp = ax.boxplot([eda.loc[eda["nivel_riesgo"] == c, "max_hr"] for c in ORDEN], patch_artist=True, tick_labels=ORDEN)
        for patch, c in zip(bp["boxes"], ORDEN):
            patch.set_facecolor(COLORES[c])
        ax.set_ylabel("Frecuencia cardíaca máxima (lpm)"); ax.set_title("Frecuencia máxima según riesgo")
        fig.tight_layout(); st.pyplot(fig)
    etiquetas_dolor = {v: k for k, v in DOLOR.items()}
    tabla = pd.crosstab(eda["chest_pain_type"], eda["nivel_riesgo"], normalize="index")[ORDEN] * 100
    tabla.index = [etiquetas_dolor.get(i, i) for i in tabla.index]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    tabla.plot(kind="bar", stacked=True, ax=ax, color=[COLORES[c] for c in ORDEN])
    ax.set_ylabel("% de pacientes"); ax.set_xlabel(""); ax.set_title("Nivel de riesgo según tipo de dolor torácico")
    plt.xticks(rotation=0); fig.tight_layout(); st.pyplot(fig)

with tab3:
    st.subheader("Ingreso de datos del paciente")
    with st.form("formulario"):
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad (años)", *RANGOS["age"], value=55, step=1)
            sexo = st.selectbox("Sexo", list(SEXO))
            dolor = st.selectbox("Tipo de dolor torácico", list(DOLOR))
            pa = st.number_input("Presión arterial en reposo (mm Hg)", *RANGOS["resting_bp"], value=130, step=1)
        with c2:
            sin_col = st.checkbox("No dispongo del colesterol (se imputará con la mediana del entrenamiento)")
            col = st.number_input("Colesterol sérico (mg/dl)", *RANGOS["cholesterol"], value=220, step=1, disabled=sin_col)
            gluc = st.selectbox("Glucosa en ayunas", list(GLUCOSA))
            ecg = st.selectbox("ECG en reposo", list(ECG))
            fc = st.number_input("Frecuencia cardíaca máxima (lpm)", *RANGOS["max_hr"], value=140, step=1)
        enviar = st.form_submit_button("Predecir")

    if enviar:
        datos = {"age": edad, "sex": SEXO[sexo], "chest_pain_type": DOLOR[dolor], "resting_bp": pa,
                 "cholesterol": np.nan if sin_col else col, "fasting_bs": GLUCOSA[gluc],
                 "resting_ecg": ECG[ecg], "max_hr": fc}
        errores = validar(datos)
        if errores:
            st.error("Revisa los datos ingresados:")
            for e in errores:
                st.write("• " + e)
        else:
            if fc > 220 - edad + 20:
                st.warning("La frecuencia máxima ingresada es inusualmente alta para la edad; verifica el dato.")
            X = pd.DataFrame([datos])
            pred = modelo.predict(X)[0]
            st.markdown(f"### Nivel de riesgo estimado: **{pred}**")
            if hasattr(modelo, "predict_proba"):
                prob = pd.Series(modelo.predict_proba(X)[0], index=modelo.classes_).reindex(ORDEN)
                st.write("Probabilidad estimada por clase:")
                st.bar_chart(prob * 100, y_label="%")
                st.table((prob * 100).round(1).astype(str).add(" %").rename("Probabilidad"))
            st.caption("Resultado de un modelo exploratorio con métricas moderadas; la clase Medio se predice con poca fiabilidad.")

with tab4:
    st.subheader(f"Modelo final: {metricas['modelo']}")
    st.write(f"Evaluado en {metricas['n_test']} registros de prueba (entrenado con {metricas['n_train']}).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metricas['accuracy']:.3f}")
    c2.metric("F1 macro", f"{metricas['f1_macro']:.3f}")
    c3.metric("Recall macro", f"{metricas['recall_macro']:.3f}")
    c4.metric("Recall Alto", f"{metricas['por_clase']['Alto']['recall']:.3f}")
    st.caption(f"IC 95 % del accuracy (Wilson): [{metricas['ic95'][0]:.3f}; {metricas['ic95'][1]:.3f}] · "
               f"Validación cruzada (5 pliegues): accuracy {metricas['cv']['accuracy_mean']:.3f} ± {metricas['cv']['accuracy_std']:.3f}, "
               f"F1 macro {metricas['cv']['f1_mean']:.3f} ± {metricas['cv']['f1_std']:.3f}")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(fig_matriz(metricas))
    with c2:
        st.write("**Resultados por clase**")
        st.dataframe(pd.DataFrame(metricas["por_clase"]).T.rename(
            columns={"precision": "Precision", "recall": "Recall", "f1": "F1", "support": "Soporte"}).round(3))
    with st.expander("Comparación de modelos candidatos"):
        st.dataframe(pd.DataFrame(metricas["comparacion"]))
    st.markdown("**Limitaciones**")
    st.markdown(
        "- La etiqueta `nivel_riesgo` es una regla del grupo, no un riesgo clínico validado.\n"
        "- La clase **Medio** casi no se detecta (recall bajo).\n"
        "- Rendimiento moderado porque se excluyen los predictores que definen la etiqueta.\n"
        "- Colesterol faltante (valores 0) no aleatorio e imputado con la mediana.\n"
        "- Solo 193 mujeres frente a 725 hombres; conjunto de prueba pequeño; sin validación externa.")

with tab5:
    st.subheader("Verificación contra el notebook")
    st.write("Elige un registro del conjunto de prueba: la predicción de la app debe coincidir con la del notebook.")
    muestra = cargar_muestra()
    idx = st.selectbox("Registro de prueba (id)", muestra["id_registro"].tolist())
    fila = muestra[muestra["id_registro"] == idx].iloc[0]
    X = pd.DataFrame([fila.drop(["id_registro", "real", "pred_notebook"]).to_dict()])
    for c in ["age", "resting_bp", "cholesterol", "fasting_bs", "max_hr"]:
        X[c] = pd.to_numeric(X[c])
    pred_app = modelo.predict(X)[0]
    st.dataframe(X)
    c1, c2, c3 = st.columns(3)
    c1.metric("Real", fila["real"]); c2.metric("Notebook", fila["pred_notebook"]); c3.metric("App", pred_app)
    if pred_app == fila["pred_notebook"]:
        st.success("La predicción de la app coincide con la del notebook.")
    else:
        st.error("No coincide: revisa versiones de scikit-learn y el archivo .joblib.")

st.divider()
st.caption("Prototipo académico – Minería de Datos, Universidad de la Costa. " + "Integrantes: " + ", ".join(INTEGRANTES))
