"""
streamlit_app.py
Interfaz web para la API de predicción de abandono estudiantil.
La app no carga el modelo: llama a la API desplegada en Render, que es la
que predice. La API key se lee de los secrets de Streamlit (nunca del código).

Correr local:  streamlit run app/streamlit_app.py
"""

import streamlit as st

from cliente import CARRERAS, ErrorAPI, construir_features, predecir

API_URL_POR_DEFECTO = "https://prediccion-abandono-api.onrender.com"
REPO_URL = "https://github.com/NiurkaVane11/prediccion-del-abandono-estudiantil"

st.set_page_config(page_title="Riesgo de abandono estudiantil", page_icon="🎓", layout="centered")

st.title("🎓 Riesgo de abandono estudiantil")
st.write(
    "Estima la probabilidad de que un estudiante universitario **abandone** sus estudios, "
    "con la información disponible al terminar el **1er semestre**. "
    "Pensado como alerta temprana para ofrecer apoyo a tiempo."
)

api_url = st.secrets.get("API_URL", API_URL_POR_DEFECTO)
api_key = st.secrets.get("API_KEY")
if not api_key:
    st.error("Falta configurar `API_KEY` en los secrets de la app.")
    st.stop()

with st.form("estudiante"):
    st.subheader("Datos del estudiante")
    col1, col2 = st.columns(2)
    with col1:
        carrera = st.selectbox(
            "Carrera",
            options=list(CARRERAS),
            format_func=CARRERAS.get,
            index=list(CARRERAS).index(12),
        )
        edad = st.number_input("Edad al matricularse", min_value=17, max_value=70, value=18)
        genero = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
    with col2:
        becado = st.checkbox("Tiene beca")
        deudor = st.checkbox("Tiene deudas con la universidad")
        matricula_al_dia = st.checkbox("Matrícula al día", value=True)
        desplazado = st.checkbox("Vive fuera de su ciudad de origen", value=True)

    st.subheader("Resultados del 1er semestre")
    col3, col4 = st.columns(2)
    with col3:
        inscritas = st.number_input("Materias inscritas", min_value=0, max_value=26, value=6)
        aprobadas = st.number_input("Materias aprobadas", min_value=0, max_value=26, value=6)
        nota = st.slider("Nota promedio (escala 0–20)", min_value=0.0, max_value=20.0, value=13.0, step=0.5)
    with col4:
        evaluadas = st.number_input("Evaluaciones rendidas", min_value=0, max_value=45, value=8)
        convalidadas = st.number_input("Materias convalidadas", min_value=0, max_value=20, value=0)
        sin_evaluacion = st.number_input("Materias sin evaluación", min_value=0, max_value=12, value=0)

    enviado = st.form_submit_button("Calcular riesgo", type="primary", width="stretch")

if enviado:
    try:
        features = construir_features(
            carrera=carrera,
            edad=edad,
            genero_masculino=genero == "Masculino",
            desplazado=desplazado,
            becado=becado,
            deudor=deudor,
            matricula_al_dia=matricula_al_dia,
            materias_inscritas=inscritas,
            materias_evaluadas=evaluadas,
            materias_aprobadas=aprobadas,
            materias_convalidadas=convalidadas,
            materias_sin_evaluacion=sin_evaluacion,
            nota_promedio=nota,
        )
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    with st.spinner("Consultando el modelo... (si la API estaba dormida puede tardar ~50 segundos)"):
        try:
            resultado = predecir(features, api_url, api_key)
        except ErrorAPI as e:
            st.error(str(e))
            st.stop()

    probabilidad = resultado["probabilidad_dropout"]
    riesgo = resultado["riesgo"]
    mensaje = f"Probabilidad de abandono: **{probabilidad:.0%}** · riesgo **{riesgo}**"

    st.subheader("Resultado")
    st.metric("Probabilidad de abandono", f"{probabilidad:.0%}")
    st.progress(probabilidad)
    if riesgo == "alto":
        st.error(f"🔴 {mensaje}. Se recomienda contactar al estudiante y ofrecer apoyo.")
    elif riesgo == "medio":
        st.warning(f"🟡 {mensaje}. Conviene hacer seguimiento.")
    else:
        st.success(f"🟢 {mensaje}.")

st.divider()
st.caption(
    "Herramienta de apoyo, **no** para decisiones automáticas sobre estudiantes (becas, admisión, sanciones). "
    "Modelo: red neuronal (AUC-ROC 0.946) entrenada con datos de una universidad de Portugal; "
    "otros datos (estudios y ocupación de los padres, contexto económico) se completan con valores típicos. "
    f"[Código y model card]({REPO_URL}) · [API]({API_URL_POR_DEFECTO}/docs)"
)
