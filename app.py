import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.figure_factory as ff

# Configuración de página
st.set_page_config(
    page_title="Predicción Temprana de Diabetes - Minería de Datos",
    page_icon="🩺",
    layout="wide"
)

# Cargar modelo serializado y datos limpios para visualización
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_diabetes.joblib")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("diabetes.csv")
    cols_biologicas = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = df.copy()
    for col in cols_biologicas:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean

modelo = cargar_modelo()
df = cargar_datos()

# Barra lateral con información institucional
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/120px-Python-logo-notext.svg.png", width=60)
    st.title("Proyecto de Aula")
    st.markdown("**Asignatura:** Minería de Datos")
    st.markdown("**Institución:** Universidad de la Costa (CUC)")
    st.markdown("**Programa:** Ingeniería de Sistemas")
    st.markdown("**Autor:** Jorge Mario Camargo Pérez")
    st.divider()
    st.info(
        "⚠️ **Aviso Clínico:** Esta aplicación es un prototipo académico desarrollado con fines educativos "
        "y no sustituye el criterio ni el diagnóstico de un profesional de la salud."
    )

# Encabezado principal
st.title("🩺 Modelo Predictivo para la Detección Temprana de Diabetes Mellitus")
st.markdown(
    "**Objetivo:** Estimar el riesgo de diabetes mellitus tipo 2 en pacientes adultas mediante aprendizaje automático "
    "supervisado (*Regresión Logística Balanceada*), utilizando biomarcadores clínicos y antropométricos rutinarios.\n"
    "**Fuente de datos:** *Pima Indians Diabetes Database* (NIDDK / Kaggle — 768 registros clínicos)."
)

# Pestañas de navegación
tab1, tab2, tab3 = st.tabs(["🔮 Predicción Interactiva", "📊 Exploración de Datos (EDA)", "📈 Rendimiento y Limitaciones"])

# ==========================================
# TAB 1: PREDICCIÓN INTERACTIVA
# ==========================================
with tab1:
    st.subheader("Ingreso de Parámetros Clínicos del Paciente")
    st.markdown("Complete los siguientes campos respetando los rangos fisiológicos válidos. Puede cargar un caso de prueba verificado para validar la consistencia con el notebook.")

    # Botón para precargar caso de prueba conocido del test set
    if st.button("📋 Cargar Registro Verificado del Conjunto de Prueba (Índice 0)"):
        st.session_state["preg"] = 3
        st.session_state["gluc"] = 125.5
        st.session_state["bp"] = 69.5
        st.session_state["skin"] = 40.7
        st.session_state["ins"] = 113.85
        st.session_state["bmi"] = 43.3
        st.session_state["dpf"] = 0.942
        st.session_state["age"] = 28

    with st.form("formulario_prediccion"):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input(
                "Número de Embarazos (Pregnancies)", 
                min_value=0, max_value=20, value=st.session_state.get("preg", 2), step=1,
                help="Cantidad de gestaciones previas (Rango válido: 0 a 20)."
            )
            glucose = st.number_input(
                "Glucosa Plasmática a 2h (mg/dL)", 
                min_value=40.0, max_value=300.0, value=st.session_state.get("gluc", 115.0), step=0.5,
                help="Concentración de glucosa en prueba de tolerancia oral (Rango fisiológico: 40 - 300 mg/dL)."
            )
            blood_pressure = st.number_input(
                "Presión Arterial Diastólica (mm Hg)", 
                min_value=20.0, max_value=150.0, value=st.session_state.get("bp", 72.0), step=0.5,
                help="Presión diastólica en reposo (Rango fisiológico: 20 - 150 mm Hg)."
            )
            skin_thickness = st.number_input(
                "Grosor Pliegue Cutáneo Tríceps (mm)", 
                min_value=5.0, max_value=100.0, value=st.session_state.get("skin", 29.0), step=0.5,
                help="Medición de adiposidad subcutánea (Rango: 5 - 100 mm)."
            )

        with col2:
            insulin = st.number_input(
                "Insulina Sérica a 2h (μU/mL)", 
                min_value=10.0, max_value=900.0, value=st.session_state.get("ins", 115.0), step=1.0,
                help="Nivel de insulina sérica post-carga (Rango: 10 - 900 μU/mL)."
            )
            bmi = st.number_input(
                "Índice de Masa Corporal - IMC (kg/m²)", 
                min_value=12.0, max_value=70.0, value=st.session_state.get("bmi", 31.5), step=0.1,
                help="Peso en kg dividido por altura al cuadrado (Rango: 12 - 70 kg/m²)."
            )
            dpf = st.number_input(
                "Función de Pedigrí de Diabetes (DPF)", 
                min_value=0.050, max_value=2.500, value=st.session_state.get("dpf", 0.450), step=0.005, format="%.3f",
                help="Puntaje adimensional de carga genética familiar (Rango: 0.05 - 2.50)."
            )
            age = st.number_input(
                "Edad Cronológica (Años)", 
                min_value=18, max_value=100, value=st.session_state.get("age", 33), step=1,
                help="Edad de la paciente en años cumplidos (Rango: 18 - 100 años)."
            )

        submitted = st.form_submit_button("🔍 Ejecutar Predicción", use_container_width=True)

    if submitted:
        # Validaciones adicionales de seguridad clínica
        errores = []
        if glucose <= 0 or blood_pressure <= 0 or bmi <= 0:
            errores.append("Los parámetros fisiológicos vitales (Glucosa, Presión Arterial, IMC) deben ser mayores a cero.")
        if age < 21:
            st.warning("⚠️ Nota: El modelo fue entrenado con pacientes adultas de al menos 21 años. Interprete el resultado con precaución.")

        if errores:
            for err in errores:
                st.error(err)
        else:
            entrada = pd.DataFrame([{
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }])

            prediccion = int(modelo.predict(entrada)[0])
            probabilidades = modelo.predict_proba(entrada)[0]
            prob_negativa = probabilidades[0] * 100
            prob_positiva = probabilidades[1] * 100

            st.divider()
            st.subheader("Resultado de la Evaluación")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if prediccion == 1:
                    st.error(f"### 🚨 Resultado: POSITIVO (Riesgo Alto de Diabetes)")
                    st.markdown(f"El modelo estima una probabilidad de **diabetes mellitus del {prob_positiva:.2f}%** frente a un {prob_negativa:.2f}% de ausencia.")
                else:
                    st.success(f"### ✅ Resultado: NEGATIVO (Bajo Riesgo de Diabetes)")
                    st.markdown(f"El modelo estima una probabilidad de **ausencia de diabetes del {prob_negativa:.2f}%** frente a un {prob_positiva:.2f}% de riesgo positivo.")

            with res_col2:
                st.metric(label="Probabilidad Estimada (Clase Positiva)", value=f"{prob_positiva:.2f}%")
                st.progress(int(prob_positiva))

# ==========================================
# TAB 2: EXPLORACIÓN DE DATOS (EDA)
# ==========================================
with tab2:
    st.subheader("Visualizaciones Clave del Análisis Exploratorio (EDA)")
    st.markdown("El conjunto de datos cuenta con **768 registros clínicos** (65.1% Negativos vs. 34.9% Positivos) tras imputación estadística por mediana de ceros biológicamente inválidos.")

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        df_plot = df.copy()
        df_plot["Diagnóstico"] = df_plot["Outcome"].map({0: "No Diabético (0)", 1: "Diabético (1)"})
        fig1 = px.box(
            df_plot, x="Diagnóstico", y="Glucose", color="Diagnóstico",
            color_discrete_map={"No Diabético (0)": "#4C78A8", "Diabético (1)": "#E15759"},
            title="1. Distribución de Glucosa Plasmática según Diagnóstico",
            labels={"Glucose": "Glucosa (mg/dL)"}
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Interpretación: Las pacientes con diabetes exhiben niveles medianos de glucosa notoriamente superiores (~142 mg/dL vs ~110 mg/dL).")

    with col_g2:
        fig2 = px.scatter(
            df_plot, x="BMI", y="Glucose", color="Diagnóstico",
            color_discrete_map={"No Diabético (0)": "#4C78A8", "Diabético (1)": "#E15759"},
            opacity=0.75,
            title="2. Relación Multivariada: Glucosa vs. IMC",
            labels={"BMI": "Índice de Masa Corporal (kg/m²)", "Glucose": "Glucosa (mg/dL)"}
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Interpretación: La combinación de hiperglucemia (>130 mg/dL) e IMC elevado (>30 kg/m²) concentra la mayor densidad de casos positivos.")

    st.divider()
    corr = df.corr().round(2)
    fig3 = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="3. Matriz de Correlación de Pearson entre Biomarcadores"
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Interpretación: La glucosa presenta la correlación lineal más alta con el diagnóstico (Outcome), seguida del IMC y la edad.")

# ==========================================
# TAB 3: RENDIMIENTO Y LIMITACIONES
# ==========================================
with tab3:
    st.subheader("Métricas Finales y Evaluación Técnica del Modelo")
    st.markdown(
        "Se seleccionó la **Regresión Logística Balanceada** (`class_weight='balanced'`) con estandarización `StandardScaler` "
        "sobre el Árbol de Decisión por su mayor sensibilidad (Recall) y estabilidad en validación cruzada estratificada (5 folds)."
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy (Prueba)", "78.6%")
    m2.metric("Precision", "67.1%")
    m3.metric("Recall (Sensibilidad)", "76.1%")
    m4.metric("F1-Score (Prueba)", "71.3%")
    m5.metric("F1-Score CV (5-Fold)", "76.8% ± 2.2%")

    st.divider()
    c_mat, c_lim = st.columns([1, 1.2])

    with c_mat:
        st.markdown("#### Matriz de Confusión (Conjunto de Prueba, N=192)")
        z = [[100, 25], [16, 51]]
        x_labels = ["Predicho: Negativo (0)", "Predicho: Positivo (1)"]
        y_labels = ["Real: Negativo (0)", "Real: Positivo (1)"]
        fig_cm = ff.create_annotated_heatmap(
            z, x=x_labels, y=y_labels, colorscale="Blues", showscale=False
        )
        fig_cm.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Verdaderos Negativos: 100 | Falsos Positivos: 25 | Falsos Negativos: 16 | Verdaderos Positivos: 51")

    with c_lim:
        st.markdown("#### Limitaciones del Modelo y Consideraciones")
        st.warning(
            "**1. Naturaleza de la muestra:** El dataset *Pima Indians* se restringe a mujeres adultas (≥21 años) de ascendencia Pima; "
            "sus resultados no deben generalizarse directamente a población masculina o pediátrica sin recalibración."
        )
        st.info(
            "**2. Imputación de variables de laboratorio:** Atributos como `Insulin` y `SkinThickness` presentaban alta proporción de ceros encubiertos "
            "que fueron imputados mediante la mediana poblacional, lo cual atenúa su varianza natural."
        )
        st.error(
            "**3. Balance Sensibilidad vs. Especificidad:** Al priorizar la detección oportuna (Recall del 76.1% para minimizar falsos negativos), "
            "el modelo asume una tasa moderada de falsos positivos (25 casos en prueba), operando como herramienta de tamizaje preliminar."
        )
