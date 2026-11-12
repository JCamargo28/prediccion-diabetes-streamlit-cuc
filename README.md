# Modelo Predictivo para la Detección Temprana de Diabetes Mellitus

Aplicación web interactiva desarrollada en **Streamlit** para la asignatura **Minería de Datos** de la **Universidad de la Costa (CUC)**.

## Integrante
- **Jorge Mario Camargo Pérez** — Programa de Ingeniería de Sistemas (6to Semestre)

## Descripción del Proyecto
El sistema implementa un pipeline reproducible de aprendizaje automático supervisado (*StandardScaler* + *LogisticRegression* con balanceo de clases) entrenado sobre la base de datos *Pima Indians Diabetes Database* (NIDDK / Kaggle) para estimar el riesgo de diabetes mellitus tipo 2 a partir de 8 variables clínicas y antropométricas.

> **Aviso:** Prototipo exclusivamente académico. No sustituye el diagnóstico ni la valoración médica profesional.

## Estructura del Repositorio
- `app.py`: Código fuente de la aplicación web interactiva en Streamlit.
- `modelo_diabetes.joblib`: Pipeline completo serializado (escalador y modelo clasificador).
- `diabetes.csv`: Conjunto de datos utilizado para las visualizaciones exploratorias.
- `Proyecto_MineriaDatos_Corte2_JorgeCamargo.ipynb`: Notebook actualizado con el flujo de entrenamiento, validación y serialización.
- `requirements.txt`: Dependencias de Python necesarias para el despliegue en Streamlit Community Cloud.

## Pruebas de Verificación Realizadas
1. **Caso normal:** Ingreso de valores dentro de rangos saludables (Glucosa: 105 mg/dL, IMC: 24.5 kg/m², Edad: 29 años) -> Predicción: `Negativo (0)`.
2. **Entrada inválida:** Protección mediante `st.number_input` con límites fisiológicos estrictos (ej. glucosa mínima de 40 mg/dL) y validación lógica contra valores nulos o cero.
3. **Registro conocido del conjunto de prueba (Índice 0 de `X_test`):**
   - Valores: `Pregnancies=3, Glucose=125.5, BloodPressure=69.5, SkinThickness=40.7, Insulin=113.85, BMI=43.3, DiabetesPedigreeFunction=0.942, Age=28`
   - Resultado en Notebook: Clase `1` (Positivo) con probabilidad `64.72%`.
   - Resultado en Streamlit: Clase `1` (Positivo) con probabilidad `64.72%` (coincidencia exacta).

## Evaluación del Uso de Inteligencia Artificial

| Elemento | Detalle Reportado |
|---|---|
| **Herramienta utilizada** | Gemini (Google DeepMind) |
| **Propósito** | Asistencia en la serialización del pipeline con `joblib`, estructuración de la interfaz multipestaña en `app.py`, configuración de gráficos interactivos con Plotly y redacción de documentación técnica. |
| **Prompt principal** | *"Genera el código completo de una aplicación en Streamlit (app.py), requirements.txt y README.md integrando un pipeline serializado con joblib para predicción de diabetes, incluyendo validación de rangos, visualizaciones del EDA, matriz de confusión y evaluación crítica de IA."* |
| **Verificación de la respuesta** | Se ejecutó localmente el pipeline y se contrastó la predicción y probabilidad exacta (`64.72%`) de un registro conocido del conjunto de prueba (`X_test.iloc[0]`) entre el notebook y la interfaz web. |
| **Problemas encontrados** | La propuesta inicial utilizaba gráficos estáticos de Matplotlib que no se adaptaban dinámicamente al ancho de pantalla y carecía de un botón para cargar casos de prueba verificados. |
| **Ajustes realizados** | Se migraron las visualizaciones a `plotly` con contenedores responsivos, se añadieron validaciones estrictas de límites fisiológicos en el formulario y se incorporó un botón de precarga de registro de prueba. |
