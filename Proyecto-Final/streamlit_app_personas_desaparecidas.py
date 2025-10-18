import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Configuración de página ---
st.set_page_config(page_title='Predicción de Personas Desaparecidas', layout='wide')

st.title('🔍 Análisis y Predicción de Personas Desaparecidas en Ecuador (2017–2024)')
st.markdown("""
Esta aplicación permite **explorar las estadísticas históricas** de desapariciones en Ecuador
y **predecir la probabilidad de localización** de una persona desaparecida
según sus características.
""")

# --- Cargar modelo y transformadores ---
best_model = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# --- Crear pestañas ---
tab1, tab2 = st.tabs(["📊 Estadísticas Generales", "🤖 Predicción de Situación"])

# ============================================================
# 📊 TAB 1 — Estadísticas Generales
# ============================================================
with tab1:
    st.header("Resumen de Desapariciones (2017–2024)")

    st.markdown("**Rango temporal de desapariciones:** 2017-01-01 → 2024-12-31")

    st.subheader("Top 10 Provincias con más desapariciones")
    st.code("""
PICHINCHA                         16668
GUAYAS                            15815
MANABI                             3346
AZUAY                              3189
EL ORO                             2921
SANTO DOMINGO DE LOS TSACHILAS     2828
LOS RIOS                           2812
CHIMBORAZO                         2615
TUNGURAHUA                         2455
COTOPAXI                           1969
""")

    st.subheader("Años con mayor número de desapariciones")
    st.code("""
2017    10457
2018    10255
2019     9962
2020     6762
2021     7955
2022     7721
2023     7808
2024     7009
""")

    st.subheader("Motivos de desaparición más frecuentes")
    st.code("""
CAUSAS FAMILIARES                                        47313
CAUSAS SOCIALES                                           5277
EXTRAVIADO - DISCAPACIDAD / ENFERMEDADES / TRASTORNOS     3981
CAUSAS PERSONALES                                         3444
FALLECIDO                                                 2186
EXTRAVIADO - AUSENCIA TEMPORAL                            2024
DESCONOCIDO                                               1904
CAUSAS ACADÉMICAS                                         1246
CERRADO POR FISCALÍA / DELITO REFORMULADO                  245
VIOLENCIA                                                  147
""")

    st.subheader("Edades más comunes de personas desaparecidas")
    st.code("""
15    8403
16    7581
14    6789
17    5593
13    4255
18    2673
19    2007
12    1771
20    1564
21    1392
""")

    st.subheader("Distribución por rango de edad")
    st.code("""
ADOLESCENTES     34391
ADULTOS          26694
NIÑOS(AS)         4223
ADULTO MAYOR      2621
""")

    st.subheader("Distribución por nacionalidad")
    st.code("""
ECUADOR                 65615
VENEZUELA                1149
COLOMBIA                  638
DESCONOCIDO               264
PERU                       76
... (otros países con menor frecuencia)
""")

    st.subheader("Distribución por etnia")
    st.code("""
MESTIZO/A      58893
INDIGENA        3484
AFRO            1873
BLANCO/A        1140
OTROS            871
MONTUBIO/A       783
MULATO/A         683
DESCONOCIDO      170
ASIATICO/A        32
""")

    st.subheader("Distribución por sexo")
    st.code("""
MUJER     42981
HOMBRE    24948
""")


# ============================================================
# 🤖 TAB 2 — Predicción de Situación
# ============================================================
with tab2:
    st.header("Predicción de Situación Actual")

    # --- Entradas del usuario ---
    sexo = st.selectbox('Sexo', ['MUJER', 'HOMBRE'])
    provincia = st.selectbox('Provincia', [
        'AZUAY', 'BOLIVAR', 'CAÑAR', 'CARCHI', 'COTOPAXI', 'CHIMBORAZO',
        'EL ORO', 'ESMERALDAS', 'GALAPAGOS', 'GUAYAS', 'IMBABURA', 'LOJA',
        'LOS RIOS', 'MANABI', 'MORONA SANTIAGO', 'NAPO', 'ORELLANA',
        'PASTAZA', 'PICHINCHA', 'SANTA ELENA', 'SANTO DOMINGO DE LOS TSACHILAS',
        'SUCUMBIOS', 'TUNGURAHUA', 'ZAMORA CHINCHIPE'
    ])
    nacionalidad = st.selectbox('Nacionalidad', [
        'ECUADOR', 'VENEZUELA', 'COLOMBIA', 'DESCONOCIDO', 'PERU', 'ESPAÑA', 'ESTADOS UNIDOS',
        'ARGENTINA', 'CHINA POPULAR', 'CUBA', 'FRANCIA', 'MEXICO', 'CHILE', 'RUSIA', 'ALEMANIA',
        'BRASIL', 'REPUBLICA DOMINICANA', 'ITALIA', 'BOLIVIA', 'HAITI', 'CANADA', 'EL SALVADOR',
        'HONDURAS', 'AFGANISTAN', 'GUATEMALA', 'INDIA', 'ALBANIA', 'DOMINICA', 'BAHREIN', 'IRLANDA',
        'COSTA RICA', 'EGIPTO', 'TERRITORIO BRITANICO', 'ISRAEL', 'REPUBLICA CENTROAFRI', 'INDONESIA'
    ])

    edad_aproximada = st.number_input('Edad', min_value=0, max_value=120, value=30)
    etnia = st.selectbox('Etnia', [
        'MESTIZO/A', 'INDIGENA', 'AFRO', 'BLANCO/A', 'OTROS', 'MONTUBIO/A', 
        'MULATO/A', 'DESCONOCIDO', 'ASIATICO/A'
    ])

    # --- Codificación ---
    def encode_input(sexo, provincia, nacionalidad, etnia, edad):
        X_input = pd.DataFrame({
            'sexo': [label_encoders['sexo'].transform([sexo])[0]],
            'provincia': [label_encoders['provincia'].transform([provincia])[0]],
            'nacionalidad': [label_encoders['nacionalidad'].transform([nacionalidad])[0]],
            'etnia': [label_encoders['etnia'].transform([etnia])[0]],
            'edad_aproximada': [edad]
        })
        X_input['edad_aproximada'] = scaler.transform(X_input[['edad_aproximada']])
        return X_input

    # --- Predicción ---
    if st.button('Predecir Situación'):
        X_input = encode_input(sexo, provincia, nacionalidad, etnia, edad_aproximada)
        
        # --- Asegurar el mismo orden de columnas que en el modelo ---
        try:
            expected_features = best_model.get_booster().feature_names
            X_input = X_input[expected_features]
        except Exception as e:
            st.warning(f"No se pudo ajustar el orden de columnas automáticamente: {e}")

        prob = best_model.predict_proba(X_input)[:, 1][0]
        st.subheader('Resultado de la Predicción:')
        if prob >= 0.5:
            st.success(f'🟢 Probabilidad de ser localizado: {prob:.2f} → ENCONTRADO/FALLECIDO')
        else:
            st.error(f'🔴 Probabilidad de ser localizado: {prob:.2f} → DESAPARECIDO')

st.markdown('---')
st.caption('Desarrollado por Alexis Garzón — Maestría en Ciencia de Datos, Yachay Tech University')
