import streamlit as st
from ejercicio1_titanic import run_ejercicio1
from ejercicio2_student import run_ejercicio2
from ejercicio3_iris import run_ejercicio3

# Configuración de la página
st.set_page_config(
    page_title="Procesamiento de Datasets ML",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-weight: 600;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #ff7f0e;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/machine-learning.png", width=80)
    st.title("🎓 ML Dataset Preprocessing")
    st.markdown("---")
    
    st.markdown("""
    ### 📚 Actividad Individual
    
    **Etapas del procesamiento:**
    1. ✅ Carga del dataset
    2. 🔍 Exploración inicial
    3. 🧹 Limpieza de datos
    4. 🔢 Codificación de variables
    5. 📊 Normalización/Estandarización
    6. ✂️ División de datos
    
    ---
    """)
    
    ejercicio_seleccionado = st.radio(
        "**Selecciona un ejercicio:**",
        ["🚢 Ejercicio 1: Titanic", 
         "📚 Ejercicio 2: Student Performance", 
         "🌸 Ejercicio 3: Iris"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📌 Información
    
    **Autor:** Tu Nombre  
    **Curso:** Machine Learning  
    **Fecha:** 2024
    
    ---
    
    ### 🔗 Enlaces útiles
    - [Kaggle Datasets](https://www.kaggle.com/datasets)
    - [UCI ML Repository](https://archive.ics.uci.edu/ml)
    - [Scikit-learn Docs](https://scikit-learn.org)
    """)

# Título principal
st.title("🤖 Procesamiento de Datasets en Machine Learning")
st.markdown("""
Esta aplicación implementa las **6 etapas del procesamiento de datos** sobre 3 datasets reales:
Titanic, Student Performance e Iris. Cada ejercicio incluye exploración, limpieza, codificación,
normalización y división de datos.
""")
st.markdown("---")

# Ejecutar el ejercicio seleccionado
if ejercicio_seleccionado == "🚢 Ejercicio 1: Titanic":
    run_ejercicio1()
elif ejercicio_seleccionado == "📚 Ejercicio 2: Student Performance":
    run_ejercicio2()
elif ejercicio_seleccionado == "🌸 Ejercicio 3: Iris":
    run_ejercicio3()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Procesamiento de Datasets en Machine Learning</strong></p>
    <p>Aplicación desarrollada con Streamlit 🎈</p>
    <p>© 2024 - Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)