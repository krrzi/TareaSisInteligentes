import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def run_ejercicio2():
    st.header("📚 Ejercicio 2: Student Performance Dataset")
    st.markdown("**Objetivo:** Procesar datos para predecir la nota final (G3) de estudiantes")
    st.markdown("---")
    
    # Cargar datos
    uploaded_file = st.file_uploader("📂 Sube el archivo student-mat.csv", type=['csv'], key='student')
    
    if uploaded_file is not None:
        try:
            # 1. CARGA DEL DATASET
            st.subheader("1️⃣ Carga del Dataset")
            df = pd.read_csv(uploaded_file, sep=';')  # A veces usa punto y coma
            if df.shape[1] == 1:  # Si no funcionó con ;, intentar con ,
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
            
            with st.expander("📋 Ver datos originales"):
                st.dataframe(df.head(10))
            
            # 2. EXPLORACIÓN INICIAL
            st.subheader("2️⃣ Exploración Inicial")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Filas", df.shape[0])
            with col2:
                st.metric("Total de Columnas", df.shape[1])
            with col3:
                st.metric("Valores Nulos", df.isnull().sum().sum())
            
            # Identificar variables categóricas y numéricas
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            tab1, tab2, tab3 = st.tabs(["📊 Variables", "❌ Valores Nulos", "📈 Estadísticas"])
            
            with tab1:
                col_cat, col_num = st.columns(2)
                with col_cat:
                    st.write("**Variables Categóricas:**")
                    st.write(categorical_cols)
                with col_num:
                    st.write("**Variables Numéricas:**")
                    st.write(numerical_cols)
            
            with tab2:
                null_counts = df.isnull().sum()
                if null_counts.sum() > 0:
                    null_df = pd.DataFrame({
                        'Columna': null_counts.index,
                        'Valores Nulos': null_counts.values,
                        'Porcentaje': (null_counts.values / len(df) * 100).round(2)
                    })
                    st.dataframe(null_df[null_df['Valores Nulos'] > 0])
                else:
                    st.success("✅ No hay valores nulos en el dataset")
            
            with tab3:
                st.dataframe(df.describe(), use_container_width=True)
            
            # 3. LIMPIEZA DE DATOS
            st.subheader("3️⃣ Limpieza de Datos")
            
            df_clean = df.copy()
            
            # Eliminar duplicados
            duplicados_antes = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates()
            duplicados_eliminados = duplicados_antes - df_clean.shape[0]
            
            if duplicados_eliminados > 0:
                st.warning(f"🗑️ {duplicados_eliminados} filas duplicadas eliminadas")
            else:
                st.success("✅ No se encontraron duplicados")
            
            # Verificar valores inconsistentes
            st.write("**Verificación de valores inconsistentes:**")
            
            # Verificar que G1, G2, G3 estén en rango [0, 20]
            grade_cols = [col for col in ['G1', 'G2', 'G3'] if col in df_clean.columns]
            inconsistentes = 0
            for col in grade_cols:
                mask = (df_clean[col] < 0) | (df_clean[col] > 20)
                inconsistentes += mask.sum()
                if mask.sum() > 0:
                    df_clean = df_clean[~mask]
                    st.write(f"- {col}: {mask.sum()} valores fuera de rango [0-20] eliminados")
            
            if inconsistentes == 0:
                st.success("✅ No se encontraron valores inconsistentes en las notas")
            
            # Verificar age
            if 'age' in df_clean.columns:
                mask_age = (df_clean['age'] < 15) | (df_clean['age'] > 22)
                if mask_age.sum() > 0:
                    st.info(f"⚠️ {mask_age.sum()} estudiantes con edad fuera del rango típico [15-22]")
            
            # 4. ONE HOT ENCODING
            st.subheader("4️⃣ One Hot Encoding de Variables Categóricas")
            
            # Seleccionar variables categóricas para codificar
            categorical_to_encode = [col for col in categorical_cols if col in df_clean.columns]
            
            if categorical_to_encode:
                st.write(f"**Variables a codificar:** {', '.join(categorical_to_encode)}")
                
                # Aplicar One Hot Encoding
                df_encoded = pd.get_dummies(df_clean, columns=categorical_to_encode, drop_first=True)
                
                st.success(f"✅ Codificación completada. Nuevas dimensiones: {df_encoded.shape}")
                st.info(f"📊 Columnas antes: {df_clean.shape[1]} → Columnas después: {df_encoded.shape[1]}")
                
                with st.expander("📋 Ver primeras filas codificadas"):
                    st.dataframe(df_encoded.head(10))
            else:
                df_encoded = df_clean.copy()
                st.info("No hay variables categóricas para codificar")
            
            # 5. NORMALIZACIÓN
            st.subheader("5️⃣ Normalización de Variables Numéricas")
            
            # Identificar columnas numéricas para normalizar
            numeric_cols_to_normalize = ['age', 'absences', 'G1', 'G2']
            numeric_cols_to_normalize = [col for col in numeric_cols_to_normalize if col in df_encoded.columns]
            
            if numeric_cols_to_normalize:
                scaler = MinMaxScaler()
                df_normalized = df_encoded.copy()
                df_normalized[numeric_cols_to_normalize] = scaler.fit_transform(
                    df_encoded[numeric_cols_to_normalize]
                )
                
                st.write(f"**Columnas normalizadas:** {', '.join(numeric_cols_to_normalize)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Antes de normalizar:**")
                    st.dataframe(df_encoded[numeric_cols_to_normalize].describe())
                with col2:
                    st.write("**Después de normalizar:**")
                    st.dataframe(df_normalized[numeric_cols_to_normalize].describe())
            else:
                df_normalized = df_encoded.copy()
            
            # 6. SEPARACIÓN X y y
            st.subheader("6️⃣ Separación de Características (X) y Variable Objetivo (y)")
            
            if 'G3' in df_normalized.columns:
                X = df_normalized.drop('G3', axis=1)
                y = df_normalized['G3']
                
                st.success(f"✅ X (características): {X.shape} | y (objetivo): {y.shape}")
                
                # 7. DIVISIÓN TRAIN/TEST
                st.subheader("7️⃣ División en Entrenamiento (80%) y Prueba (20%)")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dataset Completo", f"{X.shape[0]} registros")
                with col2:
                    st.metric("Entrenamiento (80%)", f"{X_train.shape[0]} registros")
                with col3:
                    st.metric("Prueba (20%)", f"{X_test.shape[0]} registros")
                
                st.info(f"📊 Dimensiones - X_train: {X_train.shape} | X_test: {X_test.shape}")
                
                # RETO ADICIONAL: Correlación entre G1, G2, G3
                st.subheader("🎯 Reto Adicional: Análisis de Correlación")
                
                grade_cols_available = [col for col in ['G1', 'G2', 'G3'] if col in df_clean.columns]
                
                if len(grade_cols_available) >= 2:
                    correlation_matrix = df_clean[grade_cols_available].corr()
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**Matriz de Correlación:**")
                        st.dataframe(correlation_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                                    center=0, square=True, linewidths=1, ax=ax,
                                    vmin=-1, vmax=1, fmt='.3f')
                        ax.set_title('Correlación entre Notas', fontsize=14, fontweight='bold')
                        st.pyplot(fig)
                    
                    st.write("**Interpretación:**")
                    if 'G1' in grade_cols_available and 'G2' in grade_cols_available:
                        corr_g1_g2 = correlation_matrix.loc['G1', 'G2']
                        st.write(f"- Correlación G1-G2: {corr_g1_g2:.3f}")
                    if 'G1' in grade_cols_available and 'G3' in grade_cols_available:
                        corr_g1_g3 = correlation_matrix.loc['G1', 'G3']
                        st.write(f"- Correlación G1-G3: {corr_g1_g3:.3f}")
                    if 'G2' in grade_cols_available and 'G3' in grade_cols_available:
                        corr_g2_g3 = correlation_matrix.loc['G2', 'G3']
                        st.write(f"- Correlación G2-G3: {corr_g2_g3:.3f}")
                
                # SALIDA ESPERADA
                st.subheader("📤 Salida Esperada")
                st.write("**Primeros 5 registros procesados:**")
                result_df = df_normalized.head()
                st.dataframe(result_df, use_container_width=True)
                
                # Botón de descarga
                csv = df_normalized.to_csv(index=False)
                st.download_button(
                    label="💾 Descargar datos procesados (CSV)",
                    data=csv,
                    file_name="student_performance_procesado.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ No se encontró la columna 'G3' en el dataset")
            
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
            st.info("Intenta verificar el delimitador del CSV (puede ser ';' o ',')")
    else:
        st.info("👆 Por favor, sube un archivo CSV del dataset Student Performance")
        st.markdown("""
        **Puedes obtener el dataset de:**
        - [Kaggle - Student Performance](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)
        - [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/student+performance)
        """)