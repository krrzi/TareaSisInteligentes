import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def run_ejercicio1():
    st.header("🚢 Ejercicio 1: Análisis del Dataset Titanic")
    st.markdown("**Objetivo:** Preparar los datos para predecir la supervivencia de los pasajeros")
    st.markdown("---")
    
    # Cargar datos
    uploaded_file = st.file_uploader("📂 Sube el archivo titanic.csv", type=['csv'], key='titanic')
    
    if uploaded_file is not None:
        try:
            # 1. CARGA DEL DATASET
            st.subheader("1️⃣ Carga del Dataset")
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset cargado correctamente: {df.shape[0]} filas y {df.shape[1]} columnas")
            
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
            
            tab1, tab2, tab3 = st.tabs(["📊 Info General", "❌ Valores Nulos", "📈 Estadísticas"])
            
            with tab1:
                st.write("**Tipos de Datos:**")
                info_df = pd.DataFrame({
                    'Columna': df.columns,
                    'Tipo': df.dtypes.values,
                    'No Nulos': df.count().values
                })
                st.dataframe(info_df, use_container_width=True)
            
            with tab2:
                null_counts = df.isnull().sum()
                null_df = pd.DataFrame({
                    'Columna': null_counts.index,
                    'Valores Nulos': null_counts.values,
                    'Porcentaje': (null_counts.values / len(df) * 100).round(2)
                })
                st.dataframe(null_df[null_df['Valores Nulos'] > 0], use_container_width=True)
            
            with tab3:
                st.dataframe(df.describe(), use_container_width=True)
            
            # 3. LIMPIEZA DE DATOS
            st.subheader("3️⃣ Limpieza de Datos")
            
            df_clean = df.copy()
            
            # Eliminar columnas irrelevantes
            columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
            columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
            df_clean = df_clean.drop(columns=columns_to_drop)
            st.info(f"🗑️ Columnas eliminadas: {', '.join(columns_to_drop)}")
            
            # Verificar duplicados
            duplicados = df_clean.duplicated().sum()
            st.write(f"**Duplicados encontrados:** {duplicados}")
            if duplicados > 0:
                df_clean = df_clean.drop_duplicates()
                st.success(f"✅ {duplicados} filas duplicadas eliminadas")
            
            # Manejo de valores nulos
            st.write("**Tratamiento de valores nulos:**")
            
            if 'Age' in df_clean.columns:
                age_median = df_clean['Age'].median()
                df_clean['Age'].fillna(age_median, inplace=True)
                st.write(f"- Age: Reemplazado con mediana ({age_median:.2f})")
            
            if 'Fare' in df_clean.columns:
                fare_median = df_clean['Fare'].median()
                df_clean['Fare'].fillna(fare_median, inplace=True)
                st.write(f"- Fare: Reemplazado con mediana ({fare_median:.2f})")
            
            if 'Embarked' in df_clean.columns:
                embarked_mode = df_clean['Embarked'].mode()[0]
                df_clean['Embarked'].fillna(embarked_mode, inplace=True)
                st.write(f"- Embarked: Reemplazado con moda ({embarked_mode})")
            
            st.success(f"✅ Valores nulos restantes: {df_clean.isnull().sum().sum()}")
            
            # 4. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
            st.subheader("4️⃣ Codificación de Variables Categóricas")
            
            # Codificar Sex
            if 'Sex' in df_clean.columns:
                le_sex = LabelEncoder()
                df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])
                mapping_sex = dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))
                st.write(f"**Sex:** {mapping_sex}")
            
            # Codificar Embarked
            if 'Embarked' in df_clean.columns:
                le_embarked = LabelEncoder()
                df_clean['Embarked'] = le_embarked.fit_transform(df_clean['Embarked'])
                mapping_embarked = dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))
                st.write(f"**Embarked:** {mapping_embarked}")
            
            st.success("✅ Variables categóricas codificadas")
            
            # 5. ESTANDARIZACIÓN
            st.subheader("5️⃣ Estandarización de Variables Numéricas")
            
            # Separar características y variable objetivo
            if 'Survived' in df_clean.columns:
                X = df_clean.drop('Survived', axis=1)
                y = df_clean['Survived']
            else:
                X = df_clean
                y = None
            
            # Identificar columnas numéricas para estandarizar
            numeric_cols = ['Age', 'Fare']
            numeric_cols = [col for col in numeric_cols if col in X.columns]
            
            scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
            st.write(f"**Columnas estandarizadas:** {', '.join(numeric_cols)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Antes de estandarizar:**")
                st.dataframe(X[numeric_cols].describe(), use_container_width=True)
            with col2:
                st.write("**Después de estandarizar:**")
                st.dataframe(X_scaled[numeric_cols].describe(), use_container_width=True)
            
            # 6. DIVISIÓN DE DATOS
            st.subheader("6️⃣ División en Conjuntos de Entrenamiento y Prueba")
            
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dataset Completo", f"{X_scaled.shape[0]} registros")
                with col2:
                    st.metric("Entrenamiento (70%)", f"{X_train.shape[0]} registros")
                with col3:
                    st.metric("Prueba (30%)", f"{X_test.shape[0]} registros")
                
                st.info(f"📊 Dimensiones - Entrenamiento: {X_train.shape} | Prueba: {X_test.shape}")
                
                # SALIDA ESPERADA
                st.subheader("📤 Salida Esperada")
                
                st.write("**Primeros 5 registros procesados:**")
                result_df = X_scaled.head()
                if y is not None:
                    result_df['Survived'] = y.head().values
                st.dataframe(result_df, use_container_width=True)
                
                st.write("**Shape de conjuntos:**")
                st.code(f"""
Conjunto de entrenamiento: {X_train.shape}
Conjunto de prueba: {X_test.shape}
                """)
                
                # Botón de descarga
                csv = X_scaled.to_csv(index=False)
                st.download_button(
                    label="💾 Descargar datos procesados (CSV)",
                    data=csv,
                    file_name="titanic_procesado.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
    else:
        st.info("👆 Por favor, sube un archivo CSV del dataset Titanic para comenzar")
        st.markdown("""
        **Puedes obtener el dataset de:**
        - [Kaggle - Titanic Dataset](https://www.kaggle.com/c/titanic/data)
        - Librería Seaborn: `sns.load_dataset('titanic')`
        """)