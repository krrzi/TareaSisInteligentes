import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def run_ejercicio3():
    st.header("🌸 Ejercicio 3: Dataset Iris")
    st.markdown("**Objetivo:** Implementar un flujo completo de preprocesamiento y visualizar resultados")
    st.markdown("---")
    
    try:
        # 1. CARGA DEL DATASET
        st.subheader("1️⃣ Carga del Dataset desde sklearn")
        
        # Cargar dataset Iris
        iris = load_iris()
        
        st.success("✅ Dataset Iris cargado desde sklearn.datasets")
        st.info(f"📊 Dimensiones: {iris.data.shape[0]} muestras × {iris.data.shape[1]} características")
        
        # 2. CONVERSIÓN A DATAFRAME
        st.subheader("2️⃣ Conversión a DataFrame con nombres de columnas")
        
        # Crear DataFrame
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        st.success("✅ DataFrame creado con nombres descriptivos")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Nombres de características:**")
            for i, name in enumerate(iris.feature_names):
                st.write(f"{i+1}. {name}")
        
        with col2:
            st.write("**Nombres de especies (target):**")
            for i, name in enumerate(iris.target_names):
                st.write(f"{i}. {name}")
        
        with st.expander("📋 Ver datos completos"):
            st.dataframe(df.head(10))
        
        # EXPLORACIÓN INICIAL
        st.subheader("📊 Exploración Inicial")
        
        tab1, tab2, tab3 = st.tabs(["📈 Estadísticas", "🎯 Distribución", "📉 Info Dataset"])
        
        with tab1:
            st.write("**Estadísticas Descriptivas:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Distribución por Especie:**")
                species_counts = df['species'].value_counts()
                st.dataframe(species_counts)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                species_counts.plot(kind='bar', color=['#FF6B6B', '#4ECDC4', '#45B7D1'], ax=ax)
                ax.set_title('Distribución de Especies', fontsize=14, fontweight='bold')
                ax.set_xlabel('Especie')
                ax.set_ylabel('Cantidad')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        with tab3:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Muestras", df.shape[0])
            with col2:
                st.metric("Características", df.shape[1] - 2)  # Sin target y species
            with col3:
                st.metric("Clases", len(iris.target_names))
            
            st.write("**Tipos de Datos:**")
            st.dataframe(df.dtypes)
        
        # 3. ESTANDARIZACIÓN
        st.subheader("3️⃣ Estandarización con StandardScaler")
        
        # Separar características y target
        X = df[iris.feature_names]
        y = df['target']
        
        # Aplicar estandarización
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
        
        st.success("✅ Estandarización aplicada con StandardScaler")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Antes de estandarizar:**")
            st.dataframe(X.describe())
        
        with col2:
            st.write("**Después de estandarizar:**")
            st.dataframe(X_scaled_df.describe())
        
        st.info("""
        **StandardScaler** transforma los datos para que tengan:
        - Media = 0
        - Desviación estándar = 1
        """)
        
        # 4. DIVISIÓN DE DATOS
        st.subheader("4️⃣ División del Dataset (70% Entrenamiento - 30% Prueba)")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled_df, y, test_size=0.3, random_state=42, stratify=y
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Completo", f"{X_scaled_df.shape[0]} muestras")
        with col2:
            st.metric("Entrenamiento (70%)", f"{X_train.shape[0]} muestras")
        with col3:
            st.metric("Prueba (30%)", f"{X_test.shape[0]} muestras")
        
        st.info(f"📊 Dimensiones - X_train: {X_train.shape} | X_test: {X_test.shape}")
        
        # Verificar distribución estratificada
        st.write("**Distribución de clases:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Completo:")
            st.write(y.value_counts().sort_index())
        with col2:
            st.write("Entrenamiento:")
            st.write(y_train.value_counts().sort_index())
        with col3:
            st.write("Prueba:")
            st.write(y_test.value_counts().sort_index())
        
        # 5. VISUALIZACIÓN
        st.subheader("5️⃣ Visualización: Sepal Length vs Petal Length por Clase")
        
        # Crear figura con datos estandarizados
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico 1: Datos originales
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        species_names = ['setosa', 'versicolor', 'virginica']
        
        for i, (species, color) in enumerate(zip(species_names, colors)):
            mask = df['species'] == species
            axes[0].scatter(df[mask]['sepal length (cm)'], 
                          df[mask]['petal length (cm)'],
                          c=color, label=species, alpha=0.6, s=100, edgecolors='black')
        
        axes[0].set_xlabel('Sepal Length (cm)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Petal Length (cm)', fontsize=12, fontweight='bold')
        axes[0].set_title('Datos Originales', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Datos estandarizados
        for i, (species, color) in enumerate(zip(species_names, colors)):
            mask = df['species'] == species
            axes[1].scatter(X_scaled_df[mask]['sepal length (cm)'], 
                          X_scaled_df[mask]['petal length (cm)'],
                          c=color, label=species, alpha=0.6, s=100, edgecolors='black')
        
        axes[1].set_xlabel('Sepal Length (estandarizado)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Petal Length (estandarizado)', fontsize=12, fontweight='bold')
        axes[1].set_title('Datos Estandarizados', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Gráfico adicional: Pairplot
        st.subheader("📊 Visualización Adicional: Relaciones entre todas las características")
        
        fig2 = plt.figure(figsize=(12, 10))
        
        # Crear DataFrame con datos estandarizados y species
        df_plot = X_scaled_df.copy()
        df_plot['species'] = df['species'].values
        
        # Pairplot manual usando subplots
        features = iris.feature_names
        n_features = len(features)
        
        for i in range(n_features):
            for j in range(n_features):
                ax = plt.subplot(n_features, n_features, i * n_features + j + 1)
                
                if i == j:
                    # Histograma en la diagonal
                    for species, color in zip(species_names, colors):
                        mask = df_plot['species'] == species
                        ax.hist(df_plot[mask][features[i]], alpha=0.5, color=color, bins=15)
                    ax.set_yticks([])
                else:
                    # Scatter plot fuera de la diagonal
                    for species, color in zip(species_names, colors):
                        mask = df_plot['species'] == species
                        ax.scatter(df_plot[mask][features[j]], 
                                 df_plot[mask][features[i]], 
                                 alpha=0.4, c=color, s=10)
                
                if i == n_features - 1:
                    ax.set_xlabel(features[j].split()[0], fontsize=8)
                else:
                    ax.set_xticks([])
                
                if j == 0:
                    ax.set_ylabel(features[i].split()[0], fontsize=8)
                else:
                    ax.set_yticks([])
        
        plt.suptitle('Matriz de Dispersión - Dataset Iris (Estandarizado)', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # ESTADÍSTICAS DEL DATASET ESTANDARIZADO
        st.subheader("📈 Estadísticas Descriptivas del Dataset Estandarizado")
        
        stats_df = X_scaled_df.describe().T
        stats_df['range'] = stats_df['max'] - stats_df['min']
        
        st.dataframe(stats_df.style.background_gradient(cmap='YlOrRd', subset=['mean', 'std']), 
                     use_container_width=True)
        
        # SALIDA ESPERADA
        st.subheader("📤 Salida Esperada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Primeros registros estandarizados:**")
            result_df = X_scaled_df.head()
            result_df['target'] = y.head().values
            result_df['species'] = df['species'].head().values
            st.dataframe(result_df)
        
        with col2:
            st.write("**Shape de conjuntos:**")
            st.code(f"""
Dataset completo: {X_scaled_df.shape}
Entrenamiento: {X_train.shape}
Prueba: {X_test.shape}

Características: {list(iris.feature_names)}
Clases: {list(iris.target_names)}
            """)
        
        # Botones de descarga
        col1, col2 = st.columns(2)
        with col1:
            csv_original = df.to_csv(index=False)
            st.download_button(
                label="💾 Descargar datos originales (CSV)",
                data=csv_original,
                file_name="iris_original.csv",
                mime="text/csv"
            )
        
        with col2:
            df_scaled_export = X_scaled_df.copy()
            df_scaled_export['target'] = y.values
            df_scaled_export['species'] = df['species'].values
            csv_scaled = df_scaled_export.to_csv(index=False)
            st.download_button(
                label="💾 Descargar datos estandarizados (CSV)",
                data=csv_scaled,
                file_name="iris_estandarizado.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"❌ Error al procesar el dataset: {str(e)}")