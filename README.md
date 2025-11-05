# 🤖 Procesamiento de Datasets en Machine Learning

Aplicación interactiva en Streamlit para el procesamiento completo de 3 datasets: Titanic, Student Performance e Iris.

## 📋 Descripción

Esta aplicación implementa las **6 etapas del procesamiento de datos**:
1. ✅ Carga del dataset
2. 🔍 Exploración inicial
3. 🧹 Limpieza de datos
4. 🔢 Codificación de variables
5. 📊 Normalización/Estandarización
6. ✂️ División de datos

## 📁 Estructura del Proyecto

```
proyecto/
├── main.py                    # Archivo principal de la aplicación
├── ejercicio1_titanic.py      # Ejercicio 1: Dataset Titanic
├── ejercicio2_student.py      # Ejercicio 2: Student Performance
├── ejercicio3_iris.py         # Ejercicio 3: Dataset Iris
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Este archivo
```

## 🚀 Instalación

### 1. Clonar o descargar el repositorio

```bash
git clone <tu-repositorio>
cd proyecto
```

### 2. Crear un entorno virtual (opcional pero recomendado)

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 💻 Uso Local

Para ejecutar la aplicación localmente:

```bash
streamlit run main.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## ☁️ Despliegue en Streamlit Cloud

### Paso 1: Preparar los archivos

Asegúrate de tener todos estos archivos en tu repositorio:
- `main.py`
- `ejercicio1_titanic.py`
- `ejercicio2_student.py`
- `ejercicio3_iris.py`
- `requirements.txt`
- `README.md`

### Paso 2: Subir a GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <tu-repositorio-github>
git push -u origin main
```

### Paso 3: Desplegar en Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Click en "New app"
4. Selecciona tu repositorio
5. Branch: `main`
6. Main file path: `main.py`
7. Click en "Deploy"

¡Listo! Tu aplicación estará disponible en unos minutos.

## 📊 Datasets Requeridos

### Ejercicio 1: Titanic
- **Fuente:** [Kaggle - Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- **Archivo:** `titanic.csv`
- Descarga y súbelo a través de la interfaz de la aplicación

### Ejercicio 2: Student Performance
- **Fuente:** [Kaggle - Student Alcohol Consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption)
- **Archivo:** `student-mat.csv`
- Descarga y súbelo a través de la interfaz de la aplicación

### Ejercicio 3: Iris
- **Fuente:** Incluido en scikit-learn
- No requiere descarga, se carga automáticamente

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Streamlit** - Framework para la aplicación web
- **Pandas** - Manipulación de datos
- **NumPy** - Operaciones numéricas
- **Scikit-learn** - Preprocesamiento y datasets
- **Matplotlib & Seaborn** - Visualizaciones

## 📝 Funcionalidades por Ejercicio

### 🚢 Ejercicio 1: Titanic
- Eliminación de columnas irrelevantes
- Manejo de valores nulos (media/moda)
- Codificación de variables categóricas (Sex, Embarked)
- Estandarización de variables numéricas
- División 70/30

### 📚 Ejercicio 2: Student Performance
- Análisis de variables categóricas
- Eliminación de duplicados
- One Hot Encoding
- Normalización con MinMaxScaler
- División 80/20
- **Reto adicional:** Correlación entre G1, G2, G3

### 🌸 Ejercicio 3: Iris
- Carga desde sklearn
- Conversión a DataFrame
- Estandarización con StandardScaler
- División 70/30
- Visualizaciones de dispersión por clase

## 📈 Características Adicionales

- ✨ Interfaz interactiva e intuitiva
- 📊 Visualizaciones en tiempo real
- 💾 Descarga de datos procesados
- 📱 Responsive design
- 🎨 Diseño moderno y profesional

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

## 👤 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- Email: tu-email@ejemplo.com

## 🙏 Agradecimientos

- Kaggle por proporcionar los datasets
- Scikit-learn por las herramientas de ML
- Streamlit por el framework
- UCI Machine Learning Repository

---

⭐ Si te gustó este proyecto, no olvides darle una estrella en GitHub