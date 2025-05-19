# 🧠 Backend de Evaluación del Riesgo de Trastorno del Espectro Autista (TEA)

Este backend implementado con **FastAPI** permite procesar evaluaciones clínicas, predecir riesgos de TEA utilizando modelos de Machine Learning, y almacenar los resultados en una base de datos PostgreSQL. El proyecto puede ejecutarse localmente y también ser desplegado fácilmente en **Railway**.

---

## 🚀 Tecnologías utilizadas

- FastAPI + Uvicorn
- SQLAlchemy
- PostgreSQL
- Pandas & Joblib
- Alembic (opcional para migraciones)
- Railway (despliegue)

---

## ⚙️ Requisitos

- Python 3.10+
- Git
- PostgreSQL (local o en la nube)
- Railway (opcional para despliegue)

---

## 📦 Instalación local

### 1. Clona el repositorio

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

### 2. Crear y activar un entorno virtual

#### En Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

#### En PowerShell (Windows):
```bash
.\venv\Scripts\Activate.ps1
```

Si se lanza un error por permisos:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Crear el archivo .env

Crea un archivo llamado `.env` en la raíz del proyecto con este contenido:

```
DATABASE_URL=postgresql+psycopg2://usuario:contraseña@host:puerto/nombrebd
```

Reemplaza los valores con tus credenciales de PostgreSQL.

### 5. Ejecutar la aplicación localmente

```bash
uvicorn app.main:app --reload
```

Abre en tu navegador: http://localhost:8000/docs

---

## 🗄️ Estructura del proyecto

```
.
├── app/
│   ├── db/
│   ├── model/
│   ├── schemas/
│   ├── utils/
│   ├── api.py
│   └── main.py
├── model.pkl
├── pca_model.pkl
├── requirements.txt
├── Procfile
├── .env (no subir a GitHub)
├── .gitignore
└── README.md
```

---

## 🛣️ Endpoints principales

### Documentación interactiva
- `/docs` - Documentación Swagger UI
- `/redoc` - Documentación ReDoc

### 🔍 Endpoint principal
- `POST /predict` - Este endpoint:
  - Recibe una lista de 25 valores numéricos o binarios
  - Realiza preprocesamiento usando un modelo PCA
  - Aplica un modelo ML previamente entrenado (`model.pkl`)
  - Devuelve una clase (`0` o `1`) y un nivel de confianza
  - Guarda la evaluación completa en la base de datos PostgreSQL

---

## 🔄 Configuración de la base de datos

### Crear las tablas (primera vez)

```bash
# Usando SQLAlchemy
python -c "from app.db.database import Base, engine; Base.metadata.create_all(bind=engine)"

# O usando Alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head
```

---

## 🧪 Ejecutar tests

```bash
pytest
```

---

## 🚂 Despliegue en Railway

### 1. Crear proyecto y base de datos
- Ve a https://railway.app
- Crea un nuevo proyecto
- Agrega PostgreSQL como plugin
- Copia la cadena DATABASE_URL

### 2. Subir el repositorio a GitHub
```bash
git init
git add .
git commit -m "Primera versión"
git branch -M main
git remote add origin https://github.com/tu-usuario/tu-repo.git
git push -u origin main
```

### 3. En Railway: New Project > Deploy from GitHub
- Selecciona el repositorio subido
- Railway instalará automáticamente las dependencias usando `requirements.txt`

### 4. Agregar variables de entorno en Railway
En el panel de "Variables", define:
```
DATABASE_URL=postgresql+psycopg2://usuario:contraseña@host:puerto/nombrebd
AUTO_CREATE_TABLES=true
```

### 5. Railway desplegará tu backend automáticamente
Puedes acceder al endpoint en:
```
https://<tu-app>.up.railway.app/docs
```

---

## 📊 Modelado de Machine Learning

El sistema utiliza modelos pre-entrenados para la predicción del riesgo de TEA:

1. **Preprocesamiento**: El modelo PCA (`pca_model.pkl`) reduce la dimensionalidad de los 25 valores de entrada
2. **Clasificación**: El modelo principal (`model.pkl`) implementa un algoritmo de clasificación que determina:
   - Una clase binaria (`0` = Sin riesgo de TEA, `1` = Con riesgo de TEA)
   - Un nivel de confianza para la predicción

Ambos modelos fueron entrenados previamente y se cargan mediante `joblib` para realizar las predicciones en tiempo real.

---

## 🤝 Contribuir

1. Haz fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/amazing-feature`)
3. Haz commit de tus cambios (`git commit -m 'Add some amazing feature'`)
4. Empuja a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

---

## 📄 Archivos de configuración

### Procfile
Este archivo le dice a Railway cómo ejecutar FastAPI:
```bash
web: uvicorn app.main:app --host=0.0.0.0 --port=${PORT:-8000}
```

### .gitignore recomendado
Asegúrate de tener este contenido para evitar subir archivos innecesarios:
```bash
venv/
__pycache__/
*.pyc
.env
```

## 📞 Contacto

Desarrollado por [Ernesto Saniel Castro Lozano]  
📧 Email: [ernestosaniel123@gmail.com]

Proyecto académico orientado a la predicción del riesgo de TEA utilizando modelos de aprendizaje automático.
