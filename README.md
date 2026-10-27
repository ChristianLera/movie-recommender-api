# Movie Recommender API

API REST construida con **FastAPI** que sirve un recomendador colaborativo
entrenado con **MovieLens** (100.836 ratings de 610 usuarios). Empaquetada
en **Docker** para que cualquiera pueda ejecutarla sin instalar Python.

---

## ¿Qué hace?

Dado un user_id de MovieLens (1..610), devuelve una lista de películas
recomendadas para ese usuario, con título, año, géneros y score de afinidad.

**Algoritmo**: filtro colaborativo basado en usuarios. Calcula la similitud
coseno entre usuarios a partir de la matriz usuario×película, y recomienda
películas que gustaron a usuarios similares (excluyendo las ya vistas).

**Limitación conocida**: solo se sirve el modelo **colaborativo**.
El recomendador content-based (724 MB) no se incluye en el repo por límite
de tamaño de GitHub, y el modelo híbrido depende de él. Esta es una decisión
consciente de ingeniería: mejor un servicio ligero, reproducible y honesto
sobre sus límites que un repo de 1 GB que nadie va a clonar.

---

## Arquitectura

    Cliente (curl, Swagger, app externa)
              │
              ▼ HTTP POST /recommend  { "user_id": 1, "top_n": 10 }
       ┌─────────────────┐
       │  FastAPI        │
       │  (uvicorn)      │
       └────────┬────────┘
                │
        ┌───────┴─────────────┐
        │  CollaborativeRecommender │
        │  (similitud coseno) │
        └───────┬─────────────┘
                │
        ┌───────┴─────────────┐
        │ user_similarity.joblib  (610×610)
        │ ratings.parquet    (100.836)
        │ movies.parquet     (9.742)
        └───────┬─────────────┘
                │
                ▼
           JSON response

---

## Estructura del proyecto

    movie-recommender-api/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py              # FastAPI: /health, /users, /movies/{id}, /recommend
    │   ├── collaborative.py     # recomendador colaborativo (del proyecto original)
    │   ├── preprocess.py        # preprocesamiento de ratings/películas
    │   └── utils.py             # utilidades comunes
    ├── data/
    │   ├── movies.parquet       # 9.742 películas (515 KB)
    │   └── ratings.parquet      # 100.836 ratings (260 KB)
    ├── models/
    │   └── collaborative_similarity.joblib   # matriz 610×610 (2.8 MB)
    ├── requirements.txt
    ├── Dockerfile
    ├── .dockerignore
    ├── .gitignore
    └── README.md

Tamaño del repo: ~4 MB. Imagen Docker: ~1 GB.

---

## Requisitos

- **Docker** (probado con Docker Desktop 29.4.2 en Windows 11)
- **No necesitas Python instalado**: la imagen lo lleva dentro.

---

## Cómo construir y ejecutar el contenedor

### 1. Construir la imagen

Desde esta carpeta:

    docker build -t movie-recommender-api:1.0 .

### 2. Ejecutar el contenedor

    docker run -d -p 8002:8000 --name movie-recommender-api movie-recommender-api:1.0

La API queda disponible en http://localhost:8002.

### 3. Comprobar que está sana

    docker ps

Debe aparecer (healthy) en la columna STATUS (tarda ~15s en arrancar
porque reconstruye la matriz usuario-película).

### 4. Detenerlo y borrarlo

    docker stop movie-recommender-api
    docker rm movie-recommender-api

---

## Cómo probarla

### Opción A: Swagger UI (navegador)

Abre http://localhost:8002/docs y prueba los endpoints interactivamente.

### Opción B: PowerShell (Invoke-RestMethod)

**Health check:**

    Invoke-RestMethod -Uri "http://localhost:8002/health"

**Recomendaciones para el usuario 1:**

    Invoke-RestMethod -Uri "http://localhost:8002/recommend" `
      -Method Post `
      -ContentType "application/json" `
      -Body '{\"user_id\": 1, \"top_n\": 5}'

**Información de una película:**

    Invoke-RestMethod -Uri "http://localhost:8002/movies/1"

### Ejemplo de respuesta

    {
      "user_id": 1,
      "top_n": 5,
      "recomendaciones": [
        {
          "movie_id": 6460,
          "title": "Trial, The (Procès, Le) (1962)",
          "year": 1962,
          "genres": ["Drama"],
          "score": 0.9665,
          "score_pct": 96.65
        }
      ]
    }

---

## Endpoints

| Método | Ruta                | Descripción                                  |
|--------|---------------------|----------------------------------------------|
| GET    | /health           | Estado del servicio y del modelo             |
| GET    | /users            | Rango de user_id válidos (1..610)          |
| GET    | /movies/{movie_id}| Información de una película                  |
| POST   | /recommend        | Recomendaciones para un user_id            |
| GET    | /docs             | Documentación interactiva (Swagger UI)       |

---

## Detalles técnicos

- **Modelo cargado una sola vez** al arrancar (evento lifespan de FastAPI).
  La matriz usuario×película se reconstruye en memoria desde los .parquet,
  y la matriz de similitud se lee del .joblib. Todo queda en RAM.
- **Validación de entrada con Pydantic**: user_id debe existir en el modelo,
  	op_n entre 1 y 50. Errores de formato → HTTP 422; usuario inexistente → HTTP 404.
- **Healthcheck** en el Dockerfile: Docker consulta /health cada 30s.
- **Imagen base** python:3.11-slim, con uild-essential solo durante el build.
- **Versiones fijadas** en equirements.txt. **Importante**: pyarrow<26
  y 
umpy<2.0.0 son obligatorios; pyarrow>=26 exige 
umpy>=2 y rompe
  scikit-learn en este proyecto.

---

## Datos

Los tres artefactos (movies.parquet, atings.parquet,
collaborative_similarity.joblib) provienen del proyecto original
[Recomendador de películas](../Recomendador%20de%20peliculas/README.md),
que a su vez usa el dataset público [MovieLens](https://grouplens.org/datasets/movielens/).

---

## Tecnologías

- Python 3.11
- FastAPI + Uvicorn
- scikit-learn 1.9.1
- pandas 2.3.3 + pyarrow 25.x
- Docker

---

## Autor

Christian Lera — [github.com/ChristianLera](https://github.com/ChristianLera)
