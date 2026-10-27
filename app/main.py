"""
app/main.py
API FastAPI que sirve el recomendador colaborativo (MovieLens).

Endpoints:
    GET  /health            -> estado del servicio y del modelo
    GET  /users             -> rango de user_id vÃ¡lidos
    GET  /movies/{movie_id} -> informaciÃ³n de una pelÃ­cula
    POST /recommend         -> recomendaciones para un user_id

LimitaciÃ³n conocida: solo se sirve el modelo colaborativo (matriz de
similitud entre 610 usuarios de MovieLens). El content-based (724 MB) no
se incluye por lÃ­mite de tamaÃ±o de GitHub, y el hÃ­brido depende de Ã©l.
"""

import ast
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.collaborative import CollaborativeRecommender
from app.preprocess import DataPreprocessor

# Silenciar warnings de numpy/sklearn al cargar
warnings.filterwarnings("ignore")

# Rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def as_list(value) -> List[str]:
    """
    Convierte un valor de columna 'lista' a lista de Python.
    Soporta: list, tuple, numpy.ndarray, string JSON-like y NaN.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        # Fallback: separado por comas
        return [p.strip() for p in s.split(",") if p.strip()]
    # numpy.float64/nan y demÃ¡s
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return []


# ---------------------------------------------------------------------------
# Estado global (se carga UNA vez al arrancar)
# ---------------------------------------------------------------------------
class ModelState:
    recommender: CollaborativeRecommender | None = None
    movies_df: pd.DataFrame | None = None
    movies_indexed: pd.DataFrame | None = None
    ratings_df: pd.DataFrame | None = None
    user_ids: List[int] = []
    movie_ids: List[int] = []

    @property
    def cargado(self) -> bool:
        return self.recommender is not None and self.recommender.is_fitted


estado = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga datos y modelo al arrancar, libera al parar."""
    print("[startup] Cargando ratings y pelÃ­culas...")
    ratings_df = pd.read_parquet(DATA_DIR / "ratings.parquet")
    movies_df = pd.read_parquet(DATA_DIR / "movies.parquet")
    ratings_df["user_id"] = ratings_df["user_id"].astype(int)
    ratings_df["movie_id"] = ratings_df["movie_id"].astype(int)
    movies_df["movie_id"] = movies_df["movie_id"].astype(int)

    print(f"[startup] {len(ratings_df):,} ratings, {len(movies_df):,} pelÃ­culas")

    print("[startup] Reconstruyendo matriz usuario-pelÃ­cula...")
    preprocesador = DataPreprocessor()
    ratings_filtrados = preprocesador.filter_ratings(ratings_df)
    user_movie_matrix = preprocesador.create_user_movie_matrix(ratings_filtrados)

    print(f"[startup] Matriz: {user_movie_matrix.shape}")

    # Crear instancia del recomendador y cargar la matriz de similitud
    rec = CollaborativeRecommender(preprocessor=preprocesador)
    rec.user_movie_matrix = user_movie_matrix
    rec.user_to_idx, rec.idx_to_user = preprocesador.get_user_index_mapping(user_movie_matrix)
    movies_in_matrix = user_movie_matrix.columns.tolist()
    rec.movie_to_idx = {m: i for i, m in enumerate(movies_in_matrix)}
    rec.idx_to_movie = {i: m for i, m in enumerate(movies_in_matrix)}

    sim_path = MODELS_DIR / "collaborative_similarity.joblib"
    print(f"[startup] Cargando matriz de similitud: {sim_path.name}")
    rec.user_similarity_matrix = joblib.load(sim_path)
    rec.is_fitted = True

    print(f"[startup] Similitud: {rec.user_similarity_matrix.shape}")

    estado.recommender = rec
    estado.ratings_df = ratings_df
    estado.movies_df = movies_df
    estado.movies_indexed = movies_df.set_index("movie_id")
    estado.user_ids = sorted(int(u) for u in rec.user_to_idx.keys())
    estado.movie_ids = sorted(int(m) for m in rec.movie_to_idx.keys())

    print(f"[startup] Listo. Usuarios vÃ¡lidos: {len(estado.user_ids)}, "
          f"pelÃ­culas en matriz: {len(estado.movie_ids)}")

    yield

    estado.recommender = None
    estado.movies_df = None
    estado.movies_indexed = None
    estado.ratings_df = None
    estado.user_ids = []
    estado.movie_ids = []
    print("[shutdown] Modelo liberado")


app = FastAPI(
    title="Movie Recommender API",
    description=(
        "API que sirve un recomendador colaborativo entrenado con MovieLens. "
        "LimitaciÃ³n: solo usuarios 1..610 del dataset original."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="ID de usuario MovieLens (1..610)")
    top_n: int = Field(10, ge=1, le=50, description="NÃºmero de recomendaciones")


class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: List[str]
    score: float
    score_pct: float


class RecommendResponse(BaseModel):
    user_id: int
    top_n: int
    recomendaciones: List[RecommendedMovie]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if estado.cargado else "modelo_no_cargado",
        "usuarios": len(estado.user_ids),
        "peliculas_matriz": len(estado.movie_ids),
        "usuarios_rango": [min(estado.user_ids), max(estado.user_ids)] if estado.user_ids else None,
    }


@app.get("/users")
def users():
    """InformaciÃ³n Ãºtil para saber quÃ© user_id se pueden consultar."""
    if not estado.cargado:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        "total": len(estado.user_ids),
        "min": min(estado.user_ids),
        "max": max(estado.user_ids),
        "ejemplo": estado.user_ids[:10],
    }


@app.get("/movies/{movie_id}")
def movie_detail(movie_id: int):
    if not estado.cargado or estado.movies_indexed is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    if movie_id not in estado.movies_indexed.index:
        raise HTTPException(status_code=404, detail=f"PelÃ­cula {movie_id} no encontrada")

    r = estado.movies_indexed.loc[movie_id].to_dict()

    return {
        "movie_id": int(movie_id),
        "title": str(r.get("title") or ""),
        "title_clean": str(r.get("title_clean") or ""),
        "year": int(r.get("year", 0) or 0),
        "genres": as_list(r.get("genres_processed")),
        "poster_path": str(r.get("poster_path") or ""),
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest):
    if not estado.cargado or estado.recommender is None or estado.movies_indexed is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    if payload.user_id not in estado.recommender.user_to_idx:
        raise HTTPException(
            status_code=404,
            detail=f"Usuario {payload.user_id} no existe. "
                   f"Rango vÃ¡lido: {min(estado.user_ids)}..{max(estado.user_ids)}",
        )

    recs = estado.recommender.recommend_for_user(
        user_id=payload.user_id,
        top_n=payload.top_n,
        exclude_watched=True,
    )

    recomendaciones: List[RecommendedMovie] = []
    for movie_id, score in recs:
        if movie_id not in estado.movies_indexed.index:
            continue
        row = estado.movies_indexed.loc[movie_id]
        recomendaciones.append(
            RecommendedMovie(
                movie_id=int(movie_id),
                title=str(row.get("title") or "Desconocido"),
                year=int(row.get("year", 0) or 0),
                genres=as_list(row.get("genres_processed")),
                score=round(float(score), 4),
                score_pct=round(float(score) * 100, 2),
            )
        )

    return RecommendResponse(
        user_id=payload.user_id,
        top_n=payload.top_n,
        recomendaciones=recomendaciones,
    )
