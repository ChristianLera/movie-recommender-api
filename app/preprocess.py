"""
MÃ³dulo de preprocesamiento para MovieLens.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from app.utils import setup_logging, normalize_text

logger = setup_logging(log_to_ui=True)

class DataPreprocessor:
    """Preprocesador para datos de MovieLens."""
    
    def __init__(self, min_ratings_per_user=5, min_ratings_per_movie=3, top_actors_limit=5):
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_movie = min_ratings_per_movie
        self.top_actors_limit = top_actors_limit
        self.vectorizer = None
        logger.info("Preprocesador inicializado")
    
    def prepare_movies_for_similarity(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Prepara pelÃ­culas para similitud."""
        if movies_df.empty:
            return movies_df
        
        df = movies_df.copy()
        
        # Limpiar tÃ­tulos
        if 'title' in df.columns:
            df['title_clean'] = df['title'].apply(self._clean_title)
        else:
            df['title_clean'] = "Desconocido"
        
        # Extraer aÃ±os
        if 'year' not in df.columns and 'title' in df.columns:
            df['year'] = df['title'].apply(self._extract_year)
        elif 'year' not in df.columns:
            df['year'] = 0
        
        # Procesar gÃ©neros
        if 'genres' in df.columns:
            genres_list = []
            genres_str_list = []
            
            for genres_value in df['genres']:
                processed = self._process_genres(genres_value)
                genres_list.append(processed)
                genres_str_list.append(' '.join(processed))
            
            df['genres_processed'] = genres_list
            df['genres_str'] = genres_str_list
            df['combined_features'] = genres_str_list
        else:
            df['genres_processed'] = [['desconocido']] * len(df)
            df['genres_str'] = ['desconocido'] * len(df)
            df['combined_features'] = ['desconocido'] * len(df)
        
        # Para compatibilidad con la interfaz
        df['actors_processed'] = [['desconocido']] * len(df)
        df['director_processed'] = ['desconocido'] * len(df)
        
        logger.info(f"PelÃ­culas preparadas: {len(df)}")
        
        # Mostrar ejemplo para verificar
        if len(df) > 0:
            logger.info(f"Ejemplo de gÃ©neros: {df['genres_processed'].iloc[0]}")
        
        return df
    
    def _clean_title(self, title: str) -> str:
        """Limpia el tÃ­tulo."""
        if not isinstance(title, str):
            return "Desconocido"
        title = re.sub(r'\s*\(\d{4}\)\s*', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        return title if title else "Desconocido"
    
    def _extract_year(self, title: str) -> int:
        """Extrae el aÃ±o del tÃ­tulo."""
        if not isinstance(title, str):
            return 0
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0
    
    def _process_genres(self, genres_value) -> List[str]:
        """Procesa los gÃ©neros de forma robusta."""
        # Manejar None o NaN
        if genres_value is None or (isinstance(genres_value, float) and pd.isna(genres_value)):
            return ["desconocido"]
        
        # Convertir a string
        genres_str = str(genres_value)
        
        # Verificar si es el valor por defecto de MovieLens
        if genres_str == "nan" or genres_str == "(no genres listed)":
            return ["desconocido"]
        
        # Separar por pipe (formato MovieLens)
        if '|' in genres_str:
            genres = [g.strip().lower() for g in genres_str.split('|') if g.strip() and g.strip() != '(no genres listed)']
        else:
            genres = [genres_str.strip().lower()] if genres_str.strip() else []
        
        # Traducir gÃ©neros al espaÃ±ol para mejor visualizaciÃ³n
        genre_translation = {
            'action': 'AcciÃ³n',
            'adventure': 'Aventura',
            'animation': 'AnimaciÃ³n',
            'children': 'Infantil',
            'comedy': 'Comedia',
            'crime': 'Crimen',
            'documentary': 'Documental',
            'drama': 'Drama',
            'fantasy': 'FantasÃ­a',
            'film-noir': 'Cine Negro',
            'horror': 'Terror',
            'imax': 'IMAX',
            'musical': 'Musical',
            'mystery': 'Misterio',
            'romance': 'Romance',
            'sci-fi': 'Ciencia FicciÃ³n',
            'thriller': 'Suspenso',
            'war': 'Guerra',
            'western': 'Western'
        }
        
        # Traducir a espaÃ±ol si existe traducciÃ³n
        translated = [genre_translation.get(g, g.capitalize()) for g in genres]
        
        return translated if translated else ["desconocido"]
    
    def filter_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Filtra ratings por cantidad mÃ­nima."""
        if ratings_df.empty:
            return ratings_df
        
        # Asegurar nombres de columnas correctos
        if 'user_id' not in ratings_df.columns:
            if 'userId' in ratings_df.columns:
                ratings_df = ratings_df.rename(columns={'userId': 'user_id'})
            elif 'userid' in ratings_df.columns:
                ratings_df = ratings_df.rename(columns={'userid': 'user_id'})
        
        if 'movie_id' not in ratings_df.columns:
            if 'movieId' in ratings_df.columns:
                ratings_df = ratings_df.rename(columns={'movieId': 'movie_id'})
            elif 'movieid' in ratings_df.columns:
                ratings_df = ratings_df.rename(columns={'movieid': 'movie_id'})
        
        original_len = len(ratings_df)
        user_counts = ratings_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= self.min_ratings_per_user].index
        ratings_df = ratings_df[ratings_df['user_id'].isin(active_users)]
        
        movie_counts = ratings_df['movie_id'].value_counts()
        popular_movies = movie_counts[movie_counts >= self.min_ratings_per_movie].index
        ratings_df = ratings_df[ratings_df['movie_id'].isin(popular_movies)]
        
        logger.info(f"Ratings filtrados: {original_len} -> {len(ratings_df)}")
        return ratings_df
    
    def create_user_movie_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Crea matriz usuario-pelÃ­cula."""
        if ratings_df.empty:
            raise Exception("No hay datos")
        
        matrix = ratings_df.pivot(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
        logger.info(f"Matriz creada: {matrix.shape[0]} usuarios, {matrix.shape[1]} pelÃ­culas")
        return matrix
    
    def create_content_vectorizer(self, combined_features, max_features=5000):
        """Crea vectorizador."""
        self.vectorizer = CountVectorizer(max_features=max_features, lowercase=True)
        self.vectorizer.fit_transform(combined_features.fillna(''))
        return self.vectorizer
    
    def get_movie_index_mapping(self, movies_df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Crea mapeos de IDs a Ã­ndices."""
        movie_to_idx = {}
        idx_to_movie = {}
        
        for idx, row in movies_df.iterrows():
            movie_id = row.get('movie_id', idx)
            movie_to_idx[movie_id] = idx
            idx_to_movie[idx] = movie_id
        
        logger.info(f"Mapeos creados: {len(movie_to_idx)} pelÃ­culas")
        return movie_to_idx, idx_to_movie
    
    def get_user_index_mapping(self, user_movie_matrix: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Crea mapeos de usuarios a Ã­ndices."""
        users = user_movie_matrix.index.tolist()
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        idx_to_user = {idx: user for idx, user in enumerate(users)}
        logger.info(f"Mapeos de usuarios creados: {len(user_to_idx)} usuarios")
        return user_to_idx, idx_to_user

