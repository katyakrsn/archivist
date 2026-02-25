"""
backend.py
==========
This module handles the logic of the Film Archivist.
It includes:
1. Data ingestion and cleaning.
2. Model loading and caching (Sentence-BERT, Flan-T5).
3. Vector embedding generation.
4. Mathematical operations for clustering (PCA, KMeans).
"""

import os
from typing import Tuple, List, Any
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st

# --- CONSTANTS ---
MODEL_RETRIEVER_NAME = 'all-MiniLM-L6-v2'
MODEL_GENERATOR_NAME = 'google/flan-t5-base'

# Robust path handling to find your file wherever you run it
DATA_PATHS = [
    'Dataset/movies.csv',
    'movies.csv',
    '/Users/ekaterina/Desktop/University/2 year/ML/Project/Dataset/movies.csv',
    os.path.join(os.path.dirname(__file__), 'movies.csv')
]


@st.cache_resource
def load_models() -> Tuple[SentenceTransformer, Any]:
    """Loads and caches the neural network models.

    Returns:
        Tuple[SentenceTransformer, Any]: A tuple containing:
            - retriever: The SentenceTransformer model for semantic encoding.
            - generator: The HuggingFace text-generation pipeline.

    Raises:
        RuntimeError: If models fail to download or load.
    """
    try:
        # 1. Load Retriever (BERT)
        retriever = SentenceTransformer(MODEL_RETRIEVER_NAME)
        # 2. Load Generator (Flan-T5)
        generator = pipeline('text-generation', model=MODEL_GENERATOR_NAME)
        return retriever, generator
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please check your internet connection for the first-time model download.")
        st.stop()


@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads and cleans the movie dataset.

    It scans the `DATA_PATHS` list to find the 'movies.csv' file.
    It performs necessary data cleaning and feature engineering (creating 'archival_text').

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    file_path = None
    for path in DATA_PATHS:
        if os.path.exists(path):
            file_path = path
            break

    if not file_path:
        st.error("Could not find 'movies.csv'. Please make sure it is in the application folder.")
        st.stop()

    df = pd.read_csv(file_path)

    # Cleaning
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year.fillna(0).astype(int)
    df['director'] = df['director'].fillna("Unknown Director")

    # Create Archival Text (The text that will be embedded)
    df['archival_text'] = (
        df['title'] + " (" + df['year'].astype(str) + ")" +
        ": " + df['overview'].fillna('') +
        " Director: " + df['director'] +
        " Genres: " + df['genres'].fillna('')
    )
    return df


@st.cache_data
def generate_embeddings(_model: SentenceTransformer, text_list: List[str]) -> torch.Tensor:
    """Generates vector embeddings for a list of texts.

    Args:
        _model (SentenceTransformer): The loaded embedding model.
        text_list (List[str]): The list of strings to encode.

    Returns:
        torch.Tensor: A tensor of shape (N, 384) containing the embeddings.
    """
    return _model.encode(text_list, convert_to_tensor=True)


def compute_clusters(embeddings: torch.Tensor, n_clusters: int = 8) -> Tuple[Any, Any]:
    """Performs dimensionality reduction (PCA) and clustering (KMeans).

    Args:
        embeddings (torch.Tensor): The high-dimensional embeddings.
        n_clusters (int, optional): Number of clusters to find. Defaults to 8.

    Returns:
        Tuple[Any, Any]:
            - vis_dims: 2D array of reduced coordinates (x, y).
            - clusters: Array of cluster labels for each point.
    """
    # Move to CPU for sklearn
    cpu_embeddings = embeddings.cpu().numpy()

    # 1. Reduce dimensions for visualization
    pca = PCA(n_components=2)
    vis_dims = pca.fit_transform(cpu_embeddings)

    # 2. Cluster semantically similar films
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(cpu_embeddings)

    return vis_dims, clusters
