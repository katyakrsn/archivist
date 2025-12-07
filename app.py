import os
from collections import Counter
from typing import Tuple, List, Optional, Any

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- CONSTANTS ---
PAGE_TITLE = "AI Film Archivist"
PAGE_ICON = "🎞️"
MODEL_RETRIEVER_NAME = 'all-MiniLM-L6-v2'
MODEL_GENERATOR_NAME = 'google/flan-t5-base'
DATA_PATHS = [
    'Dataset/movies.csv',
    'movies.csv',
    '/Users/ekaterina/Desktop/University/2 year/ML/Project/Dataset/movies.csv'
]

# --- PAGE CONFIGURATION & CUSTOM STYLING ---
def setup_page_config() -> None:
    """Configures the Streamlit page settings and custom CSS."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=PAGE_ICON)

    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        .stProgress > div > div > div > div {
            background-color: #667eea;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)


# --- CACHED RESOURCES ---
@st.cache_resource
def load_models() -> Tuple[SentenceTransformer, Any]:
    """Loads and caches the retrieval and generation models.

    Returns:
        Tuple[SentenceTransformer, Any]: A tuple containing:
            - retriever: The SentenceTransformer model for embeddings.
            - generator: The HuggingFace text2text-generation pipeline.

    Raises:
        RuntimeError: If models fail to load.
    """
    try:
        retriever = SentenceTransformer(MODEL_RETRIEVER_NAME)
        generator = pipeline('text2text-generation', model=MODEL_GENERATOR_NAME)
        return retriever, generator
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please check your internet connection for the first-time model download.")
        st.stop()


@st.cache_data
def load_data() -> pd.DataFrame:
    """Loads and cleans the movie dataset from supported paths.

    It attempts to find 'movies.csv' in a list of predefined paths.
    It processes dates, fills missing values, and creates an 'archival_text'
    column for embedding generation.

    Returns:
        pd.DataFrame: The processed dataframe containing movie data.

    Raises:
        FileNotFoundError: If the CSV file cannot be found in any path.
    """
    file_path = None
    for path in DATA_PATHS:
        if os.path.exists(path):
            file_path = path
            break

    if not file_path:
        st.error("Could not find 'movies.csv'. Please make sure it is in the Dataset folder.")
        st.stop()

    df = pd.read_csv(file_path)

    # Cleaning
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year.fillna(0).astype(int)
    df['director'] = df['director'].fillna("Unknown Director")

    # Create Archival Text
    df['archival_text'] = (
        df['title'] + " (" + df['year'].astype(str) + ")" +
        ": " + df['overview'].fillna('') +
        " Director: " + df['director'] +
        " Genres: " + df['genres'].fillna('')
    )
    return df


@st.cache_data
def generate_embeddings(_model: SentenceTransformer, text_list: List[str]) -> torch.Tensor:
    """Generates embeddings for a list of texts using the provided model.

    Args:
        _model (SentenceTransformer): The model used to encode text.
        text_list (List[str]): A list of strings to encode.

    Returns:
        torch.Tensor: A tensor containing the generated embeddings.
    """
    return _model.encode(text_list, convert_to_tensor=True)


# --- UI COMPONENTS ---
def render_sidebar(movies_data: pd.DataFrame, num_embeddings: int) -> None:
    """Renders the sidebar with archive statistics.

    Args:
        movies_data (pd.DataFrame): The dataframe containing movie info.
        num_embeddings (int): The count of pre-computed vector embeddings.
    """
    with st.sidebar:
        st.header("📊 Archive Statistics")
        st.metric("Total Films", len(movies_data))

        # Filter out 0 years for stats
        valid_years = movies_data[movies_data['year'] > 1900]['year']
        if not valid_years.empty:
            min_yr = int(valid_years.min())
            max_yr = int(valid_years.max())
            st.metric("Date Range", f"{min_yr} - {max_yr}")

        st.metric("Unique Directors", movies_data['director'].nunique())

        st.subheader("Top Directors")
        top_directors = movies_data['director'].value_counts().head(5)
        for director, count in top_directors.items():
            if director != "Unknown Director":
                st.write(f"- **{director}**: {count} films")

        st.divider()
        st.caption(f"⚡ {num_embeddings} vectors pre-computed")
        st.caption(f"🧠 Model: {MODEL_RETRIEVER_NAME} (384 dims)")

        with st.expander("ℹ️ About This Project"):
            st.markdown("""
            **Film Archivist** v1.0
            
            Created for: Machine Learning for Arts & Humanities
            
            Tech Stack:
            - **PyTorch:** Vector computations
            - **Transformers:** SBERT & Flan-T5
            - **Streamlit:** Interactive UI
            """)


def render_search_tab(
    retriever: SentenceTransformer,
    generator: Any,
    movies_data: pd.DataFrame,
    embeddings: torch.Tensor
) -> None:
    """Renders the 'Chat & Explore' tab functionality.

    Args:
        retriever (SentenceTransformer): The embedding model.
        generator (Any): The text generation pipeline.
        movies_data (pd.DataFrame): The movie dataset.
        embeddings (torch.Tensor): The pre-computed embeddings.
    """
    st.header("🔎 Find films by theme, plot, or mood")

    st.markdown("""
    **How to use:** This interface allows you to query the film collection using **natural language**.  
    Unlike traditional keyword search, you don't need exact titles. Instead, describe the **plot**, **atmospheric tone**, or **cultural theme** you wish to explore.
    """)
    st.divider()

    # --- Query Inputs ---
    col_q1, col_q2, col_q3 = st.columns(3)
    example_query = ""

    if col_q1.button("Space exploration"):
        example_query = "Films about lonely space exploration and isolation"
    if col_q2.button("Anti-technology dystopias"):
        example_query = "Films about anti-technology dystopias and societal collapse"
    if col_q3.button("Post-apocalyptic futures"):
        example_query = "Post-apocalyptic futures and survival in wasteland"

    query = st.text_input(
        "Try it yourself:",
        value=example_query,
        placeholder="e.g., romantic comedies set in a snowy town during christmas"
    )

    # --- Filters ---
    with st.expander("⚙️ Advanced Options"):
        col_f1, col_f2 = st.columns(2)
        top_k = col_f1.slider("Number of films to retrieve", 3, 10, 5)

        valid_years = movies_data[movies_data['year'] > 1900]['year']
        min_db_year = int(valid_years.min())
        max_db_year = int(valid_years.max())
        year_range = col_f2.slider(
            "Filter by Year",
            min_db_year,
            max_db_year,
            (min_db_year, max_db_year)
        )

    # --- Execution ---
    if query:
        _execute_search(
            query, top_k, year_range, retriever, generator, movies_data, embeddings
        )


def _execute_search(
    query: str,
    top_k: int,
    year_range: Tuple[int, int],
    retriever: SentenceTransformer,
    generator: Any,
    movies_data: pd.DataFrame,
    embeddings: torch.Tensor
) -> None:
    """Internal helper to execute search logic and display results."""
    # 1. RETRIEVAL
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]

    # Apply year filter
    valid_indices = movies_data[
        (movies_data['year'] >= year_range[0]) &
        (movies_data['year'] <= year_range[1])
    ].index.tolist()

    if not valid_indices:
        st.warning("No films found in this year range.")
        return

    # Sort and Select
    valid_scores = [(i, cos_scores[i].item()) for i in valid_indices]
    valid_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = valid_scores[:top_k]

    retrieved_movies = []
    
    st.subheader("🔎 Retrieved Source Material")

    for idx, score in top_results:
        movie = movies_data.iloc[idx]
        confidence = score * 100
        
        with st.expander(f"{movie['title']} ({movie['year']}) - {confidence:.1f}% Match"):
            st.write(f"**Director:** {movie['director']}")
            st.write(f"**Plot:** {movie['overview']}")
            st.progress(score)
            st.caption(f"Cosine Similarity Score: {score:.4f}")
        
        retrieved_movies.append(movie)

    if not retrieved_movies:
        st.warning(f"No movies found matching theme '{query}'.")
        return

    # 2. GENERATION (Analysis)
    st.subheader("🤖 Archivist's Analysis")
    top_movie = movies_data.iloc[top_results[0][0]]

    st.markdown(f"### 🎬 {top_movie['title']} ({top_movie['year']})")
    st.markdown(f"**🎥 Director:** {top_movie['director']}")
    st.markdown(f"**📊 Relevance Score:** {top_results[0][1] * 100:.1f}%")

    prompt = (
        f"Movie: {top_movie['title']}\n"
        f"Plot: {str(top_movie['overview'])[:400]}\n\n"
        f"User Query: {query}\n\n"
        f"Explain why this movie matches the user's query in one clear sentence."
    )

    with st.spinner("Generating analysis..."):
        response = generator(
            prompt,
            max_new_tokens=100,
            min_length=20,
            repetition_penalty=2.0,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )
        clean_response = response[0]['generated_text'].replace(prompt, "").strip()
        st.success(f"**💡 Insight:** {clean_response}")

        with st.expander("📖 Read Full Plot Summary"):
            st.write(top_movie['overview'])

    # 3. EXPORT
    st.markdown("---")
    export_data = pd.DataFrame([{
        'Title': m['title'],
        'Year': m['year'],
        'Director': m['director'],
        'Overview': m['overview']
    } for m in retrieved_movies])

    csv = export_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Recommendations as CSV",
        data=csv,
        file_name="recommendations.csv",
        mime="text/csv"
    )
    
    # Educational Note
    with st.expander("⚙️ Why Simple Recommendations?"):
        st.markdown("""
        **Model Constraint Demonstration:**
        This project uses **Flan-T5-base** (220M parameters) to demonstrate 
        RAG architecture within resource constraints typical of humanities computing.
        """)


def render_visuals_tab(movies_data: pd.DataFrame, embeddings: torch.Tensor) -> None:
    """Renders the 'Cultural Map' visualization tab.

    Args:
        movies_data (pd.DataFrame): The movie dataset.
        embeddings (torch.Tensor): The pre-computed embeddings.
    """
    st.header("Visualizing the Archive")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🌌 Semantic Clustering (PCA)")
        st.markdown("This map reduces 384 dimensions of meaning into 2 dimensions.")

        if st.button("Generate Cultural Map"):
            with st.spinner("Calculating mathematical projection..."):
                cpu_embeddings = embeddings.cpu().numpy()
                pca = PCA(n_components=2)
                vis_dims = pca.fit_transform(cpu_embeddings)

                n_clusters = 8
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(cpu_embeddings)

                vis_df = pd.DataFrame({
                    'x': vis_dims[:, 0],
                    'y': vis_dims[:, 1],
                    'Title': movies_data['title'],
                    'Director': movies_data['director'],
                    'Cluster': clusters.astype(str)
                })

                fig = px.scatter(
                    vis_df, x='x', y='y', color='Cluster',
                    hover_data=['Title', 'Director'],
                    template="plotly_dark", height=500
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Temporal Evolution")
        st.markdown("Explore how themes evolved across decades.")
        
        decade = st.selectbox(
            "Select Decade", ["1970s", "1980s", "1990s", "2000s", "2010s"]
        )
        decade_start = int(decade[:4])
        
        decade_data = movies_data[
            (movies_data['year'] >= decade_start) & 
            (movies_data['year'] < decade_start + 10)
        ]
        
        if len(decade_data) > 0:
            st.metric(f"Films in the {decade}", len(decade_data))
            
            st.write("**Dominant Genres:**")
            all_genres = []
            for genres_str in decade_data['genres'].dropna():
                all_genres.extend(genres_str.split())
            
            top_genres = Counter(all_genres).most_common(5)
            genre_df = pd.DataFrame(top_genres, columns=["Genre", "Count"])
            st.dataframe(genre_df, hide_index=True)
        else:
            st.write("No data for this decade.")


def render_methodology_tab() -> None:
    """Renders the 'Methodology' tab with educational content."""
    st.header("Methodology & System Architecture")
    
    # Arts & Humanities Context
    st.subheader("🎨 Arts & Humanities Relevance")
    col_prob, col_sol = st.columns(2)
    
    with col_prob:
        st.markdown("**❌ The Problem**")
        st.markdown("""
        Traditional databases rely on **keyword matching**, missing thematic
        connections and requiring manual metadata tagging.
        """)
        
    with col_sol:
        st.markdown("**✅ Our Solution**")
        st.markdown("""
        **Semantic search** using neural embeddings understands meaning, 
        discovering thematic connections automatically.
        """)
    
    st.divider()
    
    # Technical Architecture
    st.subheader("🔧 Technical Architecture")
    with st.expander("📐 System Diagram", expanded=True):
        st.code("""
        USER QUERY -> ENCODING (Sentence-BERT) -> RETRIEVAL (Cosine Similarity) 
        -> CONTEXT ASSEMBLY -> GENERATION (Flan-T5) -> OUTPUT
        """)
    
    with st.expander("🧮 Mathematical Foundation"):
        st.markdown("Calculates **Cosine Similarity** between query and movie vectors.")


def render_evaluation_tab(
    retriever: SentenceTransformer,
    movies_data: pd.DataFrame,
    embeddings: torch.Tensor
) -> None:
    """Renders the 'Evaluation' tab to test system precision.

    Args:
        retriever (SentenceTransformer): The embedding model.
        movies_data (pd.DataFrame): The movie dataset.
        embeddings (torch.Tensor): The pre-computed embeddings.
    """
    st.header("🎯 System Evaluation")
    
    st.markdown("### Understanding the Evaluation Methodology")
    st.markdown("We use **Precision@K** to measure how many top results are relevant.")

    test_cases = [
        {
            "Theme": "Technology Critique",
            "Query": "AI rebellion and machines taking over",
            "Expected": ["Matrix", "Terminator", "Ex Machina", "I, Robot"]
        },
        {
            "Theme": "Space Isolation",
            "Query": "lonely space exploration and isolation",
            "Expected": ["Interstellar", "Gravity", "Moon", "Solaris", "Martian"]
        },
        {
            "Theme": "Fantasy Epic",
            "Query": "wizards and rings and hobbits",
            "Expected": ["Hobbit", "Lord of the Rings", "Fellowship"]
        }
    ]

    st.markdown("**Test Cases:**")
    st.dataframe(pd.DataFrame(test_cases)[['Theme', 'Query']], hide_index=True)

    if st.button("▶️ Run Precision@5 Evaluation", type="primary"):
        _run_evaluation_logic(test_cases, retriever, movies_data, embeddings)


def _run_evaluation_logic(
    test_cases: List[dict],
    retriever: SentenceTransformer,
    movies_data: pd.DataFrame,
    embeddings: torch.Tensor
) -> None:
    """Helper function to execute the evaluation loop."""
    results = []
    progress_bar = st.progress(0)
    
    for i, test in enumerate(test_cases):
        # Run Search
        q_emb = retriever.encode(test["Query"], convert_to_tensor=True)
        scores = util.cos_sim(q_emb, embeddings)[0]
        top_5_indices = torch.topk(scores, k=5).indices
        
        retrieved_titles = [movies_data.iloc[int(idx)]['title'] for idx in top_5_indices]
        
        # Check matches (fuzzy)
        hits = 0
        matched_titles = []
        for title in retrieved_titles:
            is_match = any(exp.lower() in title.lower() for exp in test["Expected"])
            if is_match:
                hits += 1
                matched_titles.append(title)
        
        precision = hits / 5.0
        results.append({
            "Theme": test["Theme"],
            "Query": test["Query"],
            "Precision@5": f"{precision:.0%}",
            "Relevant Found": ", ".join(matched_titles) if matched_titles else "None"
        })
        
        progress_bar.progress((i + 1) / len(test_cases))
    
    st.markdown("### 📈 Results")
    st.dataframe(pd.DataFrame(results), use_container_width=True)


# --- MAIN EXECUTION ---
def main() -> None:
    """Main entry point for the Streamlit application."""
    setup_page_config()

    st.markdown('<p class="main-header">Ask the Film Archivist</p>', unsafe_allow_html=True)

    with st.spinner("Initializing Digital Archive & Neural Models..."):
        retriever, generator = load_models()
        movies_data = load_data()
        embeddings = generate_embeddings(retriever, movies_data['archival_text'].tolist())

    render_sidebar(movies_data, len(embeddings))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat & Explore",
        "🌌 Cultural Map",
        "📊 Methodology",
        "🎯 Evaluation"
    ])

    with tab1:
        render_search_tab(retriever, generator, movies_data, embeddings)
    with tab2:
        render_visuals_tab(movies_data, embeddings)
    with tab3:
        render_methodology_tab()
    with tab4:
        render_evaluation_tab(retriever, movies_data, embeddings)


if __name__ == "__main__":
    main()