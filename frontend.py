"""
frontend.py
===========
This module handles the User Interface (UI) rendering for the application.
It imports logic from `backend.py` when necessary but focuses on:
1. Page configuration and styling.
2. Rendering sidebar and tabs.
3. Visualizing data (Plotly charts).
4. Displaying educational text (Methodology, Evaluation).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import torch
from sentence_transformers import util
from collections import Counter
from typing import Tuple, List, Any

# Import the backend module to access logic if needed directly
import backend


# --- CONFIGURATION ---
def setup_page_config() -> None:
    """Sets up the Streamlit page configuration and custom CSS."""
    st.set_page_config(page_title="AI Film Archivist", layout="wide", page_icon="🎞️")

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


# --- SIDEBAR ---
def render_sidebar(movies_data: pd.DataFrame, num_embeddings: int) -> None:
    """Renders the sidebar containing archive statistics and project info."""
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
        st.caption(f"🧠 Model: all-MiniLM-L6-v2 (384 dims)")

        with st.expander("ℹ️ About This Project"):
            st.markdown("""
            **Film Archivist** v1.0
            
            Created for: Machine Learning for Arts & Humanities
            
            Tech Stack:
            - **PyTorch:** Vector computations
            - **Transformers:** SBERT & Flan-T5
            - **Streamlit:** Interactive UI
            """)


# --- TAB 1: SEARCH & EXPLORE ---
def render_search_tab(
    retriever: Any,
    generator: Any,
    movies_data: pd.DataFrame,
    embeddings: torch.Tensor
) -> None:
    """Renders the semantic search interface."""
    st.header("🔎 Find films by theme, plot, or mood")

    st.markdown("""
    **How to use:** This interface allows you to query the film collection using **natural language**.  
    Unlike traditional keyword search, you don't need exact titles. Instead, describe the **plot**, **atmospheric tone**, or **cultural theme** you wish to explore.
    
    The system uses vector embeddings to understand the *semantic meaning* of your request.
    """)
    st.divider()

    # Example Query Buttons
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

    # Advanced Filters
    with st.expander("⚙️ Advanced Options"):
        col_f1, col_f2 = st.columns(2)
        top_k = col_f1.slider("Number of films to retrieve", 3, 10, 5)

        valid_years = movies_data[movies_data['year'] > 1900]['year']
        min_db_year, max_db_year = int(valid_years.min()), int(valid_years.max())
        year_range = col_f2.slider("Filter by Year", min_db_year, max_db_year, (min_db_year, max_db_year))

    if query:
        # 1. RETRIEVAL (Calculation)
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
            st.warning(f"No movies found matching the theme '{query}'.")
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
            
            **Trade-off Analysis:**
            
            | Aspect | Small Model (Choice) | Large Model (GPT-4) |
            |--------|-------------------------|-------------------|
            | Cost | Free, open-source | $0.03 per 1K tokens |
            | Speed | ~1 second | ~3-5 seconds |
            | Hardware | Runs on CPU | Requires API/GPU |
            | Output Quality | Concise, factual | Elaborate, creative |
            
            **Academic Value:** This demonstrates how RAG's retrieval component 
            provides value even with minimal generation capabilities—the factual 
            grounding matters more than prose eloquence for archival discovery.
            """)


# --- TAB 2: VISUALIZATION ---
def render_visuals_tab(movies_data: pd.DataFrame, embeddings: torch.Tensor) -> None:
    """Renders the semantic clustering map and temporal stats."""
    st.header("Visualizing the Archive")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌌 Semantic Clustering (PCA)")
        st.markdown("This map reduces 384 dimensions of meaning into 2 dimensions. Colors represent automatically discovered 'hidden genres'.")
        
        if st.button("Generate Cultural Map"):
            with st.spinner("Calculating mathematical projection..."):
                # Call backend for the heavy math
                vis_dims, clusters = backend.compute_clusters(embeddings, n_clusters=8)
                
                vis_df = pd.DataFrame({
                    'x': vis_dims[:, 0], 'y': vis_dims[:, 1],
                    'Title': movies_data['title'], 'Director': movies_data['director'],
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
        
        decade = st.selectbox("Select Decade", ["1970s", "1980s", "1990s", "2000s", "2010s"])
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


# --- TAB 3: METHODOLOGY ---
def render_methodology_tab() -> None:
    """Renders the educational content explaining the system architecture."""
    st.header("Methodology & System Architecture")
    
    # Arts & Humanities Context
    st.subheader("🎨 Arts & Humanities Relevance")
    
    col_prob, col_sol = st.columns(2)
    
    with col_prob:
        st.markdown("**❌ The Problem**")
        st.markdown("""
        Traditional film databases rely on **keyword matching**:
        - Search "Cold War paranoia" → zero results (unless explicitly tagged)
        - Search "Existential crisis" → misses thematically relevant films
        - Requires manual metadata tagging by archivists
        - Cannot understand conceptual relationships
        """)
        
    with col_sol:
        st.markdown("**✅ The Solution**")
        st.markdown("""
        **Semantic search** using neural embeddings:
        - Understands *meaning*, not just words
        - "AI rebellion" → finds *Terminator*, *Matrix*, *Ex Machina*
        - Works across languages and phrasings
        - Discovers thematic connections automatically
        """)
    
    st.info("**Real-World Application:** Film scholars can now ask: *'Show me 1950s films reflecting post-war trauma'* and receive semantically relevant results without manual tagging.")
    
    st.divider()
    
    # Technical Architecture
    st.subheader("🔧 Technical Architecture")
    
    with st.expander("📐 System Diagram", expanded=True):
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────┐
        │                    USER QUERY                           │
        │        "Films about loneliness in space"                │
        └────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │  STEP 1: ENCODING (Sentence-BERT)                      │
        │  • Converts text → 384-dimensional vector              │
        │  • Captures semantic meaning in numbers                │
        │  Output: [0.23, -0.45, 0.67, ..., 0.12]               │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │  STEP 2: RETRIEVAL (Cosine Similarity)                │
        │  • Compare query vector to 4,800 movie vectors         │
        │  • Formula: similarity = (A·B) / (||A|| × ||B||)       │
        │  • Returns Top-K most similar films                    │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │  STEP 3: CONTEXT ASSEMBLY                              │
        │  Retrieved: Interstellar (0.78), Gravity (0.72)...    │
        │  Creates text: "Movie 1: Interstellar - An astronaut  │
        │  travels through a wormhole..."                        │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │  STEP 4: GENERATION (Flan-T5)                         │
        │  Prompt: "Given these movies: [context], recommend..." │
        │  AI generates natural language explanation             │
        └────────────────────┬───────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────────────┐
        │  OUTPUT: "I recommend Interstellar because it         │
        │  explores isolation and human connection across       │
        │  vast cosmic distances..."                            │
        └────────────────────────────────────────────────────────┘
        ```
        """)
    
    with st.expander("🧮 Mathematical Foundation"):
        st.markdown("""
        **Cosine Similarity Explained:**
        
        Given two vectors A (query) and B (movie):
        
        $$
        \\text{similarity} = \\cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|} = \\frac{\\sum_{i=1}^{384} A_i B_i}{\\sqrt{\\sum A_i^2} \\times \\sqrt{\\sum B_i^2}}
        $$
        
        **Interpretation:**
        - Score = 1.0 → Identical meaning
        - Score = 0.7-0.9 → Strong thematic similarity
        - Score = 0.5-0.7 → Moderate relevance
        - Score < 0.5 → Weak connection
        
        **Why this matters:** Unlike keyword matching (binary: match or no match), 
        cosine similarity gives us a *continuous relevance score*.
        """)
        
    with st.expander("🔗 Connection to Course Topics"):
        st.markdown("""
        | Course Topic | Where It Appears in This Project |
        |--------------|----------------------------------|
        | **PyTorch** | • Tensor operations for embeddings • GPU acceleration for batch encoding • Efficient matrix operations |
        | **Language Processing** | • Sentence-BERT transformer architecture • Flan-T5 sequence-to-sequence generation • Tokenization and attention mechanisms |
        | **RAG (Retrieval-Augmented Generation)** | • Complete RAG pipeline implementation • Retrieval prevents hallucination • Grounding generation in factual data |
        | **Linear Algebra** | • Cosine similarity calculations • Vector space operations • Dimensionality reduction (PCA in Tab 2) |
        | **Machine Vision Concepts** | • Embedding space visualization • Clustering in high-dimensional space • Feature extraction principles |
        """)


# --- TAB 4: EVALUATION ---
def render_evaluation_tab(
    retriever: Any,
    movies_data: pd.DataFrame,
    embeddings: torch.Tensor
) -> None:
    """Renders the precision evaluation interface and results."""
    st.header("🎯 System Evaluation")
    
    # Explanation
    st.markdown("""
    ### Understanding the Evaluation Methodology
    
    **Why do we need evaluation?** In Machine Learning, we cannot just *assume* the system works - we must **measure** its performance objectively.
    """)
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("**📚 What is Precision@K?**")
        st.markdown("""
        Precision@K measures: *"Of the top K results returned, how many are actually relevant?"*
        
        **Formula:**
        $$
        \\text{Precision@K} = \\frac{\\text{Number of Relevant Results in Top K}}{K}
        $$
        
        **Example:** Query: "AI rebellion movies"  
        Top 5 Results: Matrix ✅, Terminator ✅, Titanic ❌, Ex Machina ✅, Avatar ❌
        
        Precision@5 = 3/5 = **60%**
        """)
    
    with col_exp2:
        st.markdown("**🎓 Why This Metric?**")
        st.markdown("""
        - **User-Focused:** People rarely look past the top 5 results
        - **Binary & Clear:** Either relevant or not (no ambiguity)
        - **Standard in IR:** Used in Google Search, recommendation systems
        - **Easy to Interpret:** 80% = 4 out of 5 results are good
        
        **Note:** The system uses "fuzzy matching" (e.g., "Terminator 2" counts as "Terminator") 
        because exact title matching is too strict.
        """)
    
    st.divider()
    
    # Test Design
    st.subheader("🧪 Test Case Design")
    
    st.markdown("""
    I designed **3 diverse test queries** representing different search scenarios:
    
    1. **Technology Critique:** Tests understanding of thematic concepts (dystopia, AI ethics)
    2. **Space Isolation:** Tests emotional/atmospheric understanding (loneliness, existential themes)  
    3. **Fantasy Epic:** Tests genre and narrative structure recognition (quest narratives, magical worlds)
    
    For each query, I defined **"Ground Truth"** - movies I *know* should appear based on human judgment.
    """)
    
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
    
    # Show test cases in a nice table
    st.markdown("**Our Test Cases:**")
    test_cases_display = pd.DataFrame([
        {
            "Theme": t["Theme"],
            "Query": t["Query"],
            "Expected Films": ", ".join(t["Expected"][:3]) + "..."
        } for t in test_cases
    ])
    st.dataframe(test_cases_display, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Run Evaluation
    if st.button("▶️ Run Precision@5 Evaluation", type="primary"):
        _run_evaluation_logic(test_cases, retriever, movies_data, embeddings)
        
    st.divider()
    
    # Comparisons
    st.subheader("📊 How Does This Compare to Research Baselines?")
    
    st.markdown("""
    This system uses **dense retrieval** (BERT embeddings) rather than traditional sparse retrieval (keyword/TF-IDF). 
    Research shows this architectural choice significantly impacts performance:
    """)
    
    comparison_data = pd.DataFrame([
        {
            "Approach": "Keyword Search",
            "Method": "Exact string matching",
            "Key Finding": "Traditional baseline",
            "Reference": "Manning et al. (2008)"
        },
        {
            "Approach": "TF-IDF (Sparse)",
            "Method": "Term frequency weighting",
            "Key Finding": "Classical NLP baseline for content-based recommendations",
            "Reference": "Lops et al. (2011)"
        },
        {
            "Approach": "Dense Retrieval (BERT)",
            "Method": "Semantic embeddings",
            "Key Finding": "9-15% improvement over sparse retrieval (BM25)",
            "Reference": "Karpukhin et al. (2020)"
        },
        {
            "Approach": "RAG (This System)",
            "Method": "Retrieval + Generation",
            "Key Finding": "Reduces hallucination, improves factual accuracy",
            "Reference": "Lewis et al. (2020)"
        }
    ])
    
    st.dataframe(comparison_data, use_container_width=True, hide_index=True)
    
    # References
    st.markdown("### 📚 References")
    
    with st.expander("Click to view full citations with links"):
        st.markdown("""
        **Core Papers Used in This Project:**
        
        1. **Sentence-BERT** (The Retrieval Model)  
           Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* 📄 [Read on arXiv](https://arxiv.org/abs/1908.10084) | [Hugging Face Model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
        
        2. **Dense Passage Retrieval** (Justification for BERT over TF-IDF)  
           Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* 📄 [Read on arXiv](https://arxiv.org/abs/2004.04906)
        
        3. **RAG: Retrieval-Augmented Generation** (The Core Architecture)  
           Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* 📄 [Read on arXiv](https://arxiv.org/abs/2005.11401)
        
        4. **Flan-T5** (The Generation Model)  
           Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models.* 📄 [Read on arXiv](https://arxiv.org/abs/2210.11416) | [Hugging Face Model](https://huggingface.co/google/flan-t5-base)
        
        **Background & Baselines:**
        
        5. **Content-Based Recommender Systems** (Traditional Approach)  
           Lops, P., de Gemmis, M., & Semeraro, G. (2011). *Content-based Recommender Systems: State of the Art and Trends.* 📖 In Recommender Systems Handbook, Springer. [Google Scholar](https://scholar.google.com/scholar?q=Content-based+Recommender+Systems+Lops)
        
        6. **Information Retrieval Fundamentals** Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* 📖 Cambridge University Press. [Free Online Version](https://nlp.stanford.edu/IR-book/)
        
        7. **Deep Learning for Recommendations** (Survey)  
           Zhang, S., et al. (2019). *Deep Learning Based Recommender System: A Survey and New Perspectives.* 📄 [Read on arXiv](https://arxiv.org/abs/1707.07435)
        """)
    
    # Limitations
    st.subheader("⚠️ Evaluation Limitations")
    
    with st.expander("Click to see limitations of this evaluation"):
        st.markdown("""
        **1. Small Test Set** Only 3 queries not statistically significant. A robust evaluation would need 50+ diverse queries.
        
        **2. Subjective Ground Truth** "Relevant" is defined by one person (me). Different users might have different expectations.
        
        **3. Fuzzy Matching Bias** The system counts "The Terminator" as matching "Terminator", but what about similar films 
        with different titles? (e.g., "Blade Runner" for AI themes)
        
        **4. No Ranking Quality** Precision@5 only checks *if* relevant films appear, not *where* they rank. 
        Ideally, the most relevant should be #1.
        
        **5. Binary Relevance** Real relevance is a spectrum (highly relevant vs. somewhat relevant), 
        but it is treated here as yes/no.
        
        **Better Metrics for Future Work:**
        - Mean Average Precision (MAP)
        - Normalized Discounted Cumulative Gain (NDCG)
        - User satisfaction surveys
        """)


def _run_evaluation_logic(
    test_cases: List[dict],
    retriever: Any,
    movies_data: pd.DataFrame,
    embeddings: torch.Tensor
) -> None:
    """Helper function to execute the evaluation loop and display results."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, test in enumerate(test_cases):
        status_text.text(f"Testing: {test['Theme']}...")
        
        # Run Search
        q_emb = retriever.encode(test["Query"], convert_to_tensor=True)
        scores = util.cos_sim(q_emb, embeddings)[0]
        top_5_indices = torch.topk(scores, k=5).indices
        
        retrieved_titles = [movies_data.iloc[int(idx)]['title'] for idx in top_5_indices]
        
        # Check for matches (fuzzy matching)
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
            "Hits": f"{hits}/5",
            "Top Matches": ", ".join(retrieved_titles[:3]),
            "Relevant Found": ", ".join(matched_titles) if matched_titles else "None"
        })
        
        progress_bar.progress((i + 1) / len(test_cases))
    
    status_text.text("✅ Evaluation Complete!")
    
    # Display Results Table
    st.markdown("### 📈 Results")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Calculate and Display Average
    avg_precision = np.mean([float(r["Precision@5"].strip('%'))/100 for r in results])
    
    col_metric1, col_metric2, col_metric3 = st.columns(3)
    col_metric1.metric("Average Precision@5", f"{avg_precision:.1%}")
    col_metric2.metric("Total Queries Tested", len(test_cases))
    col_metric3.metric("Total Films Retrieved", len(test_cases) * 5)
    
    # Interpretation
    st.markdown("### 🔍 Interpretation")
    
    if avg_precision > 0.6:
        st.success("""
        ✅ **Strong Performance (>60%)** The system demonstrates robust semantic understanding. 
        More than half of retrieved results are thematically relevant.
        """)
    elif avg_precision > 0.4:
        st.info("""
        ℹ️ **Moderate Performance (40-60%)** The system shows semantic understanding but inconsistent precision.
        """)
    else:
        st.warning("""
        ⚠️ **Needs Improvement (<40%)** Low precision suggests the embedding space doesn't capture film semantics well.
        """)
    
    # Show detailed breakdown for each query
    with st.expander("📋 Detailed Breakdown by Query"):
        for i, result in enumerate(results):
            st.markdown(f"**Query {i+1}: {result['Theme']}**")
            st.write(f"- Search: *\"{result['Query']}\"*")
            st.write(f"- Precision: **{result['Precision@5']}**")
            st.write(f"- Relevant films found: {result['Relevant Found']}")
            st.write(f"- Top 3 retrieved: {result['Top Matches']}")
            st.divider()