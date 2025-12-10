"""
app.py
======
Main Entry Point
----------------
This script coordinates the application. It:
1. Initializes the UI configuration (via frontend).
2. Loads models and data (via backend).
3. Renders the main interface components.
"""

import streamlit as st
import backend   # Imports logic from backend.py
import frontend  # Imports UI from frontend.py


def main() -> None:
    """Main execution function for the Streamlit application."""
    
    # 1. SETUP UI CONFIGURATION
    # Must be the first Streamlit command
    frontend.setup_page_config()

    st.markdown('<p class="main-header">Ask the Film Archivist</p>', unsafe_allow_html=True)

    # 2. LOAD BACKEND RESOURCES
    # load heavy models/data here so they are available to all tabs
    with st.spinner("Initializing Digital Archive & Neural Models..."):
        retriever, generator = backend.load_models()
        movies_data = backend.load_data()
        
        # Pre-compute embeddings for the entire dataset
        embeddings = backend.generate_embeddings(retriever, movies_data['archival_text'].tolist())

    # 3. RENDER SIDEBAR
    # Pass the data needed for statistics
    frontend.render_sidebar(movies_data, len(embeddings))

    # 4. RENDER MAIN TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat & Explore", 
        "🌌 Cultural Map", 
        "📊 Methodology", 
        "🎯 Evaluation"
    ])

    # --- TAB 1: Search ---
    with tab1:
        frontend.render_search_tab(retriever, generator, movies_data, embeddings)

    # --- TAB 2: Visualization ---
    with tab2:
        frontend.render_visuals_tab(movies_data, embeddings)

    # --- TAB 3: Methodology ---
    with tab3:
        frontend.render_methodology_tab()

    # --- TAB 4: Evaluation ---
    with tab4:
        frontend.render_evaluation_tab(retriever, movies_data, embeddings)


if __name__ == "__main__":
    main()