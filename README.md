# 🎞️ Semantic Film Archivist

**A Retrieval-Augmented Generation (RAG) system for the film discovery.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## 📖 Overview

The **Semantic Film Archivist** is a Machine Learning application designed to bridge the gap between traditional metadata search and conceptual understanding in Digital Humanities.

Unlike standard databases that rely on keyword matching (e.g., searching for "war" to find war movies), this tool utilizes **Semantic Embeddings** to understand the *meaning* of a query. Users can describe a plot, an atmospheric tone, or a complex feeling (e.g., movies about the isolation of space travel), and the system retrieves thematically relevant films even if they don't share specific keywords.

## 🎓 Academic Context

This project was developed as the final examination for the **Machine Learning for Arts and Humanities** course at the **University of Bologna**.

* **Program:** Master's in Digital Humanities and Digital Knowledge (DHDK)
* **Academic Year:** 2024/2025
* **Student:** Ekaterina Krasnova
* **Course Repository:** [UNIBO_MachineLearning by Prof. Giovanni1085](https://github.com/Giovanni1085/UNIBO_MachineLearning)

## 📂 Project Structure

This repository contains the following key files:

| File | Description |
| :--- | :--- |
| **`app.py`** | **Main Application.** Run this file to launch the Semantic Search interface, RAG Chatbot, and Visualization dashboard. |
| **`requirements.txt`** | List of all Python dependencies required to run the project (PyTorch, Transformers, Streamlit, etc.). |
| **`Dataset/movies.csv`** |  Movie Dataset used for the archive. Contains metadata for ~4,800 films. |
| **`Semantic Film Archivist.docx`** | The official project report detailing methodology, evaluation metrics (Precision@5), and architectural decisions. |
| *`Movie Recommendation System.py`* | *Draft script.* Included for archival purposes to show the development process from simple recommendations to the final RAG application. |

## 🚀 Key Features

1.  **💬 Semantic Search & RAG Chatbot:** Ask the archivist questions in natural language. The system retrieves relevant documents and uses a Generative AI (`Flan-T5`) to synthesize an answer.
2.  **🌌 Cultural Map (Visualization):** An interactive 2D scatter plot using **Principal Component Analysis (PCA)** to visualize how 4,800 movies cluster together based on narrative similarity.
3.  **🎯 Automated Evaluation:** A built-in testing suite that calculates **Precision@5** to quantitatively measure the retrieval accuracy against ground-truth thematic queries.
4.  **📅 Temporal Analysis:** Tools to filter and analyze the dominance of specific genres across different decades of cinema history.

## 🛠️ Tech Stack

This project implements a full ML pipeline aligned with the course syllabus:

* **Deep Learning:** [PyTorch](https://pytorch.org/) (Tensor computations)
* **NLP & Transformers:** `sentence-transformers` (SBERT embeddings) & `transformers` (Flan-T5 generation)
* **Unsupervised Learning:** Scikit-Learn (K-Means Clustering & PCA)
* **Visualization:** Plotly Express
* **Frontend:** [Streamlit](https://streamlit.io/)

## ⚙️ Installation & Usage

You can run this project locally on your machine.

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR-USERNAME/semantic-film-archivist.git](https://github.com/YOUR-USERNAME/semantic-film-archivist.git)
cd semantic-film-archivist
