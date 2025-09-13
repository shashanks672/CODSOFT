# Task 4 - Movie Genre Classification (CODSOFT Internship)

## 📌 Overview
This project predicts a movie's **genre** from its **plot/description** using Natural Language Processing (NLP) and machine learning.

## 📊 Dataset
Dataset: [IMDb Movies Dataset](https://www.kaggle.com/datasets/ramjasmaurya/imdb-movies-dataset)

- Contains movie title, description, and genre.
- Genres: Action, Comedy, Drama, Romance, etc.

## ⚙️ Workflow
1. Load dataset  
2. Clean text (lowercasing, punctuation removal, stopword removal)  
3. Convert text to **TF-IDF vectors**  
4. Train models: Logistic Regression, Naive Bayes  
5. Evaluate with Accuracy, Precision, Recall  

## 🚀 How to Run
```bash
pip install -r requirements.txt
jupyter notebook movie_genre_classification.ipynb