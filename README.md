
# Quora Duplicate Question Pairs

This repository contains experiments and implementations on the **Quora Duplicate Question Pairs dataset** (\~442k pairs). The main goal was to explore and compare different **feature engineering techniques** and **machine learning/deep learning models** for duplicate question detection.

## Files Included

* **`bow_with_basic_features.ipynb`**
  Implemented Bag of Words (BoW) with simple hand-crafted features (like question length, word overlap, etc.).

  * Trained on **50k sample data**.

* **`bow_with_textPreprocessing_and_advanced_features.ipynb`**
  Extended BoW approach with text preprocessing (lowercasing, stopword removal, etc.) and additional advanced features (like fuzzy matching, token-based similarity).

  * Trained on **100k data**.

* **`tfidf_with_basic_features.ipynb`**
  Used TF-IDF vectorization combined with basic engineered features to improve representation of text.

  * Trained on **100k data**.

* **`tfidf_with_textPreprocessing_and_advanced_features.ipynb`**
  Combined TF-IDF vectors with text preprocessing and advanced features to evaluate the impact on accuracy.

  * Trained on **100k data**.

* **`DL_using_word2vec+BiLSTM.ipynb`**
  Built a deep learning model using **Word2Vec embeddings** and **BiLSTM layers** for semantic understanding of questions.

  * Trained on **400k data**.

## Accuracy Comparison

| Approach                                   | Dataset Size | Accuracy |
| ------------------------------------------ | ------------ | -------- |
| BoW + Basic Features                       | 50k          | 78.68%    |
| BoW + Preprocessing + Advanced Features    | 100k         | 80.78%  |
| TF-IDF + Basic Features                    | 100k         | 79.31%  |
| TF-IDF + Preprocessing + Advanced Features | 100k         | 80.57%   |
| Word2Vec + BiLSTM (Deep Learning)          | 400k         | 77.05%   |

*(Note: Accuracy may slightly vary depending on random states and hyperparameters.)*

## Key Highlights

* Explored both **classical ML approaches** (Random Forest, XGBoost) and **Deep Learning approaches** (BiLSTM).
* Compared performance across BoW, TF-IDF, and Word2Vec embeddings.
* Investigated the effect of **basic vs. advanced features** and **with/without preprocessing** on model performance.
* Showed how dataset size impacts performance (small sample vs. full dataset).

## Dataset

* **Quora Duplicate Question Pairs**: 404,000+ training question pairs.
* Publicly available dataset (Kaggle).

## Goal

The repository serves as a **comprehensive practice project** to understand:

* Text preprocessing, feature engineering, and vectorization (BoW, TF-IDF, Word2Vec).
* Balancing data for classification tasks.
* Comparing ML models vs. Deep Learning for NLP problems.

With this repo, you can track how accuracy changes when shifting from **basic BoW** to **advanced features** and finally to **Deep Learning with embeddings**.
