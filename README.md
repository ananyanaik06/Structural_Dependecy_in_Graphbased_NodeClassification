# Structural Dependencies in Graph-Based Node Classification

This project studies how **graph structure alone** influences the performance of **classical machine learning models** for node classification. Using a citation network, we analyze whether node labels are primarily determined by **individual node properties** or by their **neighborhood structure**.

---

## 📌 Overview

- **Task:** Node classification using *graph-structural features only*
- **Dataset:** Cora citation network
- **Graph Type:** Directed citation graph  
  *(Undirected version used for cohesion and centrality features)*
- **Models Evaluated:**
  - Naive Bayes
  - Softmax (Multinomial Logistic Regression)
  - Random Forest

> **Note:**  
> Text/content features are intentionally excluded.  
> The focus is entirely on **structural information derived from graph topology**.

---

## 🔍 Key Findings

- **Neighborhood-level features** (e.g., neighbor degree and centrality) significantly improve classification performance.
- Removing **node-only features** has minimal impact when neighborhood features are present.
- **Naive Bayes** performs poorly due to strong feature independence assumptions.
- **Non-linear models** such as **Random Forest** better capture graph-induced feature dependencies.

---
## Project Structure

```text
graphStructures_MLClassification/
├── data/
│   ├── cora/                  # Extracted Cora dataset files
│   ├── cora.tgz               # Original dataset archive
│   ├── nodes.csv              # Node list
│   ├── node_labels.csv        # Ground-truth labels
│   ├── graph_edges.csv        # Directed edge list
│   ├── node_dataset.csv       # Node features + labels
│   └── graph_features.csv     # Extracted structural features
│
├── feature_extraction.py      # Graph structural feature computation
├── create_dataset.py          # Dataset construction from raw files
├── split_dataset.py           # Stratified train/test split
├── experiment.ipynb           # Model training and evaluation
├── project_report.tex         # LaTeX report
├── project_report.pdf         # Final project report
├── notes.txt                  # Development notes
└── .gitignore
```




---

## 🧩 Structural Features Used

### Node-Level Features
- In-degree
- Out-degree
- Clustering coefficient
- Triangle count

### Centrality Measures
- Betweenness centrality
- Closeness centrality
- Eigenvector centrality

### Neighborhood-Level Features
- Mean neighbor degree
- Maximum neighbor degree
- Mean neighbor clustering coefficient
- Mean neighbor centrality measures

---

## ⚙️ Experimental Setup

- **Stratified train–test split** to preserve class distribution
- **Feature normalization** before model training
- **Evaluation metrics:**
  - Accuracy
  - Macro F1 score
  - Confusion matrix

---

## 📊 Results Summary

| Model               | Accuracy | Macro F1 |
|--------------------|----------|----------|
| Naive Bayes        | ~0.42    | ~0.21    |
| Softmax Regression | ~0.43    | ~0.29    |
| Random Forest      | ~0.62    | ~0.61    |

✅ Including **neighborhood-level features** leads to a **significant improvement** across all models.

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn networkx
```
### 2️⃣ Generate datasets and features
```bash
python create_dataset.py
python feature_extraction.py
python split_dataset.py
```
### 3️⃣ Run experiments
```bash
jupyter notebook experiment.ipynb
```
📄 Report

📄 **Report**

A detailed analysis covering methodology, assumptions, feature design, and experimental results is available here:

[📄 Project Report (PDF)](project_report.pdf)


## 🎯 Motivation

This project aims to understand what **classical machine learning models** can extract from **graph structure alone**, without relying on deep learning or content-based features.

The findings:

- Highlight the **importance of neighborhood structure**
- Motivate the need for **more expressive graph-based learning methods**

## 🔮 Possible Extensions

- **Incorporate Graph Neural Networks (GNNs):**  
  Compare classical ML models with GCN, GraphSAGE, or GAT to quantify gains from learned message passing.

- **Ablation Studies on Structural Features:**  
  Systematically remove centrality, clustering, or neighborhood features to analyze their individual contributions.

- **Multi-hop Neighborhood Features:**  
  Extend feature extraction to include 2-hop and 3-hop neighborhood statistics.

- **Edge Weighting and Directionality Analysis:**  
  Study the impact of weighted edges and explicit directionality on classification performance.

- **Cross-Dataset Evaluation:**  
  Validate findings on other citation networks such as CiteSeer or PubMed.

- **Temporal Graph Analysis:**  
  Model evolving citation graphs and analyze how structural dependencies change over time.

- **Hybrid Models:**  
  Combine structural features with content-based features to study complementary effects.

- **Explainability and Feature Importance:**  
  Use permutation importance or SHAP to interpret model decisions and identify influential structural features.

