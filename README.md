# 🚀 AutoML 2Stage for Tabular Datasets

An automated machine learning (AutoML) system that automatically detects the problem type, evaluates multiple machine learning algorithms, selects the top-performing models, performs hyperparameter tuning, and returns the best model.

Built using Python, Scikit-Learn, XGBoost, and Streamlit.

---

## ✨ Features

- Automatic problem type detection (Classification or Regression)
- Intelligent preprocessing for numerical and categorical features
- Missing value handling and feature encoding
- Evaluation of multiple machine learning algorithms
- Cross-validation based baseline screening
- Automatic selection of top 3 performing models
- Hyperparameter optimization using RandomizedSearchCV
- Automatic best model selection
- Performance metrics generation
- Confusion Matrix visualization for classification tasks
- Download trained model as a .pkl file
- Interactive Streamlit web interface

---

## 🏗️ 2-Stage AutoML Architecture

### Stage 1: Baseline Screening

All candidate algorithms are evaluated using cross-validation.

#### Classification Models
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- XGBoost

#### Regression Models
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor
- XGBoost Regressor

The top 3 performing models are selected.

---

### Stage 2: Hyperparameter Tuning

The selected top models undergo hyperparameter optimization using RandomizedSearchCV.

The best-performing tuned model is selected as the final model.

---

## 📊 Evaluation Metrics

### Classification
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Regression
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-Learn
- XGBoost
- Pandas
- Matplotlib
- Joblib

---

## 📂 Project Structure

```text
AutoML-2Stage-for-Tabular-Datasets/
│
├── app.py
├── requirements.txt
├── README.md
└── best_model.pkl (generated after training)
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AutoML-2Stage-for-Tabular-Datasets.git
```

Move into the project directory:

```bash
cd AutoML-2Stage-for-Tabular-Datasets
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📋 How to Use

1. Launch the Streamlit application.
2. Upload a CSV dataset.
3. Select the target column.
4. Choose the evaluation metric (for classification).
5. Click **Run AutoML**.
6. View:
   - Baseline model scores
   - Top 3 selected models
   - Hyperparameter tuning results
   - Final ranking
   - Performance metrics
7. Download the best trained model.

---

## 🎯 Example Workflow

Dataset → Problem Type Detection → Data Preprocessing → Baseline Model Screening → Top 3 Model Selection → Hyperparameter Tuning → Best Model Selection → Performance Evaluation → Model Download

---

## Future Improvements

- Feature Selection
- Automated Feature Engineering
- Class Imbalance Handling
- Ensemble Model Generation
- Explainable AI (SHAP)
- Automated Report Generation
- Multi-objective Optimization

---

