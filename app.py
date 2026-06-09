import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

# ---------------- UI ----------------
st.title("🚀 AutoML System (2-Stage)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin-1")

    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---------------- PROBLEM TYPE ----------------
    def detect_problem_type(y):
        if y.dtype == "object":
            return "classification"

        if pd.api.types.is_integer_dtype(y):
            if y.nunique() <= 10:
                return "classification"

        return "regression"

    problem_type = detect_problem_type(y)
    st.write("Problem Type:", problem_type)

    if problem_type == "classification":
        y = LabelEncoder().fit_transform(y)

    # ---------------- PREPROCESSORS ----------------
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    preprocessor_scaled = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

    preprocessor_no_scale = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="mean"))
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols)
    ])

    # ---------------- MODELS ----------------
    if problem_type == "classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

        param_dist = {

            "Logistic Regression": {
                "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "model__solver": ["lbfgs", "liblinear"],
                "model__class_weight": [None, "balanced"]
            },

            "Decision Tree": {
                "model__max_depth": [None, 5, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            },

            "Random Forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": [None, "balanced"]
            },

            "SVM": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__kernel": ["linear", "rbf"]
            },

            "KNN": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]
            },

            "XGBoost": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        }

        scoring = st.selectbox(
            "Choose Evaluation Metric",
            ["accuracy", "f1_weighted"]
        )

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "SVM": SVR(),
            "KNN": KNeighborsRegressor(),
            "XGBoost": XGBRegressor()
        }

        param_dist = {

            "Decision Tree": {
                "model__max_depth": [None, 5, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            },

            "Random Forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            },

            "SVM": {
                "model__C": [0.01, 0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__epsilon": [0.01, 0.1, 0.5]
            },

            "KNN": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]
            },

            "XGBoost": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0]
            }
        }
        scoring = "r2"
        st.info(f"Model Selection Metric: {scoring}")

    # ---------------- RUN ----------------
    if st.button("Run AutoML 🚀"):
        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if problem_type == "classification" else None
        )

        baseline_results = []

        st.subheader("Step 1: Baseline Screening")

        for name, model in models.items():

            if isinstance(model, (SVC, SVR, KNeighborsClassifier, KNeighborsRegressor, LogisticRegression)):
                prep = preprocessor_scaled
            else:
                prep = preprocessor_no_scale

            pipe = Pipeline([
                ("preprocess", prep),
                ("model", model)
            ])

            score = cross_val_score(pipe, X_train, y_train, cv=3, scoring=scoring).mean()

            baseline_results.append((name, score))
            st.write(f"{name}: {score:.4f}")

        # ---------------- SELECT TOP MODELS ----------------
        baseline_results.sort(key=lambda x: x[1], reverse=True)
        top_models = [name for name, _ in baseline_results[:3]]

        st.subheader(f"Top Models: {top_models}")

        # ---------------- STEP 2: TUNING ----------------
        st.subheader("Step 2: Hyperparameter Tuning")

        best_params_dict = {}

        final_results = []

        best_pipeline = None
        best_score = float("-inf")

        for name in top_models:

            model = models[name]

            if isinstance(model, (SVC, SVR, KNeighborsClassifier, KNeighborsRegressor, LogisticRegression)):
                prep = preprocessor_scaled
            else:
                prep = preprocessor_no_scale

            pipe = Pipeline([
                ("preprocess", prep),
                ("model", model)
            ])

            if name in param_dist:
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_dist[name],
                    n_iter=5,
                    cv=3,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=42
                )

                search.fit(X_train, y_train)

                score = search.score(X_test, y_test)

                best_params_dict[name] = search.best_params_

            else:
                pipe.fit(X_train, y_train)
                score = pipe.score(X_test, y_test)

            final_results.append((name, score))
            st.write(f"{name}: {score:.4f}")

            if score > best_score:
                best_score = score

                if name in param_dist:
                    best_pipeline = search.best_estimator_
                else:
                    best_pipeline = pipe

        # ---------------- FINAL OUTPUT ----------------
        st.subheader("🏆 Final Ranking")

        results_df = pd.DataFrame(final_results, columns=["Model", "Score"])
        results_df = results_df.sort_values(by="Score", ascending=False)

        st.dataframe(results_df)

        st.subheader("Best Hyperparameters")
        for model_name, params in best_params_dict.items():
            st.write(model_name)
            st.json(params)

        end_time = time.time()
        st.info(f"Total Training Time: {end_time - start_time:.2f} seconds")

        best_model = results_df.iloc[0]
        st.success(f"Best Model: {best_model['Model']}")
        st.success(f"Score: {best_model['Score']:.4f}")

        # Save model
        joblib.dump(best_pipeline, "best_model.pkl")

        # Download button
        with open("best_model.pkl", "rb") as f:
            st.download_button(
                label="📥 Download Best Model",
                data=f,
                file_name="best_model.pkl",
                mime="application/octet-stream"
            )

        # Predictions from best model
        y_pred = best_pipeline.predict(X_test)

        st.subheader("📈 Performance Metrics")
        if problem_type == "classification":

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Value": [acc, prec, rec, f1]
            })

            st.dataframe(metrics_df)

        else:

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            metrics_df = pd.DataFrame({
                "Metric": ["R² Score", "MAE", "RMSE"],
                "Value": [r2, mae, rmse]
            })

            st.dataframe(metrics_df)

        if problem_type == "classification":

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(4,4))
            ax.matshow(cm)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), va='center', ha='center')

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)