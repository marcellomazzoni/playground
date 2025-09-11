import streamlit as st
import pandas as pd
from src.util import show_centered_plot
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from src.util import debug_cross_val, are_params_empty

# ------------------------ Step 1: Parameter Selection ------------------------
st.title("K-Nearest Neighbors (KNN) Model Training & Testing")

# Check if data was uploaded and confirmed
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first in the Home page.")
    st.stop()

if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.header(f"Analysis of < {st.session_state.uploaded_file.name} >")

# ------------------------ Session State Init ------------------------
if 'KNN_trained' not in st.session_state:
    st.session_state.KNN_trained = False

if 'KNN_to_train' not in st.session_state:
    st.session_state.KNN_to_train = False

if 'KNN_tested' not in st.session_state:
    st.session_state.KNN_tested = False
if 'KNN_to_test' not in st.session_state:
    st.session_state.KNN_to_test = False
if 'KNN_params_changed' not in st.session_state:
    st.session_state.KNN_params_changed = False
if 'KNN_first_entered' not in st.session_state:
    st.session_state.KNN_first_entered = True

# ------------------------ UI & Param Capture ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['ml_dataset']
    target = st.session_state['target']
    first_time = st.session_state.KNN_first_entered

    if first_time:
        st.session_state.KNN_last_params = {
            'test_size': None,
            'cv_folds': None,
            'n_neighbors': None,
            'weights': None,
            'metric': None,
        }

    # Widgets collect parameter inputs from user
    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Size (%)', min_value=5, max_value=50, value=20, step=5) / 100
    cv_folds = st.sidebar.slider('CV Folds', min_value=2, max_value=10, value=5)
    st.sidebar.markdown('---')
    n_neighbors = st.sidebar.multiselect('n_neighbors', [3, 5, 7, 9, 11], default=[3, 5], accept_new_options=True, max_selections=5)
    weights = st.sidebar.multiselect('weights', ['uniform', 'distance'], default=['uniform'])
    metric = st.sidebar.multiselect('distance metric', ['cosine', 'chebyshev', 'euclidean', 'manhattan'], default=['euclidean'])
    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State (seed)', min_value=0, max_value=2_147_483_647, value=42, step=1)

    # Store last-used hyperparameters to detect changes
    KNN_current_params = {
        'test_size': test_size,
        'cv_folds': cv_folds,
        'n_neighbors': n_neighbors,
        'weights': weights,
        'metric': metric,
        'random_state': seed,
    }

    # TRAIN trigger
    if st.button("🚀 Start Training"):
        st.session_state.KNN_to_train = True
        st.session_state.KNN_first_entered = False
        st.session_state.KNN_params_changed = False
        st.session_state.KNN_last_params = KNN_current_params
        st.session_state.KNN_to_test = False
        st.session_state.KNN_tested = False

    # Param change detection after button logic
    if (KNN_current_params != st.session_state.KNN_last_params) and st.session_state.KNN_trained is True:
        st.session_state.KNN_params_changed = True
        st.session_state.KNN_to_train = False
        st.session_state.KNN_trained = False
        st.session_state.KNN_to_test = False
        st.session_state.KNN_tested = False
        st.session_state.pop('KNN_test_metrics', None)
        st.session_state.pop('KNN_cv_summary', None)
        st.session_state.pop('KNN_y_pred', None)
        st.session_state.pop('KNN_y_proba', None)

    if st.session_state.KNN_params_changed is True:
        st.warning("⚠️ Parameters have changed. Please re-train the model.")

    # ------------------------ Step 2: Training (Compute) ------------------------
    if st.session_state.KNN_to_train is True and st.session_state.KNN_to_test is False:
        with st.spinner("Training model…"):
            a_clean = dataframe.dropna()
            X = a_clean.select_dtypes(include=['float64', 'int64']).drop([target], axis=1, errors='ignore')
            y = a_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=st.session_state.KNN_last_params['test_size'],
                random_state=st.session_state.KNN_last_params['random_state']
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if st.session_state["problem_type"] in ["classification_multi", "classification_binary"]:
                knn = KNeighborsClassifier()
                param_grid = [{
                    'n_neighbors': st.session_state.KNN_last_params['n_neighbors'],
                    'weights': st.session_state.KNN_last_params['weights'],
                    'metric': st.session_state.KNN_last_params['metric']
                }]
                
                if are_params_empty(param_grid, necessary_params = ['n_neighbors','weights','metric']):
                    st.stop()
                    
                grid_search = GridSearchCV(
                    knn,
                    param_grid,
                    cv=st.session_state.KNN_last_params['cv_folds'],
                    scoring='accuracy',
                    return_train_score=False
                )
                grid_search.fit(X_train_scaled, y_train)
                debug_cross_val(grid_search)

                best_idx = grid_search.best_index_
                cv_mean = grid_search.cv_results_['mean_test_score'][best_idx]
                cv_std  = grid_search.cv_results_['std_test_score'][best_idx]
                st.session_state.KNN_cv_summary = {
                    "k": st.session_state.KNN_last_params['cv_folds'],
                    "metrics": {"Accuracy": {"mean": float(cv_mean), "std": float(cv_std)}},
                    "primary": "Accuracy"
                }

            elif st.session_state["problem_type"] == "regression":
                knn = KNeighborsRegressor()
                param_grid = [{
                    'n_neighbors': st.session_state.KNN_last_params['n_neighbors'],
                    'weights': st.session_state.KNN_last_params['weights'],
                    'metric': st.session_state.KNN_last_params['metric']
                }]
                
                if are_params_empty(param_grid, necessary_params = ['n_neighbors','weights','metric']):
                    st.stop()

                grid_search = GridSearchCV(
                    knn,
                    param_grid,
                    cv=st.session_state.KNN_last_params['cv_folds'],
                    scoring=['neg_mean_absolute_error','neg_root_mean_squared_error','r2'],
                    refit='neg_root_mean_squared_error',
                    return_train_score=False
                )
                grid_search.fit(X_train_scaled, y_train)
                debug_cross_val(grid_search)

                best_idx = grid_search.best_index_
                cv = grid_search.cv_results_
                mae_mean = -float(cv['mean_test_neg_mean_absolute_error'][best_idx])
                mae_std  =  float(cv['std_test_neg_mean_absolute_error'][best_idx])
                rmse_mean = -float(cv['mean_test_neg_root_mean_squared_error'][best_idx])
                rmse_std  =  float(cv['std_test_neg_root_mean_squared_error'][best_idx])
                r2_mean = float(cv['mean_test_r2'][best_idx])
                r2_std  = float(cv['std_test_r2'][best_idx])

                st.session_state.KNN_cv_summary = {
                    "k": st.session_state.KNN_last_params['cv_folds'],
                    "metrics": {
                        "RMSE": {"mean": rmse_mean, "std": rmse_std},
                        "MAE":  {"mean": mae_mean,  "std": mae_std},
                        "R²":   {"mean": r2_mean,   "std": r2_std},
                    },
                    "primary": "RMSE"
                }

            st.success("✅ Training Completed")

            st.session_state.KNN_cv_results = grid_search
            st.session_state.KNN_X_test_scaled = X_test_scaled
            st.session_state.KNN_y_test = y_test
            st.session_state.KNN_trained = True

    if st.session_state.KNN_trained is True:
        st.markdown("### 🏋 Training Set Operations")
        st.markdown("")
        st.markdown("#### 🎯 Best Parameters")
        cv_results = st.session_state.KNN_cv_results
        col1, col2, col3 = st.columns(3)
        for idx, (param, value) in enumerate(cv_results.best_params_.items()):
            col = [col1, col2, col3][idx % 3]
            col.metric(f"{param}", f"{value}")

        st.markdown("#### 🧪 Cross-Validation Performance")
        cvsum = st.session_state.get("KNN_cv_summary", {})
        if cvsum:
            k = cvsum.get("k", "?")
            mets = cvsum.get("metrics", {})
            cols = st.columns(min(3, max(1, len(mets))))
            for i, (name, stats) in enumerate(mets.items()):
                mean_val = stats["mean"]
                std_val = stats["std"]
                label = f"CV {name} (k={k})"
                cols[i % len(cols)].metric(label, f"{mean_val:.3f}")

        st.session_state.KNN_to_train = False

        st.markdown("---")
        if st.button("🧮 Run Test Evaluation"):
            st.session_state.KNN_to_test = True

        if st.session_state.KNN_to_test is True:
            with st.spinner("Testing model…"):
                st.markdown("### 🔍 Test Set Evaluation")
                cv_results = st.session_state.KNN_cv_results
                y_test = st.session_state.KNN_y_test
                y_pred = cv_results.predict(st.session_state.KNN_X_test_scaled)
                st.session_state.KNN_y_pred = y_pred

                match st.session_state.problem_type:
                    case 'classification_binary':
                        st.session_state.KNN_y_proba = cv_results.predict_proba(st.session_state.KNN_X_test_scaled)[:, 1]
                        st.session_state.KNN_test_metrics = {
                            "accuracy": float(accuracy_score(y_test, y_pred)),
                            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                            "confusion_matrix": confusion_matrix(y_test, y_pred)
                        }

                    case 'classification_multi':
                        st.session_state.KNN_y_proba = cv_results.predict_proba(st.session_state.KNN_X_test_scaled)
                        st.session_state.KNN_test_metrics = {
                            "accuracy": float(accuracy_score(y_test, y_pred)),
                            "macro_f1": float(f1_score(y_test, y_pred, average='macro')),
                            "weighted_f1": float(f1_score(y_test, y_pred, average='weighted')),
                            "confusion_matrix": confusion_matrix(y_test, y_pred)
                        }

                    case 'regression':
                        st.session_state.KNN_test_metrics = {
                            "mse": float(mean_squared_error(y_test, y_pred)),
                            "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
                            "mae": float(mean_absolute_error(y_test, y_pred)),
                            "r2": float(r2_score(y_test, y_pred))
                        }

                st.session_state.KNN_tested = True
                st.session_state.KNN_to_test = False

        if st.session_state.KNN_tested is True:
            match st.session_state.problem_type:
                
                case 'classification_binary':
                    st.markdown("")
                    st.markdown("#### 📊 Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{st.session_state.KNN_test_metrics['accuracy']:.3f}")
                    col2.metric("Precision", f"{st.session_state.KNN_test_metrics['precision']:.3f}")
                    col3.metric("Recall", f"{st.session_state.KNN_test_metrics['recall']:.3f}")
                    col4.metric("F1-score", f"{st.session_state.KNN_test_metrics['f1']:.3f}")
                    y_test = st.session_state.KNN_y_test
                    y_proba = st.session_state.KNN_y_proba
                    y_pred = st.session_state.KNN_y_pred

                    st.markdown("")
                    st.markdown("#### 🧩 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(4.2, 3.6))
                    sns.heatmap(
                        st.session_state.KNN_test_metrics["confusion_matrix"],
                        annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False
                    )
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    fig.tight_layout()
                    show_centered_plot(fig)
                    
                    st.markdown("")
                    st.markdown("#### 📑 Classification Report")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose().drop('accuracy')
                    report_df['support'] = report_df['support'].fillna('')
                    styled_df = report_df.style.format(
                        formatter={
                            'precision': '{:.2f}',
                            'recall': '{:.2f}',
                            'f1-score': '{:.2f}',
                            'support': '{:.0f}',
                        }
                    ).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                    st.dataframe(styled_df, width='stretch')


                    st.markdown("")
                    st.markdown("#### 📈 AUC Analysis")
                    
                    curve_type = st.radio(
                        "Select curve type:",
                        ['ROC Curve', 'Precision-Recall Curve'],
                        key="knn_auc_curve_type",
                        horizontal=True
                    )

                    if curve_type == 'ROC Curve':
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig, ax = plt.subplots(figsize=(4.2, 3.6))
                        ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
                        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('ROC Curve')
                        ax.legend()
                        fig.tight_layout()
                        show_centered_plot(fig)

                    else:
                        precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
                        pr_auc = auc(recalls, precisions)
                        fig, ax = plt.subplots(figsize=(4.2, 3.6))
                        ax.plot(recalls, precisions, label=f"PR AUC = {pr_auc:.3f}")
                        ax.set_xlabel('Recall')
                        ax.set_ylabel('Precision')
                        ax.set_title('Precision-Recall Curve')
                        ax.legend()
                        fig.tight_layout()
                        show_centered_plot(fig)

                case 'classification_multi':
                    st.markdown("")
                    st.markdown("#### 📊 Key Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{st.session_state.KNN_test_metrics['accuracy']:.3f}")
                    col2.metric("Macro F1", f"{st.session_state.KNN_test_metrics['macro_f1']:.3f}")
                    col3.metric("Weighted F1", f"{st.session_state.KNN_test_metrics['weighted_f1']:.3f}")
                    y_test = st.session_state.KNN_y_test
                    y_pred = st.session_state.KNN_y_pred

                    st.markdown("")
                    st.markdown("#### 🧩 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        st.session_state.KNN_test_metrics["confusion_matrix"],
                        annot=True, fmt='d', cmap='Blues', ax=ax
                    )
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    fig.tight_layout()
                    show_centered_plot(fig)

                    st.markdown("")
                    st.markdown("#### 📑 Classification Report")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose().drop('accuracy')
                    report_df['support'] = report_df['support'].fillna('')
                    styled_df = report_df.style.format(
                        formatter={
                            'precision': '{:.2f}',
                            'recall': '{:.2f}',
                            'f1-score': '{:.2f}',
                            'support': '{:.0f}',
                        }
                    ).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                    st.dataframe(styled_df, width='stretch')

                case 'regression':
                    st.markdown("")
                    st.markdown("#### 📊 Regression Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MSE", f"{st.session_state.KNN_test_metrics['mse']:.3f}")
                    col2.metric("RMSE", f"{st.session_state.KNN_test_metrics['rmse']:.3f}")
                    col3.metric("MAE", f"{st.session_state.KNN_test_metrics['mae']:.3f}")
                    col4.metric("R²", f"{st.session_state.KNN_test_metrics['r2']:.3f}")

                    y_test = st.session_state.KNN_y_test
                    y_pred = st.session_state.KNN_y_pred

                    st.markdown("")
                    st.markdown("#### 📈 Actual vs Predicted Values")
                    fig, ax = plt.subplots(figsize=(4.2, 3.6))
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    fig.tight_layout()
                    show_centered_plot(fig)

                    st.markdown("")
                    st.markdown("#### 📊 Residuals Plot")
                    residuals = y_test - y_pred
                    fig, ax = plt.subplots(figsize=(4.2, 3.6))
                    ax.scatter(y_pred, residuals, alpha=0.5)
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_xlabel('Predicted Values')
                    ax.set_ylabel('Residuals')
                    fig.tight_layout()
                    show_centered_plot(fig)
