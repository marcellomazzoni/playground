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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report
)
from src.util import debug_cross_val, are_params_empty, generate_model_formula_latex, get_numeric_x_and_y_from_df, load_descriptions

tooltips = load_descriptions()

# ------------------------ Step 1: Parameter Selection ------------------------
st.title("Logistic Regression Model Training & Testing")

# Check if data was uploaded and confirmed
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first in the Home page.")
    st.stop()

if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.header(f"Analysis of: ` {st.session_state.uploaded_file.name} `")

# ------------------------ Session State Init ------------------------
if 'LR_trained' not in st.session_state:
    st.session_state.LR_trained = False

if 'LR_to_train' not in st.session_state:
    st.session_state.LR_to_train = False

if 'LR_tested' not in st.session_state:
    st.session_state.LR_tested = False
if 'LR_to_test' not in st.session_state:
    st.session_state.LR_to_test = False
if 'LR_params_changed' not in st.session_state:
    st.session_state.LR_params_changed = False
if 'LR_first_entered' not in st.session_state:
    st.session_state.LR_first_entered = True

# ------------------------ UI & Param Capture ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['ml_dataset']
    target = st.session_state['target']
    first_time = st.session_state.LR_first_entered
    
    X, y = get_numeric_x_and_y_from_df(dataframe, target)
    a = generate_model_formula_latex(y, X , model_type = 'logistic_regression', model=None)
    st.latex(a)

    if first_time:
        st.session_state.LR_last_params = {
            'test_size': None,
            'cv_folds': None,
            'C': None,
            'penalty': None,
            'solver': None,
        }

    # Widgets collect parameter inputs from user
    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Size (%)', min_value=5, max_value=50, value=20, step=5, help = tooltips['general']['test_size']) / 100
    cv_folds = st.sidebar.slider('CV Folds', min_value=2, max_value=10, value=5,  help = tooltips['general']['cv_folds'])
    st.sidebar.markdown('---')
    C = st.sidebar.multiselect('C', [0.1, 1, 10, 100], default=[1], accept_new_options=True, max_selections=4, help=tooltips["logistic_regression"]["C"])
    penalty = st.sidebar.multiselect('penalty', ['l1', 'l2', 'elasticnet', 'none'], default=['l2'], help=tooltips["logistic_regression"]["penalty"])
    solver = st.sidebar.multiselect('solver', ['newton-cholesky', 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], default=['lbfgs'], help=tooltips["logistic_regression"]["solver"])
    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State (seed)', min_value=0, max_value=2_147_483_647, value=42, step=1, help = tooltips['general']['random_state'])

    # Store last-used hyperparameters to detect changes
    LR_current_params = {
        'test_size': test_size,
        'cv_folds': cv_folds,
        'C': C,
        'penalty': penalty,
        'solver': solver,
        'random_state': seed,
    }

    # TRAIN trigger
    if st.button("🚀 Start Training"):
        st.session_state.LR_to_train = True
        st.session_state.LR_first_entered = False
        st.session_state.LR_params_changed = False
        st.session_state.LR_last_params = LR_current_params
        st.session_state.LR_to_test = False
        st.session_state.LR_tested = False

    # Param change detection after button logic
    if (LR_current_params != st.session_state.LR_last_params) and st.session_state.LR_trained is True:
        st.session_state.LR_params_changed = True
        st.session_state.LR_to_train = False
        st.session_state.LR_trained = False
        st.session_state.LR_to_test = False
        st.session_state.LR_tested = False
        st.session_state.pop('LR_test_metrics', None)
        st.session_state.pop('LR_cv_summary', None)
        st.session_state.pop('LR_y_pred', None)
        st.session_state.pop('LR_y_proba', None)

    if st.session_state.LR_params_changed is True:
        st.warning("⚠️ Parameters have changed. Please re-train the model.")

    # ------------------------ Step 2: Training (Compute) ------------------------
    if st.session_state.LR_to_train is True and st.session_state.LR_to_test is False:
        with st.spinner("Training model…"):
            X, y = get_numeric_x_and_y_from_df(dataframe, target)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=st.session_state.LR_last_params['test_size'],
                random_state=st.session_state.LR_last_params['random_state']
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            lr = LogisticRegression(random_state=st.session_state.LR_last_params['random_state'])

            # Get the selected parameters
            selected_solvers = st.session_state.LR_last_params['solver']
            selected_penalties = st.session_state.LR_last_params['penalty']
            
            if selected_solvers == []:
                st.write("The solver variable is an empty list.")

            if not selected_solvers:
                st.write("The solver variable is an empty list (truthy check).")

            converted_penalties = []
            for p in selected_penalties:
                if p == 'none':
                    converted_penalties.append(None)
                else:
                    converted_penalties.append(p)
                    
            C_values = st.session_state.LR_last_params['C']

            # Initialize an empty list for the parameter grid
            param_grid = []
            # Build the grid dynamically based on compatible solver-penalty pairs
            if 'liblinear' in selected_solvers:
                compatible_penalties = [p for p in converted_penalties if p in ['l1', 'l2']]
                if compatible_penalties:
                    param_grid.append({
                        'solver': ['liblinear'],
                        'penalty': compatible_penalties,
                        'C': C_values
                    })
                    
            if 'saga' in selected_solvers:
                regularized_penalties = [p for p in converted_penalties if p in ['l1', 'l2']]
                elasticnet_penalty = [p for p in converted_penalties if p == 'elasticnet']
                none_penalty = [p for p in converted_penalties if p is None]

                if regularized_penalties:
                    param_grid.append({
                        'solver': ['saga'],
                        'penalty': regularized_penalties,
                        'C': C_values
                    })
                if elasticnet_penalty:
                    param_grid.append({
                        'solver': ['saga'],
                        'penalty': elasticnet_penalty,
                        'l1_ratio': [0.1, 0.5, 0.9],
                        'C': C_values
                    })
                if none_penalty:
                    param_grid.append({
                        'solver': ['saga'],
                        'penalty': none_penalty,
                        'C': [1.0]
                    })
                
                
            for solver in ['lbfgs', 'newton-cg', 'sag', 'newton-cholesky']:
                if solver in selected_solvers:
                    for penalty in converted_penalties:
                        if penalty == 'l2':
                            Cs = C_values
                        elif penalty is None:  # no regularization → C ignored
                            Cs = [1.0]  # more explicit than [1.0]
                        else:
                            continue  # skip unsupported penalties for this solver

                        param_grid.append({
                            'solver': [solver],
                            'penalty': [penalty],
                            'C': Cs,
                        })
            
            if are_params_empty(param_grid, necessary_params=['C'], not_necessary_params=None):
                st.stop()
            
            grid_search = GridSearchCV(
                lr,
                param_grid,
                cv=st.session_state.LR_last_params['cv_folds'],
                scoring='accuracy',
                return_train_score=False,
                error_score='raise'
            )
            try:
                grid_search.fit(X_train_scaled, y_train)

                best_idx = grid_search.best_index_
                cv_mean = grid_search.cv_results_['mean_test_score'][best_idx]
                cv_std = grid_search.cv_results_['std_test_score'][best_idx]
                st.session_state.LR_cv_summary = {
                    "k": st.session_state.LR_last_params['cv_folds'],
                    "metrics": {"Accuracy": {"mean": float(cv_mean), "std": float(cv_std)}},
                    "primary": "Accuracy"
                }

                st.success("✅ Training Completed")

                st.session_state.LR_cv_results = grid_search
                st.session_state.LR_X_test_scaled = X_test_scaled
                st.session_state.LR_y_test = y_test
                st.session_state.LR_trained = True

            except Exception as e:
                st.error(f"An error occurred during model training: {e}")
                st.session_state.LR_to_train = False


    # ------------------------ Step 2: Training (Display CV metrics) ------------------------
    if st.session_state.LR_trained is True:
        debug_cross_val(st.session_state.LR_cv_results)
        st.markdown("### 🏋 Training Set Operations")
        st.markdown("")
        st.markdown("#### 🎯 Best Parameters")
        cv_results = st.session_state.LR_cv_results 
        col1, col2, col3 = st.columns(3)
        for idx, (param, value) in enumerate(cv_results.best_params_.items()):
            col = [col1, col2, col3][idx % 3]
            col.metric(f"{param}", f"{value}")
        st.markdown("")
        st.markdown("#### 🧪 Cross-Validation Performance")
        cvsum = st.session_state.get("LR_cv_summary", {})
        if cvsum:
            k = cvsum.get("k", "?")
            mets = cvsum.get("metrics", {})
            cols = st.columns(min(3, max(1, len(mets))))
            for i, (name, stats) in enumerate(mets.items()):
                mean_val = stats["mean"]
                std_val = stats["std"]
                label = f"CV {name} (k={k})"
                cols[i % len(cols)].metric(label, f"{mean_val:.3f}")

        st.session_state.LR_to_train = False
        st.markdown("")
        st.markdown("### 🧩 Best model estimated")
        st.latex(generate_model_formula_latex(y, X, model_type = 'logistic_regression', model = cv_results.best_estimator_))

        # ------------------------ Step 3: Testing (Trigger + Compute) ------------------------
        st.markdown("---")
        if st.button("🧮 Run Test Evaluation"):
            st.session_state.LR_to_test = True

        if st.session_state.LR_to_test is True:
            with st.spinner("Testing model…"):
                st.markdown("### 🔍 Test Set Evaluation")
                cv_results = st.session_state.LR_cv_results
                y_test = st.session_state.LR_y_test
                y_pred = cv_results.predict(st.session_state.LR_X_test_scaled)
                st.session_state.LR_y_pred = y_pred

                match st.session_state.problem_type:
                    case 'classification_binary':
                        st.session_state.LR_y_proba = cv_results.predict_proba(st.session_state.LR_X_test_scaled)[:, 1]
                        st.session_state.LR_test_metrics = {
                            "accuracy": float(accuracy_score(y_test, y_pred)),
                            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                            "confusion_matrix": confusion_matrix(y_test, y_pred)
                        }

                    case 'classification_multi':
                        st.session_state.LR_y_proba = cv_results.predict_proba(st.session_state.LR_X_test_scaled)
                        st.session_state.LR_test_metrics = {
                            "accuracy": float(accuracy_score(y_test, y_pred)),
                            "macro_f1": float(f1_score(y_test, y_pred, average='macro')),
                            "weighted_f1": float(f1_score(y_test, y_pred, average='weighted')),
                            "confusion_matrix": confusion_matrix(y_test, y_pred)
                        }

                st.session_state.LR_tested = True
                st.session_state.LR_to_test = False

        # ------------------------ Step 3: Testing (Display) ------------------------
        if st.session_state.LR_tested is True:
            match st.session_state.problem_type:
                case 'classification_binary':
                    st.markdown("")
                    st.markdown("#### 📊 Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{st.session_state.LR_test_metrics['accuracy']:.3f}")
                    col2.metric("Precision", f"{st.session_state.LR_test_metrics['precision']:.3f}")
                    col3.metric("Recall", f"{st.session_state.LR_test_metrics['recall']:.3f}")
                    col4.metric("F1-score", f"{st.session_state.LR_test_metrics['f1']:.3f}")
                    y_test = st.session_state.LR_y_test
                    y_proba = st.session_state.LR_y_proba
                    y_pred = st.session_state.LR_y_pred

                    st.markdown("")
                    st.markdown("#### 🧩 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(4.2, 3.6))
                    sns.heatmap(
                        st.session_state.LR_test_metrics["confusion_matrix"],
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
                        key="lr_auc_curve_type",
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
                    col1.metric("Accuracy", f"{st.session_state.LR_test_metrics['accuracy']:.3f}")
                    col2.metric("Macro F1", f"{st.session_state.LR_test_metrics['macro_f1']:.3f}")
                    col3.metric("Weighted F1", f"{st.session_state.LR_test_metrics['weighted_f1']:.3f}")
                    y_test = st.session_state.LR_y_test
                    y_pred = st.session_state.LR_y_pred

                    st.markdown("")
                    st.markdown("#### 🧩 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(
                        st.session_state.LR_test_metrics["confusion_matrix"],
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
                            'macro avg': '{:.2f}',
                            'weighted avg': '{:.2f}',
                            'support': '{:.0f}',
                        }
                    ).background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])
                    st.dataframe(styled_df, width='stretch')
