import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ------------------------ Step 1: Parameter Selection ------------------------
st.header("SVM Model Training & Testing (Classification Only)")

# Check if data was uploaded and confirmed
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first in the Home page.")
    st.stop()
    
if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.subheader(f"Analysis of < {st.session_state.uploaded_file.name} >")

# Initialize session state for SVM
if 'SVM_trained' not in st.session_state:
    st.session_state.SVM_trained = False     # Tracks if the model has been trained
if 'SVM_run_test' not in st.session_state:
    st.session_state.SVM_run_test = False    # Tracks if the test evaluation has been run
if 'SVM_params_changed' not in st.session_state:
    st.session_state.SVM_params_changed = False  # Tracks if any parameter has changed and retraining is needed
if 'SVM_first_entered' not in st.session_state:
    st.session_state.SVM_first_entered = True  # Tracks first entry for param init

if st.session_state.confirmed:
    dataframe = st.session_state['dataframe']
    target = st.session_state['target']
    first_time = st.session_state.SVM_first_entered
    
    if first_time:
        st.session_state.SVM_last_params = {
            'test_size': None,
            'cv_folds': None,
            'C': None,
            'kernel': None,
            'gamma': None
        }
        st.session_state.SVM_first_entered = False
            
    # Widgets collect parameter inputs from user
    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Size (%)', min_value=5, max_value=50, value=20, step=5) / 100
    cv_folds = st.sidebar.slider('CV Folds', min_value=2, max_value=10, value=5)
    st.sidebar.markdown('---')
    C_values = st.sidebar.multiselect('C (Regularization)', [0.1, 1, 10, 100], default=[1, 10], accept_new_options=True, max_selections=5)
    kernel_values = st.sidebar.multiselect('kernel', ['linear', 'rbf', 'poly', 'sigmoid'], default=['linear', 'rbf'], accept_new_options=True, max_selections=4)
    gamma_values = st.sidebar.multiselect('gamma', ['scale', 'auto'], default=['scale'], accept_new_options=False, max_selections=2)
    
    # Check if parameters have changed
    SVM_current_params = {
        'test_size': test_size,
        'cv_folds': cv_folds,
        'C': C_values,
        'kernel': kernel_values,
        'gamma': gamma_values
    }

    st.write(st.session_state.SVM_trained)
    if st.session_state.SVM_params_changed == True:
        st.warning("⚠️ Parameters have changed. Please re-train the model.")

    # If user changes params, force them to retrain the model (for correctness)
    if (SVM_current_params != st.session_state.SVM_last_params) and st.session_state.SVM_trained == True:
        st.session_state.SVM_params_changed = True
        st.session_state.SVM_trained = False
        st.session_state.SVM_run_test = False 

    if st.button("🚀 Start Training"):
        # When training starts, update session state accordingly
        st.session_state.SVM_trained = True
        st.session_state.SVM_params_changed = False
        st.session_state.SVM_last_params = SVM_current_params
        st.session_state.SVM_run_test = False

# ------------------------ Step 2: Training ------------------------
if st.session_state.SVM_trained and st.session_state.get('target'):
    st.subheader("Model Training in Progress...")
    
    a_clean = dataframe.dropna()
    X = a_clean.select_dtypes(include=['float64', 'int64']).drop([target], axis=1, errors='ignore')
    y = a_clean[target]

    st.dataframe(X.head(10))
    st.dataframe(y.head(10))
    
    # Training/validation/test split using parameters saved in session_state
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=st.session_state.SVM_last_params['test_size'], random_state=42, stratify=y if st.session_state.get("problem_type","").startswith("classification") else None
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval if st.session_state.get("problem_type","").startswith("classification") else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # CLASSIFICATION ONLY
    if st.session_state.get("problem_type") in ["classification_multi", "classification_binary"]:
        svc = SVC(probability=True, random_state=42)
        param_grid = {
            'C': st.session_state.SVM_last_params['C'],
            'kernel': st.session_state.SVM_last_params['kernel'],
            'gamma': st.session_state.SVM_last_params['gamma']
        }
        
        grid_search = GridSearchCV(
            svc,
            param_grid, 
            cv=st.session_state.SVM_last_params['cv_folds'],
            scoring='accuracy'
        )
        
        grid_search.fit(X_train_scaled, y_train)
        st.success("✅ Training Completed")
        
        # Display best parameters in a more organized way
        st.subheader("🎯 Best Parameters")
        col1, col2, col3 = st.columns(3)
        for idx, (param, value) in enumerate(grid_search.best_params_.items()):
            col = [col1, col2, col3][idx % 3]
            col.metric(f"{param}", f"{value}")
            
        # Display validation accuracy
        st.subheader("📊 Model Performance")
        st.metric("Validation Accuracy", f"{grid_search.score(X_val_scaled, y_val):.3f}")
        
    else:
        st.warning("SVM module only supports classification in this app. Please select a classification problem type.")
        st.stop()

    if st.button("🧪 Run Test Evaluation"):
        # Save all artifacts for test evaluation in session_state for downstream use
        st.session_state.SVM_run_test = True
        st.session_state.SVM_best_model = grid_search
        st.session_state.SVM_X_test_scaled = X_test_scaled
        st.session_state.SVM_y_test = y_test

# ------------------------ Step 5: Testing ------------------------
if st.session_state.get('SVM_run_test', False):
    st.subheader("🔍 Test Set Evaluation")

    # Recover saved objects from session_state (persistence across reruns)
    best_model = st.session_state.SVM_best_model
    X_test_scaled = st.session_state.SVM_X_test_scaled
    y_test = st.session_state.SVM_y_test

    y_pred = best_model.predict(X_test_scaled)
    
    # Metrics display
    ## Binary Classification
    if st.session_state.problem_type == 'classification_binary':
        # Binary Classification metrics and plots
        y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Compute key metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Display metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Precision", f"{prec:.3f}")
        col3.metric("Recall", f"{rec:.3f}")
        col4.metric("F1-score", f"{f1:.3f}")

        # Confusion matrix
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        # ROC and PR curves
        st.subheader("📈 AUC Analysis")
        curve_type = st.radio("Select curve type:", ['ROC Curve', 'Precision-Recall Curve'])

        if curve_type == 'ROC Curve':
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            st.pyplot(fig)
        else:
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            st.pyplot(fig)

    ## Multiclass classification
    elif st.session_state.problem_type == 'classification_multi':
        # Multiclass Classification metrics and plots
        # SVC with probability=True provides per-class probabilities if needed
        _ = best_model.predict_proba(X_test_scaled)
        
        # Compute key metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        # Display metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.3f}")
        col2.metric("Macro F1", f"{macro_f1:.3f}")
        col3.metric("Weighted F1", f"{weighted_f1:.3f}")

        # Confusion matrix
        st.subheader("🧩 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        # Classification Report
        st.subheader("📑 Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)
