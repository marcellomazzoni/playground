import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score

# ------------------------ Step 1: Parameter Selection ------------------------
st.header("Random Forest Model Training & Testing")

# Check if data was uploaded and confirmed
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first in the Home page.")
    st.stop()
    
if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.subheader(f"Analysis of < {st.session_state.uploaded_file.name} >")

# Set parameters to None if last used parameters are not in the session state (The user has updated them)
if 'RF_trained' not in st.session_state:
    st.session_state.RF_trained = False     # Tracks if the model has been trained
if 'RF_run_test' not in st.session_state:
    st.session_state.RF_run_test = False    # Tracks if the test evaluation has been run
if 'RF_params_changed' not in st.session_state:
    st.session_state.RF_params_changed = False  # Tracks if any parameter has changed and retraining is needed
if 'RF_first_entered' not in st.session_state:
    st.session_state.RF_first_entered = True  # Tracks if any parameter has changed and retraining is needed

if st.session_state.confirmed:
    dataframe = st.session_state['dataframe']
    target = st.session_state['target']
    first_time = st.session_state.RF_first_entered
    
    if first_time:
        st.session_state.RF_last_params = {
            'test_size': None,
            'cv_folds': None,
            'n_estimators': None,
            'max_depth': None,
            'min_samples_split': None
        }
            
    # Widgets collect parameter inputs from user
    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Size (%)', min_value=5, max_value=50, value=20, step=5) / 100
    cv_folds = st.sidebar.slider('CV Folds', min_value=2, max_value=10, value=5)
    st.sidebar.markdown('---')
    n_estimators = st.sidebar.multiselect('n_estimators', [10, 20, 100, 200, 300], default=[100, 200], accept_new_options=True, max_selections=5)
    max_depth = st.sidebar.multiselect('max_depth', [5, 10, 20, None], default=[10, 20], accept_new_options=True, max_selections=5)
    min_samples_split = st.sidebar.multiselect('min_samples_split', [2, 5, 10], default=[2, 5], accept_new_options=True, max_selections=5)
    
    # Check if parameters have changed
    # Store last-used hyperparameters to detect changes        
    RF_current_params = {
        'test_size': test_size,
        'cv_folds': cv_folds,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }
    
    st.write(st.session_state.RF_trained)
    if st.session_state.RF_params_changed == True:
        st.warning("⚠️ Parameters have changed. Please re-train the model.")

    # If user changes params, force them to retrain the model (for correctness)
    if (RF_current_params != st.session_state.RF_last_params) and st.session_state.RF_trained == True:
        st.session_state.RF_params_changed = True
        st.session_state.RF_trained = False
        st.session_state.RF_run_test = False 

    if st.button("🚀 Start Training"):
        # When training starts, update session state accordingly
        st.session_state.RF_trained = True
        st.session_state.RF_first_entered = False
        st.session_state.RF_params_changed = False
        st.session_state.RF_last_params = RF_current_params
        st.session_state.RF_run_test = False

# ------------------------ Step 2: Training ------------------------
if st.session_state.RF_trained and target:
    st.subheader("Model Training in Progress...")
    
    a_clean = dataframe.dropna()
    X = a_clean.select_dtypes(include=['float64', 'int64']).drop([target], axis=1, errors='ignore')
    y = a_clean[target]

    st.dataframe(X.head(10))
    st.dataframe(y.head(10))
    
    # Training/validation/test split using parameters saved in session_state
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, 
                                                              test_size=st.session_state.RF_last_params['test_size'], 
                                                              random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, 
                                                      test_size=0.2, 
                                                      random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # CLASSIFICATION
    if st.session_state["problem_type"] in ["classification_multi", "classification_binary"]:
        
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': st.session_state.RF_last_params['n_estimators'],
            'max_depth': st.session_state.RF_last_params['max_depth'],
            'min_samples_split': st.session_state.RF_last_params['min_samples_split']
        }
        # grid_search = GridSearchCV(rf,
        #                            param_grid, 
        #                            cv = st.session_state.RF_last_params['cv_folds'],
        #                            scoring='accuracy')
        
        # grid_search.fit(X_train_scaled, y_train)
        # st.success("✅ Training Completed")
        # st.write("Best Parameters Found:", grid_search.best_params_)
        # st.write("Validation Accuracy:", grid_search.score(X_val_scaled, y_val))
        
        grid_search = GridSearchCV(rf,
                       param_grid, 
                       cv = st.session_state.RF_last_params['cv_folds'],
                       scoring='accuracy')
        
        grid_search.fit(X_train_scaled, y_train)
        st.success("✅ Training Completed")
        
        # Display best parameters in a more organized way
        st.subheader("🎯 Best Parameters")
        col1, col2, col3 = st.columns(3)
        for idx, (param, value) in enumerate(grid_search.best_params_.items()):
            col = [col1, col2, col3][idx % 3]
            col.metric(f"{param}", f"{value}")
            
        # Display validation accuracy with metric widget
        st.subheader("📊 Model Performance")
        st.metric("Validation Accuracy", f"{grid_search.score(X_val_scaled, y_val):.3f}")
        
        
    # REGRESSION
    elif st.session_state["problem_type"] == "regression":
        rf = RandomForestRegressor(random_state=42)
        
        param_grid = {
            'n_estimators': st.session_state.RF_last_params['n_estimators'],
            'max_depth': st.session_state.RF_last_params['max_depth'],
            'min_samples_split': st.session_state.RF_last_params['min_samples_split']
        }
        
        grid_search = GridSearchCV(rf,
                                   param_grid,
                                   cv=st.session_state.RF_last_params['cv_folds'],
                                   scoring=['neg_mean_absolute_error','neg_root_mean_squared_error','r2'],
                                   refit = 'neg_root_mean_squared_error')
        
        grid_search.fit(X_train_scaled, y_train)
        st.success("✅ Training Completed")
        # Display best parameters in a more organized way
        st.subheader("🎯 Best Parameters")
        col1, col2, col3 = st.columns(3)
        for idx, (param, value) in enumerate(grid_search.best_params_.items()):
            col = [col1, col2, col3][idx % 3]
            col.metric(f"{param}", f"{value}")
            
        # Display validation accuracy with metric widget
        st.subheader("📊 Model Performance")
        st.metric("Validation Accuracy", f"{grid_search.score(X_val_scaled, y_val):.3f}")

    if st.button("🧪 Run Test Evaluation"):
        # Save all artifacts for test evaluation in session_state for downstream use
        st.session_state.RF_run_test = True
        st.session_state.RF_best_model = grid_search
        st.session_state.RF_X_test_scaled = X_test_scaled
        st.session_state.RF_y_test = y_test

# ------------------------ Step 3: Testing ------------------------
if st.session_state.RF_run_test:
    st.subheader("🔍 Test Set Evaluation")

    # Recover saved objects from session_state (persistence across reruns)
    best_model = st.session_state.RF_best_model
    X_test_scaled = st.session_state.RF_X_test_scaled
    y_test = st.session_state.RF_y_test

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
        y_proba = best_model.predict_proba(X_test_scaled)
        
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
        
    ## Regression
    else:  
        # Compute regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display metrics
        st.subheader("📊 Regression Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{mse:.3f}")
        col2.metric("RMSE", f"{rmse:.3f}")
        col3.metric("MAE", f"{mae:.3f}")
        col4.metric("R²", f"{r2:.3f}")

        # Actual vs Predicted Plot
        st.subheader("📈 Actual vs Predicted Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        st.pyplot(fig)

        # Residuals Plot
        st.subheader("📊 Residuals Plot")
        residuals = y_test - y_pred
        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        st.pyplot(fig)
