import streamlit as st
import pandas as pd
from src.util import show_centered_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ------------------------ Step 1: Parameter Selection ------------------------
st.title("Linear Models Training & Testing")

# Check if data was uploaded and confirmed
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first in the Home page.")
    st.stop()

if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.header(f"Analysis of < {st.session_state.uploaded_file.name} >")

# ------------------------ Session State Init ------------------------
if 'LM_trained' not in st.session_state:
    st.session_state.LM_trained = False

if 'LM_to_train' not in st.session_state:
    st.session_state.LM_to_train = False

if 'LM_tested' not in st.session_state:
    st.session_state.LM_tested = False
if 'LM_to_test' not in st.session_state:
    st.session_state.LM_to_test = False
if 'LM_params_changed' not in st.session_state:
    st.session_state.LM_params_changed = False
if 'LM_first_entered' not in st.session_state:
    st.session_state.LM_first_entered = True

# ------------------------ UI & Param Capture ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['dataframe']
    target = st.session_state['target']
    first_time = st.session_state.LM_first_entered

    st.sidebar.header('Model Type')
    model_type = st.sidebar.selectbox("Select Linear Model", ["Linear Regression", "Polynomial Regression", "Lasso", "Ridge"])

    if first_time:
        st.session_state.LM_last_params = {
            'test_size': None,
            'model_type': None,
            'degree': None,
            'alpha': None,
        }

    # Widgets collect parameter inputs from user
    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Size (%)', min_value=5, max_value=50, value=20, step=5) / 100

    degree = None
    if model_type == "Polynomial Regression":
        degree = st.sidebar.slider('Polynomial Degree', min_value=2, max_value=10, value=2, step=1)

    alpha = None
    if model_type in ["Lasso", "Ridge"]:
        alpha = st.sidebar.slider('Alpha (Regularization Strength)', min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        
    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State (seed)', min_value=0, max_value=2_147_483_647, value=42, step=1)

    # Store last-used hyperparameters to detect changes
    LM_current_params = {
        'test_size': test_size,
        'model_type': model_type,
        'degree': degree,
        'alpha': alpha,
        'random_state': seed,
    }

    # TRAIN trigger
    if st.button("🚀 Start Training"):
        st.session_state.LM_to_train = True
        st.session_state.LM_first_entered = False
        st.session_state.LM_params_changed = False
        st.session_state.LM_last_params = LM_current_params
        st.session_state.LM_to_test = False
        st.session_state.LM_tested = False

    # Param change detection after button logic
    if (LM_current_params != st.session_state.LM_last_params) and st.session_state.LM_trained is True:
        st.session_state.LM_params_changed = True
        st.session_state.LM_to_train = False
        st.session_state.LM_trained = False
        st.session_state.LM_to_test = False
        st.session_state.LM_tested = False
        st.session_state.pop('LM_test_metrics', None)
        st.session_state.pop('LM_y_pred', None)

    if st.session_state.LM_params_changed is True:
        st.warning("⚠️ Parameters have changed. Please re-train the model.")

    # ------------------------ Step 2: Training (Compute) ------------------------
    if st.session_state.LM_to_train is True and st.session_state.LM_to_test is False:
        with st.spinner("Training model…"):
            a_clean = dataframe.dropna()
            X = a_clean.select_dtypes(include=['float64', 'int64']).drop([target], axis=1, errors='ignore')
            y = a_clean[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=st.session_state.LM_last_params['test_size'],
                random_state=st.session_state.LM_last_params['random_state']
            )

            scaler = StandardScaler()

            model_type = st.session_state.LM_last_params['model_type']

            if model_type == "Linear Regression":
                model = LinearRegression()
                pipeline = Pipeline(steps=[('scaler', scaler), ('model', model)])
                pipeline.fit(X_train, y_train)

            elif model_type == "Polynomial Regression":
                degree = st.session_state.LM_last_params['degree']
                model = Pipeline([
                    ('poly', PolynomialFeatures(degree=degree)),
                    ('scaler', scaler),
                    ('regressor', LinearRegression())
                ])
                model.fit(X_train, y_train)
                pipeline = model

            elif model_type == "Lasso":
                alpha = st.session_state.LM_last_params['alpha']
                model = Lasso(alpha=alpha)
                pipeline = Pipeline(steps=[('scaler', scaler), ('model', model)])
                pipeline.fit(X_train, y_train)

            elif model_type == "Ridge":
                alpha = st.session_state.LM_last_params['alpha']
                model = Ridge(alpha=alpha)
                pipeline = Pipeline(steps=[('scaler', scaler), ('model', model)])
                pipeline.fit(X_train, y_train)

            st.success("✅ Training Completed")

            st.session_state.LM_best_model = pipeline
            st.session_state.LM_X_test = X_test
            st.session_state.LM_y_test = y_test
            st.session_state.LM_trained = True

    if st.session_state.LM_trained is True:
        st.markdown("### 🏋 Training Set Operations")
        st.markdown("")
        st.markdown("#### 🎯 Model Parameters")

        params = st.session_state.LM_last_params
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Type", params['model_type'])
        if params['degree']:
            col2.metric("Polynomial Degree", params['degree'])
        if params['alpha']:
            col3.metric("Alpha", params['alpha'])

        st.session_state.LM_to_train = False

        st.markdown("---")
        if st.button("🧮 Run Test Evaluation"):
            st.session_state.LM_to_test = True

        if st.session_state.LM_to_test is True:
            with st.spinner("Testing model…"):
                st.markdown("### 🔍 Test Set Evaluation")
                best_model = st.session_state.LM_best_model
                X_test = st.session_state.LM_X_test
                y_test = st.session_state.LM_y_test
                y_pred = best_model.predict(X_test)
                st.session_state.LM_y_pred = y_pred

                st.session_state.LM_test_metrics = {
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "r2": float(r2_score(y_test, y_pred))
                }

                st.session_state.LM_tested = True
                st.session_state.LM_to_test = False

        if st.session_state.LM_tested is True:
            st.markdown("")
            st.markdown("#### 📊 Regression Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{st.session_state.LM_test_metrics['mse']:.3f}")
            col2.metric("RMSE", f"{st.session_state.LM_test_metrics['rmse']:.3f}")
            col3.metric("MAE", f"{st.session_state.LM_test_metrics['mae']:.3f}")
            col4.metric("R²", f"{st.session_state.LM_test_metrics['r2']:.3f}")

            y_test = st.session_state.LM_y_test
            y_pred = st.session_state.LM_y_pred

            st.markdown("")
            st.markdown("#### 📈 Actual vs Predicted Values")
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            fig.tight_layout()
            show_centered_matplotlib(fig)

            st.markdown("")
            st.markdown("#### 📊 Residuals Plot")
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ax.scatter(y_pred, residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            fig.tight_layout()
            show_centered_matplotlib(fig)
