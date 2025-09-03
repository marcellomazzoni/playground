import streamlit as st
import pandas as pd
from src.util import (
    show_centered_matplotlib, generate_model_formula_latex,
    debug_cross_val, get_numeric_x_and_y_from_df
)
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split, GridSearchCV
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
for key, default in [
    ('LM_trained', False),
    ('LM_to_train', False),
    ('LM_tested', False),
    ('LM_to_test', False),
    ('LM_params_changed', False),
    ('LM_first_entered', True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------------ UI & Param Capture ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['dataframe']
    target = st.session_state['target']
    first_time = st.session_state.LM_first_entered
    
    X, y = get_numeric_x_and_y_from_df(dataframe, target)
    if not st.session_state.LM_trained:
        st.latex(generate_model_formula_latex(y, X, model_type='linear_regression', model=None))

    if first_time:
        st.session_state.LM_last_params = {
            'test_size': None,
            'cv_folds': None,
            'degree_choice': [],
            'alpha_choice': [],
            'model_choices': [],
            'random_state': 42
        }

    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Size (%)', min_value=5, max_value=50, value=20, step=5) / 100
    cv_folds = st.sidebar.slider('CV Folds', min_value=2, max_value=10, value=5)
    st.sidebar.markdown('---')

    degree_choice = st.sidebar.multiselect(
        'Polynomial Degrees',
        options=[1, 2, 3, 4],
        default=[1],
        help="Degrees to test. All degrees up to the max selected will be included."
    )

    alpha_choice = st.sidebar.multiselect(
        'Regularization Strengths (Alpha)',
        options=[0.01, 0.1, 1.0, 10.0],
        help="Alpha values for Lasso and Ridge."
    )

    model_choices = []
    if alpha_choice:
        model_choices = st.sidebar.multiselect(
            'Regularized Models',
            options=['Lasso', 'Ridge'],
            default=['Lasso', 'Ridge'],
            help="Choose Lasso, Ridge, or both."
        )

    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State (seed)', min_value=0, max_value=2_147_483_647, value=42, step=1)

    # Store last-used hyperparameters to detect changes
    LM_current_params = {
        'test_size': test_size,
        'cv_folds': cv_folds,
        'degree_choice': degree_choice,
        'alpha_choice': alpha_choice,
        'model_choices': model_choices,
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
            X, y = get_numeric_x_and_y_from_df(dataframe, target)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=st.session_state.LM_last_params['test_size'],
                random_state=st.session_state.LM_last_params['random_state']
            )

            # Base pipeline: (preprocessor -> scaler -> regressor)
            pipeline = Pipeline([
                ('preprocessor', 'passthrough'),   # IMPORTANT: use 'passthrough', not None
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])

            # Dynamically build the parameter grid
            param_grid = []

            # User choices
            degree_choice = st.session_state.LM_last_params['degree_choice']
            alpha_choice = st.session_state.LM_last_params['alpha_choice']
            model_choices = st.session_state.LM_last_params['model_choices']

            max_deg = max(degree_choice) if degree_choice else 1
            degrees_to_test = list(range(2, max_deg + 1)) if max_deg > 1 else []

            # --- Plain Linear Regression (no polynomial) ---
            # Use passthrough for "no poly" branch
            param_grid.append({
                'preprocessor': ['passthrough'],
                'regressor': [LinearRegression()]
            })

            # --- Polynomial + Linear Regression ---
            if degrees_to_test:
                param_grid.append({
                    'preprocessor': [PolynomialFeatures(include_bias=False)],
                    'preprocessor__degree': degrees_to_test,
                    'regressor': [LinearRegression()]
                })

            # --- Regularized Models (no polynomial) ---
            if alpha_choice and model_choices:
                if 'Lasso' in model_choices:
                    param_grid.append({
                        'preprocessor': ['passthrough'],
                        'regressor': [Lasso(max_iter=10000)],
                        'regressor__alpha': alpha_choice
                    })
                if 'Ridge' in model_choices:
                    param_grid.append({
                        'preprocessor': ['passthrough'],
                        'regressor': [Ridge(max_iter=10000)],
                        'regressor__alpha': alpha_choice
                    })

            # --- Regularized Models (with polynomial) ---
            if degrees_to_test and alpha_choice and model_choices:
                if 'Lasso' in model_choices:
                    param_grid.append({
                        'preprocessor': [PolynomialFeatures(include_bias=False)],
                        'preprocessor__degree': degrees_to_test,
                        'regressor': [Lasso(max_iter=10000)],
                        'regressor__alpha': alpha_choice
                    })
                if 'Ridge' in model_choices:
                    param_grid.append({
                        'preprocessor': [PolynomialFeatures(include_bias=False)],
                        'preprocessor__degree': degrees_to_test,
                        'regressor': [Ridge(max_iter=10000)],
                        'regressor__alpha': alpha_choice
                    })

            # Safety check
            if not param_grid:
                st.error("⚠️ Error: Please select at least one valid model configuration to test.")
                st.stop()
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=st.session_state.LM_last_params['cv_folds'],
                scoring='r2',
                return_train_score=False,
                n_jobs=-1,
                error_score='raise'
            )
            
            try:
                grid_search.fit(X_train, y_train)

                st.session_state.LM_cv_results = grid_search
                st.session_state.LM_X_test = X_test
                st.session_state.LM_y_test = y_test
                st.session_state.LM_trained = True

                st.success("✅ Training Completed")

            except Exception as e:
                st.error(f"An error occurred during model training: {e}")
                st.session_state.LM_to_train = False

    # ------------------------ Step 2: Training (Display CV metrics) ------------------------
    if st.session_state.LM_trained is True:
        st.markdown("### 🏋 Training Set Operations")
        st.markdown("")
        st.markdown("#### 🎯 Best Parameters")
        cv_results = st.session_state.LM_cv_results
        col1, col2, col3, col4 = st.columns(4)
        # Metric display (prettify preprocessor name if needed)
        def _pretty(v):
            if isinstance(v, PolynomialFeatures):
                return f"PolynomialFeatures(deg={getattr(v, 'degree', '?')})"
            return str(v)
        items = list(cv_results.best_params_.items())
        for idx, (param, value) in enumerate(items):
            col = [col1, col2, col3, col4][idx % 4]
            col.metric(f"{param.split('__')[-1]}", f"{_pretty(value)}")
        
        st.markdown("")
        st.markdown("#### 🧪 Cross-Validation Performance")
        best_idx = cv_results.best_index_
        cv_mean = cv_results.cv_results_['mean_test_score'][best_idx]
        st.metric("R² Score (CV)", f"{cv_mean:.3f}")

        st.session_state.LM_to_train = False
        st.markdown("---")
        
        best_model = cv_results.best_estimator_
        
        # Determine features used for display (raw or polynomial)
        preproc = best_model.named_steps.get('preprocessor', 'passthrough')
        if hasattr(preproc, 'get_feature_names_out'):
            # Derive feature names from polynomial transformer
            X_formula = pd.DataFrame(
                preproc.fit_transform(X),
                columns=preproc.get_feature_names_out(X.columns)
            )
        else:
            X_formula = X
        
        # Display the best model's formula (note: coefficients reflect standardized features)
        st.markdown("### 🧩 Best model estimated")
        final_regressor = best_model.named_steps['regressor']
        st.latex(generate_model_formula_latex(y, X_formula, model_type='linear_regression', model=final_regressor))

        # ------------------------ Step 3: Testing (Trigger + Compute) ------------------------
        st.markdown("---")
        if st.button("🧮 Run Test Evaluation"):
            st.session_state.LM_to_test = True

        if st.session_state.LM_to_test is True:
            with st.spinner("Testing model…"):
                st.markdown("### 🔍 Test Set Evaluation")
                cv_results = st.session_state.LM_cv_results
                X_test = st.session_state.LM_X_test
                y_test = st.session_state.LM_y_test
                
                # Predict with full best pipeline
                y_pred = cv_results.best_estimator_.predict(X_test)
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
