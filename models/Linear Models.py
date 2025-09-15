import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
# Assuming 'src.util' contains your custom utility functions.
# If running standalone, replace with the mock functions provided below.
from src.util import debug_cross_val, generate_model_formula_latex, get_numeric_x_and_y_from_df, show_centered_plot, plot_residuals, plot_actual_vs_predicted, load_descriptions

tooltips = load_descriptions()

# ------------------------ Page Configuration ------------------------
st.title("Linear Models Training & Testing 📈")

# Check if data was uploaded and confirmed from a previous step
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first on the Home page.")
    st.stop()

if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.header(f"Analysis of: `{st.session_state.uploaded_file.name}`")

# ------------------------ Session State Initialization ------------------------
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


# ------------------------ Step 1: UI & Parameter Selection ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['ml_dataset']
    target = st.session_state['target']
    first_time = st.session_state.LM_first_entered

    X, y = get_numeric_x_and_y_from_df(dataframe, target)
    st.markdown("### 📝 Model Specification")
    st.latex(generate_model_formula_latex(y, X, model_type='linear_regression', model=None))

    if first_time:
        st.session_state.LM_last_params = {
            'test_size': None,
            'cv_folds': None,
            'regularization': None,
            'alphas': None,
            'random_state': None
        }

    # --- Sidebar Widgets for Parameter Input ---
    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Size (%)', min_value=5, max_value=50, value=20, step=5, help = tooltips['general']['test_size']) / 100
    cv_folds = st.sidebar.slider('CV Folds', min_value=2, max_value=10, value=5,  help = tooltips['general']['cv_folds'])
    regularization = st.sidebar.multiselect("Regularization", ["None", "Lasso", "Ridge"], default=["None"], key="lm_regularization", help = tooltips["linear_models"]["regularization"])

    alphas = None
    if "Lasso" in regularization or "Ridge" in regularization:
        alphas = st.sidebar.multiselect(
            'Alpha (Regularization Strength)',
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            default=[0.1, 1.0],
            help = tooltips["linear_models"]["alpha"]
        )

    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State (seed)', min_value=0, max_value=2_147_483_647, value=42, step=1, help = tooltips['general']['random_state'])

    # Store current hyperparameters to detect changes
    LM_current_params = {
        'test_size': test_size,
        'cv_folds': cv_folds,
        'regularization': sorted(regularization), # Sort for consistent comparison
        'alphas': alphas,
        'random_state': seed
    }

    # ------------------------ State & Trigger Logic ------------------------
    if st.button("🚀 Start Training"):
        st.session_state.LM_to_train = True
        st.session_state.LM_first_entered = False
        st.session_state.LM_params_changed = False
        st.session_state.LM_last_params = LM_current_params
        # Reset downstream states
        st.session_state.LM_trained = False
        st.session_state.LM_to_test = False
        st.session_state.LM_tested = False
        st.session_state.pop('LM_cv_results', None)
        st.session_state.pop('LM_test_metrics', None)

    # Detect parameter changes after the first training run
    if not first_time and (LM_current_params != st.session_state.LM_last_params):
        st.session_state.LM_params_changed = True
        st.session_state.LM_trained = False
        st.session_state.LM_to_train = False
        st.session_state.LM_tested = False
        st.session_state.LM_to_test = False

    if st.session_state.LM_params_changed:
        st.warning("⚠️ Parameters have changed. Please click 'Start Training' to apply them.")

    # ------------------------ Step 2: Training (Computation) ------------------------
    if st.session_state.get('LM_to_train', False):
        with st.spinner("Finding the best model... This may take a moment. ⏳"):
            X, y = get_numeric_x_and_y_from_df(dataframe, target)
            params = st.session_state.LM_last_params

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=params['test_size'],
                random_state=params['random_state']
            )

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', 'passthrough')
            ])

            param_grid = []
            if "None" in params['regularization']:
                param_grid.append({'model': [LinearRegression()]})

            if "Lasso" in params['regularization']:
                if not params['alphas']:
                    st.error("Please select at least one Alpha value for Lasso.")
                    st.stop()
                param_grid.append({
                    'model': [Lasso(random_state=params['random_state'])],
                    'model__alpha': params['alphas']
                })

            if "Ridge" in params['regularization']:
                if not params['alphas']:
                    st.error("Please select at least one Alpha value for Ridge.")
                    st.stop()
                param_grid.append({
                    'model': [Ridge(random_state=params['random_state'])],
                    'model__alpha': params['alphas']
                })

            if not param_grid:
                st.error("Please select at least one regularization type to train a model.")
                st.stop()

            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=params['cv_folds'],
                scoring='neg_root_mean_squared_error',
                return_train_score=False,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            debug_cross_val(grid_search) # Optional: for server-side debugging

            st.session_state.LM_cv_results = grid_search
            st.session_state.LM_X_test = X_test
            st.session_state.LM_y_test = y_test
            st.session_state.LM_trained = True
            st.session_state.LM_to_train = False # Computation is done
            st.success("✅ Training Completed!")

    # ------------------------ Step 3: Display Training & CV Results ------------------------
    if st.session_state.LM_trained:
        cv_results = st.session_state.LM_cv_results
        st.markdown("---")
        st.markdown("### 🏋️ Training Set Performance")

        st.markdown("#### 🎯 Best Model Found")
        best_model_pipeline = cv_results.best_estimator_
        model_step = best_model_pipeline.named_steps['model']
        model_type = model_step.__class__.__name__

        best_params = {key.split('__')[-1]: val for key, val in cv_results.best_params_.items()}
        best_params.pop('model', None)

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Type", model_type)
        param_cols = [col2, col3]
        for i, (param, value) in enumerate(best_params.items()):
            display_val = f"{value:.4f}" if isinstance(value, float) else str(value)
            param_cols[i % len(param_cols)].metric(f"Best {param.capitalize()}", display_val)

        st.markdown("#### 🧪 Cross-Validation Performance")
        cv_results_df = pd.DataFrame(cv_results.cv_results_)
        best_index = cv_results.best_index_
        mean_rmse = -cv_results_df.loc[best_index, 'mean_test_score']
        std_rmse = cv_results_df.loc[best_index, 'std_test_score']
        k = st.session_state.LM_last_params['cv_folds']
        st.metric(f"Mean CV RMSE (k={k})", f"{mean_rmse:.3f} (± {std_rmse:.3f})")

        st.markdown("#### 🧩 Best Model Equation")
        st.latex(generate_model_formula_latex(y, X, model_type='linear_regression', model=cv_results.best_estimator_))

        # ------------------------ Step 4: Test Set Evaluation (Trigger) ------------------------
        st.markdown("---")
        if st.button("🧮 Evaluate on Test Set"):
            st.session_state.LM_to_test = True

        # ------------------------ Step 4: Test Set Evaluation (Compute) ------------------------
        if st.session_state.get('LM_to_test', False):
            with st.spinner("Running predictions on the test set..."):
                best_model = st.session_state.LM_cv_results.best_estimator_
                X_test = st.session_state.LM_X_test
                y_test = st.session_state.LM_y_test
                y_pred = best_model.predict(X_test)

                st.session_state.LM_y_pred = y_pred
                st.session_state.LM_test_metrics = {
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": math.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred)
                }
                st.session_state.LM_tested = True
                st.session_state.LM_to_test = False

        # ------------------------ Step 4: Test Set Evaluation (Display) ------------------------
        if st.session_state.LM_tested:
            st.markdown("### 🔍 Test Set Evaluation")
            st.markdown("#### 📊 Regression Metrics")

            metrics = st.session_state.LM_test_metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{metrics['mse']:.3f}")
            col2.metric("RMSE", f"{metrics['rmse']:.3f}")
            col3.metric("MAE", f"{metrics['mae']:.3f}")
            col4.metric("R²", f"{metrics['r2']:.3f}")

            y_test = st.session_state.LM_y_test
            y_pred = st.session_state.LM_y_pred

            st.markdown("#### 📈 Actual vs Predicted Values")
            fig = plot_actual_vs_predicted(y_true = y_test, y_pred = y_pred)
            show_centered_plot(fig)

            st.markdown("#### 📊 Residuals Plot")
            fig = plot_residuals(y_true = y_test, y_pred = y_pred)
            show_centered_plot(fig)
