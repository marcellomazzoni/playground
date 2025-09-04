import streamlit as st
import pandas as pd
from src.util import show_centered_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from src.util import get_numeric_x_and_y_from_df

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
    
    #HERE YOU HAVE TO CREATE MAGIC
    if st.toggle("Show Correlation Chart"):
        st.subheader("Correlation Matrix")
        # Consider using a smaller sample for performance on large datasets
        sample_df = dataframe.sample(min(1000, len(dataframe)))
        fig = sns.pairplot(sample_df, diag_kind='kde')
        st.pyplot(fig)


    st.sidebar.header('Model Parameters')
    test_size = st.sidebar.slider('Test Size (%)', min_value=5, max_value=50, value=20, step=5) / 100
    cv_folds = st.sidebar.slider('CV Folds', min_value=2, max_value=10, value=5, help="Number of folds for cross-validation during GridSearchCV.")
    regularization = st.sidebar.multiselect("Regularization", ["None", "Lasso", "Ridge"], default="None")

    alphas = None
    if "Lasso" in regularization or "Ridge" in regularization:
        alphas = st.sidebar.multiselect('Alpha (Regularization Strength)', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], default=[0.1, 1.0], help="Select multiple values for GridSearchCV")

    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State (seed)', min_value=0, max_value=2_147_483_647, value=42, step=1)


    if first_time:
        st.session_state.LM_last_params = {
            'test_size': None,
            'cv_folds': None,
            'regularization': None,
            'alphas': None,
            'random_state': None
        }

    # Store last-used hyperparameters to detect changes
    LM_current_params = {
        'test_size': test_size,
        'cv_folds': cv_folds,
        'regularization': regularization,
        'alphas': alphas,
        'random_state': seed
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
        st.session_state.pop('LM_cv_summary', None)


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
            
            # The scaler should be part of the pipeline to prevent data leakage from the test set
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression()) # Placeholder
            ])

            regularization_types = st.session_state.LM_last_params['regularization']
            alphas_to_tune = st.session_state.LM_last_params['alphas']

            if "None" in regularization_types and len(regularization_types) == 1:
                pipeline.fit(X_train, y_train)
                st.session_state.LM_best_model = pipeline
                st.session_state.LM_cv_summary = None

            else:
                if not alphas_to_tune:
                    st.error("Please select at least one Alpha value for regularization.")
                    st.stop()

                # Create a list of models for each regularization type
                models = []
                if "Lasso" in regularization_types:
                    models.append(('Lasso', Lasso(random_state=st.session_state.LM_last_params['random_state'])))
                if "Ridge" in regularization_types:
                    models.append(('Ridge', Ridge(random_state=st.session_state.LM_last_params['random_state'])))

                # Create parameter grid including model type and alpha
                param_grid = {
                    'model': [model[1] for model in models],
                    'model__alpha': alphas_to_tune
                }

                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=st.session_state.LM_last_params['cv_folds'],
                    scoring='neg_root_mean_squared_error',
                    return_train_score=True
                )
                grid_search.fit(X_train, y_train)
                
                st.session_state.LM_best_model = grid_search.best_estimator_
                st.session_state.LM_cv_summary = pd.DataFrame(grid_search.cv_results_)


            st.success("✅ Training Completed")

            st.session_state.LM_X_test = X_test
            st.session_state.LM_y_test = y_test
            st.session_state.LM_trained = True

    if st.session_state.LM_trained is True:
        st.markdown("### 🏋 Training Set Operations")
        st.markdown("")

        params = st.session_state.LM_last_params
        
        if "None" not in params['regularization']:
            st.markdown("#### 🎯 Best Parameters")
            best_model = st.session_state.LM_best_model
            col1, col2 = st.columns(2)
            model_type = best_model.named_steps['model'].__class__.__name__
            col1.metric("Best Model Type", model_type)
            best_alpha = best_model.named_steps['model'].alpha
            col2.metric("Best Alpha", f"{best_alpha:.4f}")
            
            st.markdown("#### 🧪 Cross-Validation Performance")
            cv_results = st.session_state.get('LM_cv_summary')
            if cv_results is not None:
                st.write("CV Results Summary:")
                # Add model type to the summary
                cv_summary = cv_results[['param_model', 'param_model__alpha', 'mean_test_score', 'std_test_score', 'mean_train_score']]
                # Extract model type from the parameter
                cv_summary['Model Type'] = cv_summary['param_model'].apply(lambda x: x.__class__.__name__)
                cv_summary = cv_summary.drop('param_model', axis=1)
                cv_summary = cv_summary.rename(columns={
                    'param_model__alpha': 'Alpha',
                    'mean_test_score': 'Mean Test RMSE',
                    'std_test_score': 'Std Test RMSE',
                    'mean_train_score': 'Mean Train RMSE'
                })
                # Invert the sign of scores since we used 'neg_root_mean_squared_error'
                cv_summary['Mean Test RMSE'] = -cv_summary['Mean Test RMSE']
                cv_summary['Mean Train RMSE'] = -cv_summary['Mean Train RMSE']
                st.dataframe(cv_summary.style.format({
                    'Alpha': '{:.4f}',
                    'Mean Test RMSE': '{:.4f}',
                    'Std Test RMSE': '{:.4f}',
                    'Mean Train RMSE': '{:.4f}'
                }))

        else: # No regularization
            st.markdown("#### 🎯 Model Parameters")
            col1, _ = st.columns(2)
            col1.metric("Model Type", "Linear Regression")

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
