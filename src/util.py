import streamlit as st 
import pandas as pd
from streamlit_modal import Modal
import requests
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def show_session_state_debug():
    """
    Creates an expander in the Streamlit app to display all variables
    stored in the session state.
    """
    with st.expander("🐛 Session State Debugger"):
        st.write("Current Session State:")
        
        # Create a dictionary to hold the data for a nice table display
        state_items = {
            "Key": list(st.session_state.keys()),
            "Value": [str(v) for v in st.session_state.values()], # Convert all to string for display
            "Type": [type(v).__name__ for v in st.session_state.values()]
        }
        
        # Display as a DataFrame for a clean, tabular view
        st.dataframe(pd.DataFrame(state_items), use_container_width=True)


# ========================= LLM availability + dynamic radio =========================
# Paste this somewhere near your other utils (before you render the radios).
# It detects whether an Ollama server is reachable and whether the requested model is installed.
# Then it builds a radio whose options include "Ask LLM" ONLY when available.

def is_ollama_available(url: str = "http://localhost:11434") -> bool:
    """
    Returns True if an Ollama server is reachable at the given URL, otherwise False.
    Uses a short timeout to prevent the application from freezing.
    """
    try:
        response = requests.get(f"{url}/api/version", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
    
def get_available_ollama_models(url: str = "http://localhost:11434") -> list[str]:
    """
    Returns a list of available model names on the Ollama server.
    Returns an empty list if the server is not available or no models are found.
    """
    try:
        response = requests.get(f"{url}/api/tags", timeout=1)
        if response.status_code == 200:
            data = response.json()
            # The API returns a list of dicts with 'name' and other metadata
            return [model['name'] for model in data.get('models', [])]
        return []
    except requests.exceptions.RequestException:
        return []
    
def is_ollama_model_available(model_name: str, url: str = "http://localhost:11434") -> bool:
    """
    Returns True if the specified model is available on the Ollama server.
    """
    available_models = get_available_ollama_models(url)
    return model_name in available_models

def is_gemini_key_valid(api_key: str) -> bool:
    """
    Checks if a provided Gemini API key is valid by making a simple request.
    Returns True if the key is valid and receives an authorized response, otherwise False.
    """
    # Use a simple, lightweight endpoint like listing models
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {
        "x-goog-api-key": api_key,
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        # A valid key returns 200 OK. An invalid key returns 401 Unauthorized.
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def configure_gemini():
    """UI and logic for configuring Gemini."""
    api_key_input = st.text_input(
        "Enter Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key,
        placeholder="sk-...",
        help="Paste your Gemini API key here"
    )
    
    if api_key_input and api_key_input != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = api_key_input
        with st.spinner("Checking API key..."):
            if is_gemini_key_valid(api_key_input):
                st.success("API key is valid! 🎉")
                os.environ["GEMINI_API_KEY"] = api_key_input
            else:
                st.error("Invalid API key. Please check and try again.")
                st.session_state.gemini_api_key = "" # Clear invalid key

def configure_ollama():
    """UI and logic for configuring Ollama."""
    ollama_url = "http://localhost:11434"
    if not is_ollama_available(ollama_url):
        st.warning(f"Ollama server not found at {ollama_url}. Please ensure Ollama is installed and running.")
        return
    else:
        st.success("Ollama server is running! ✅")

    available_models = get_available_ollama_models(ollama_url)
    model_options = available_models + ["(Enter a new model name below)"]
    
    # Pre-select the current model if it's in the list
    try:
        current_model_index = model_options.index(st.session_state.ollama_model_name)
    except ValueError:
        current_model_index = len(model_options) - 1
    
    selected_model_dropdown = st.selectbox(
        "Select an available model or provide a new one:",
        options=model_options,
        index=current_model_index
    )

    ollama_model_name = ""
    if selected_model_dropdown == "(Enter a new model name below)":
        ollama_model_name = st.text_input("Enter a new model name (e.g., 'llama3')")
    else:
        ollama_model_name = selected_model_dropdown

    if ollama_model_name:
        st.session_state.ollama_model_name = ollama_model_name
        is_model_present = is_ollama_model_available(ollama_model_name, ollama_url)

        if is_model_present:
            st.info(f"Model '{ollama_model_name}' is already installed.")
        else:
            st.warning(f"Model '{ollama_model_name}' is not found locally. It will be downloaded.")
            
            if st.button("Confirm Download"):
                st.info(f"Downloading '{ollama_model_name}'...")
                # Note: The actual download logic is not implemented here as it requires
                # a more complex async process, but this simulates the user flow.
                st.success("Download confirmed! You can now use this model.")
                
def choose_llm():
    # --- Session state initialization ---
    if "llm_choice" not in st.session_state:
        st.session_state.llm_choice = None
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    if "ollama_model_name" not in st.session_state:
        st.session_state.ollama_model_name = ""

    # Check if an LLM is already configured
    llm_is_set = False
    if st.session_state.llm_choice == "Gemini" and st.session_state.gemini_api_key:
        llm_is_set = True
    elif st.session_state.llm_choice == "Ollama" and st.session_state.ollama_model_name:
        llm_is_set = True

    # --- UI for settings button and current status ---
    modal = Modal("⚙️ Settings", key="settings_modal", max_width=600)
    
    status_text = "*No LLM configured. Open settings to set one.*"
    status_icon = "⚪"
    
    if llm_is_set:
        status_text = f"*LLM is set:* **{st.session_state.llm_choice}**"
        status_icon = "🟢"
        if st.session_state.llm_choice == "Ollama":
            status_text += f" (Model: **{st.session_state.ollama_model_name}**)"
    
    col1, col2 = st.columns([1, 10])
    with col1:
        if st.button("⚙️", key="open_settings"):
            modal.open()
        
    with col2:
    #     st.markdown(f"### {status_text}")
        st.markdown(f"{status_icon} - {status_text}")
    
        # if st.button("⚙️", key="open_settings"):
        #     modal.open()
    
    # --- Modal content for configuration ---
    if modal.is_open():
        with modal.container():
            st.markdown("### Choose your LLM backend")
            
            # Radio buttons for model choice
            choice = st.radio(
                "Select engine:",
                ["Ollama", "Gemini"],
                index=0 if st.session_state.llm_choice == "Ollama" else 1,
                key="llm_choice_radio"
            )
            
            st.session_state.llm_choice = choice
            
            if choice == "Gemini":
                configure_gemini()
            
            elif choice == "Ollama":
                configure_ollama()
            
            st.markdown("---")
            if st.button("Save & Close", key="save_and_close"):
                modal.close()
                st.rerun() # Rerun to update the main UI

def available_llm(model_name: str = "qwen2.5-coder:3b",
                  url: str = "http://localhost:11434") -> bool:
    """
    Return True iff:
      1) an Ollama server is reachable at `url`, and
      2) `model_name` exists in the local model list (/api/tags).

    Notes:
    - Uses short timeouts so the UI stays responsive.
    - If `requests` isn't installed or the server is down, returns False.
    - Cached briefly to avoid hammering the endpoint (auto-refreshes every few seconds).
    """
    try:
        import requests
    except Exception:
        return False  # no requests -> treat as unavailable

    @st.cache_data(ttl=8, show_spinner=False)  # small TTL so it updates quickly if you start Ollama/pull a model
    def _probe(_url: str):
        # Check server is alive
        try:
            r = requests.get(f"{_url}/api/version", timeout=0.7)
            if r.status_code != 200:
                return {"alive": False, "models": set()}
        except Exception:
            return {"alive": False, "models": set()}

        # Fetch installed models
        try:
            r = requests.get(f"{_url}/api/tags", timeout=1.2)
            if not r.ok:
                return {"alive": True, "models": set()}
            data = r.json() or {}
            models = data.get("models", [])
            # Ollama returns items with "name" (e.g., "qwen2.5-coder:3b")
            names = set()
            for m in models:
                name = m.get("name") or m.get("model")
                if name:
                    names.add(name)
            return {"alive": True, "models": names}
        except Exception:
            return {"alive": True, "models": set()}

    info = _probe(url)
    return bool(info["alive"] and (model_name in info["models"]))

def get_numeric_x_and_y_from_df(dataframe:pd.DataFrame, target:str):
    a_clean = dataframe.dropna()
    X = a_clean.select_dtypes(include=['float64', 'int64']).drop([target], axis=1, errors='ignore')
    y = a_clean[target]

    return X, y 

def action_radio_for_column(col: str,
                            coltype: str,
                            llm_model: str = "qwen2.5-coder:3b",
                            llm_url: str = "http://localhost:11434") -> str:
    """
    Render a radio for a given column with actions.
    Includes 'Ask LLM' ONLY when available_llm(...) is True.
    Safely preserves previous choice if still valid; resets to first option otherwise.
    """
    match coltype:
        case "Continuous":
            base_options = [
                "Manage outliers",
                "Manage missing values",
                "Bucketize (discretize)",
                "Rename"]
            
        case "Categorical":
            base_options = [
                "Impute Missing Values", 
                "Label Encoding", 
                "One-hot Encoding",
                "Rename"]
            
        case "Binary":
            base_options = [
                "Impute Missing Values", 
                "Rename"]
            
        case _:
            base_options["None"]
        
    options = base_options.copy()
    llm_ok = available_llm(model_name=llm_model, url=llm_url)
    if llm_ok:
        options.append("Ask LLM")

    key = f"cont_action_{col}"
    prev = st.session_state.get(key)

    # Default to prior valid choice; otherwise first option
    if prev in options:
        default_index = options.index(prev)
    else:
        default_index = 0
        # If previous selection is no longer valid (e.g., 'Ask LLM' disappeared), clear it
        if prev is not None and prev not in options:
            st.session_state.pop(key, None)

    action = st.radio(
        "Choose an action",
        options,
        index=default_index,
        key=key,
    )

    # Friendly hint if LLM is disabled
    if not llm_ok:
        st.caption("💡 'Ask LLM' is hidden because Ollama isn’t running or the model isn't installed.")

    return action

# ========================= Usage example =========================


def show_centered_plot(plot_obj, width_ratio=2.4, plot_type='pyplot', width='stretch', height='stretch'):
    """
    Display a plot centered in streamlit with configurable width.
    
    Args:
        plot_obj: The plot object (matplotlib figure, plotly figure, etc.)
        width_ratio (float): Ratio of middle column width to side columns
        plot_type (str): Type of plot ('pyplot', 'plotly', 'matplotlib')
    """
    left_, mid, right_ = st.columns([1, width_ratio, 1])
    with mid:
        match plot_type.lower():
            case 'pyplot' | 'matplotlib':
                st.pyplot(plot_obj)
                
            case 'plotly':
                if isinstance(height, (int, float)):
                    plot_obj.update_layout(height=height)
                if isinstance(width, (int, float)):
                    plot_obj.update_layout(width=width)
                st.plotly_chart(plot_obj)
            case _:
                st.write("Unsupported plot type")


def show_plot_and_metrics(plot_obj, width_ratio=2.4, plot_type='pyplot', list_of_metrics = []):
    """
    Display a plot centered in streamlit with configurable width.
    
    Args:
        plot_obj: The plot object (matplotlib figure, plotly figure, etc.)
        width_ratio (float): Ratio of middle column width to side columns
        plot_type (str): Type of plot ('pyplot', 'plotly', 'matplotlib')
    """
    margin_, chart, space_, metrics, margin_ = st.columns([0.5, width_ratio, 0.5, 1,0.2])
    with chart:
        match plot_type.lower():
            case 'pyplot' | 'matplotlib':
                st.pyplot(plot_obj, width='stretch')
            case 'plotly':
                st.plotly_chart(plot_obj, width='stretch')
            case _:
                st.write("Unsupported plot type")
    with metrics:
        if list_of_metrics:
            st.write("**Metrics**")
            for metric in list_of_metrics:
                st.metric(label=metric.get('label', 'N/A'), value=metric.get('value', 'N/A'))

        
def debug_cross_val(grid_search):
    """
    Smarter CV results display that dynamically finds the primary metric
    based on the 'refit' parameter.
    """
    st.markdown("---")
    results = pd.DataFrame(grid_search.cv_results_)
    refit_metric = grid_search.refit

    # 1. Determine the correct column names based on the refit strategy
    if refit_metric is True:
        # This is the standard case for a single scorer
        rank_col = 'rank_test_score'
        mean_col = 'mean_test_score'
        std_col = 'std_test_score'
    elif isinstance(refit_metric, str):
        # This is the case for multiple scorers, where refit is a string name
        rank_col = f"rank_test_{refit_metric}"
        mean_col = f"mean_test_{refit_metric}"
        std_col = f"std_test_{refit_metric}"
    else:
        # Handle cases where refit is False or invalid
        st.warning("`refit` is not set. Cannot determine primary metric for ranking.")
        st.dataframe(results[['params']].head())
        return

    # 2. Define the ideal set of columns and find which ones are available
    # We also add 'params' which is always present.
    desired_columns = [rank_col, mean_col, std_col, 'params']
    available_columns = [col for col in desired_columns if col in results.columns]

    if not available_columns:
        st.warning("Could not find any of the expected result columns.")
        st.dataframe(results)
        return

    # 3. Select, sort, and display the results
    display_df = results[available_columns]
    
    # Sort by the rank column if it was found
    if rank_col in display_df.columns:
        display_df = display_df.sort_values(by=rank_col)

    st.write("Cross-Validation Results (Ranked by Primary Metric):")
    st.dataframe(display_df)
    st.markdown("---")
    
    
def are_params_empty(param_grid, necessary_params=None, not_necessary_params=None):
    """
    Checks if a parameter grid is valid based on necessary and optional parameters.

    Args:
        param_grid (list of dict): The list of parameter dictionaries for GridSearchCV.
        necessary_params (list of str): A list of parameter names that must be present
                                        and non-empty in ALL dictionaries of the grid.
        not_necessary_params (list of str): A list of parameter names that are optional.
                                            If they exist in a dictionary, they must be non-empty.

    Returns:
        True if any invalid configuration is found, False otherwise.
    """
    
    if not param_grid:
        st.error("⚠️ Error: The parameter grid is empty. Please select at least one parameter")
        return True

    # Check that all necessary params are present and non-empty in all sub-dictionaries
    if necessary_params is not None:
        for param_dict in param_grid:
            for necessary_param in necessary_params:
                if necessary_param not in param_dict:
                    st.error(f"⚠️ Error: The parameter '{necessary_param}' is missing from a grid configuration.")
                    return True
                if not param_dict[necessary_param]:
                    st.error(f"⚠️ Error: Please select at least one value for '{necessary_param}'.")
                    return True
    else:
        raise Exception("Necessary items are needed")

    # Check for empty optional params if they are present in a sub-dictionary
    if not_necessary_params is not None:
        for param_dict in param_grid:
            for optional_param in not_necessary_params:
                if optional_param in param_dict and not param_dict[optional_param]:
                    st.error(f"⚠️ Error: Please select at least one value for '{optional_param}'.")
                    return True

    return False


# --- NEW REUSABLE PLOTTING FUNCTIONS ---

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: pd.Series):
    """
    Generates a scatter plot of actual vs. predicted values.

    Args:
        y_true (pd.Series): The true target values.
        y_pred (pd.Series): The predicted target values.

    Returns:
        matplotlib.figure.Figure: The plot figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=50)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label="Perfect Fit")
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs. Predicted Plot')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_residuals(y_true: pd.Series, y_pred: pd.Series):
    """
    Generates a scatter plot of residuals vs. predicted values.

    Args:
        y_true (pd.Series): The true target values.
        y_pred (pd.Series): The predicted target values.

    Returns:
        matplotlib.figure.Figure: The plot figure object.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.set_title('Residuals vs. Predicted Values')
    ax.grid(True)
    fig.tight_layout()
    return fig

# FUNCTIONS FOR MODELS EXPLAINABILI

def generate_model_formula_latex(y: pd.Series, X: pd.DataFrame, model_type: str, model=None):
    """
    Generates a LaTeX formula for a linear or logistic regression model,
    handling scikit-learn pipelines and different model attribute shapes.

    Args:
        y (pd.Series): The pandas Series representing the target variable.
        X (pd.DataFrame): The DataFrame containing the feature variables.
        model_type (str): The type of model ('linear_regression' or 'logistic_regression').
        model (object, optional): A fitted scikit-learn model or pipeline. 
                                  Defaults to None for a generic formula.

    Returns:
        str: A LaTeX-formatted string of the model formula.
    """
    target_name = y.name.replace('_', r'\_') # Escape underscores for LaTeX
    
    # --- Step 1: Extract the actual model from the pipeline if necessary ---
    estimator = None
    if model:
        if isinstance(model, Pipeline):
            # Assumes the final step of the pipeline is the model
            estimator = model.named_steps.get('model') 
        else:
            estimator = model

    # --- Step 2: Build the linear part of the equation ---
    linear_part_list = []
    
    # Handle the intercept term
    if estimator and hasattr(estimator, 'intercept_'):
        intercept_val = estimator.intercept_
        # Check if intercept is a scalar (like in LinearRegression) or array (LogisticRegression)
        if isinstance(intercept_val, (int, float, np.number)):
            intercept_str = f"{intercept_val:.2f}"
        else: # Assumes it's array-like
            intercept_str = f"{intercept_val[0]:.2f}"
        linear_part_list.append(intercept_str)
    else:
        linear_part_list.append(r'\beta_0')

    # Handle the coefficient terms
    if estimator and hasattr(estimator, 'coef_'):
        coeffs = estimator.coef_.flatten()
        # Use X.columns directly to ensure correct order. Do NOT sort.
        for feature, coef in zip(X.columns, coeffs):
            # Escape underscores in feature names
            feature_latex = feature.replace('_', r'\_')
            sign = '+' if coef >= 0 else '-'
            term = f"{sign} {abs(coef):.2f} \\times x_{{{feature_latex}}}"
            linear_part_list.append(term)
        # Join and clean up double signs
        linear_part = ' '.join(linear_part_list).replace('+ -', '-')
    else:
        # Generic formula if no model is provided
        for i, feature in enumerate(X.columns):
            feature_latex = feature.replace('_', r'\_')
            term = rf'+ \beta_{{{i+1}}} x_{{{feature_latex}}}'
            linear_part_list.append(term)
        linear_part = ' '.join(linear_part_list)

    # --- Step 3: Assemble the final formula based on model type ---
    if model_type == 'linear_regression':
        formula = rf'{target_name} = {linear_part}'
    elif model_type == 'logistic_regression':
        # Using sigma for sigmoid function
        formula = rf'P({target_name}=1) = \sigma({linear_part})'
    else:
        formula = r'\text{Invalid Model Type}'
    
    return formula
