import streamlit as st 
import pandas as pd
from streamlit_modal import Modal
# ========================= LLM availability + dynamic radio =========================
# Paste this somewhere near your other utils (before you render the radios).
# It detects whether an Ollama server is reachable and whether the requested model is installed.
# Then it builds a radio whose options include "Ask LLM" ONLY when available.
def choose_llm():
    # --- Modal initialization ---
    modal = Modal("⚙️ Settings", key="settings_modal", max_width=600)

    # Floating button
    if st.button("⚙️", key="open_settings", help="Open settings"):
        modal.open()

    # Session state to hold config
    if "llm_choice" not in st.session_state:
        st.session_state.llm_choice = "Ollama"

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""

    # --- Modal content ---
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

            # If Gemini is selected, ask for API key
            if choice == "Gemini":
                st.session_state.gemini_api_key = st.text_input(
                    "Enter Gemini API Key",
                    type="password",
                    value=st.session_state.gemini_api_key,
                    placeholder="sk-...",
                    help="Paste your Gemini API key here"
                )

            # Save / Close
            if st.button("Save & Close"):
                modal.close()
                st.success(f"Settings updated → Using {st.session_state.llm_choice}")


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
    st.markdown("---")
    # Get the results and put them into a DataFrame
    results = pd.DataFrame(grid_search.cv_results_)

    # Select and display the relevant columns, sorted by mean test score
    ranked_results = results[[
    'rank_test_score',
    'mean_test_score',
    'std_test_score',
    'params'
    ]].sort_values(by='rank_test_score')
    st.dataframe(ranked_results)
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

# FUNCTIONS FOR MODELS EXPLAINABILITY

def generate_model_formula_latex(y: pd.Series, X: pd.DataFrame, model_type: str, model=None):
    """
    Generates a LaTeX formula for a linear or logistic regression model.

    Args:
        y (pd.Series): The pandas Series representing the target variable.
        X (pd.DataFrame): The DataFrame containing the feature variables.
        model_type (str): The type of model ('linear_regression' or 'logistic_regression').
        model (object, optional): A fitted scikit-learn model with a `coef_` and `intercept_`
                                  attribute. Defaults to None.

    Returns:
        str: A LaTeX-formatted string of the model formula.
    """
    # Use the .name attribute to get the name of the target variable
    target_name = y.name
    features_sorted = sorted(X.columns.tolist())
    
    # Handle the intercept term
    if model and hasattr(model, 'intercept_'):
        intercept = f"{model.intercept_[0]:.2f}" if model.intercept_.size > 0 else "0"
        linear_part_list = [intercept]
    else:
        linear_part_list = [r'\beta_0']
    
    # Handle the coefficient terms
    if model and hasattr(model, 'coef_'):
        # Flatten coef_ if it's a 2D array (e.g., from LogisticRegression)
        coeffs = model.coef_.flatten()
        for i, feature in enumerate(features_sorted):
            sign = '+' if coeffs[i] >= 0 else ''
            term = f"{sign}{coeffs[i]:.2f}x_{{{feature}}}"
            linear_part_list.append(term)
        linear_part = ' '.join(linear_part_list).replace('+ -', '- ')
    else:
        for i, feature in enumerate(features_sorted):
            term = rf'\beta_{{{i+1}}} x_{{{feature}}}'
            linear_part_list.append(term)
        linear_part = '+'.join(linear_part_list).replace('+ -', '- ')


    if model_type == 'linear_regression':
        formula = rf'{target_name} = {linear_part}'
        
    elif model_type == 'logistic_regression':
        formula = rf'P({target_name}=1) = \sigma({linear_part})'
        
    else:
        formula = r'Invalid Model Type'
    
    return formula