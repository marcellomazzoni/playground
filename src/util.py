import streamlit as st 
import pandas as pd

# ========================= LLM availability + dynamic radio =========================
# Paste this somewhere near your other utils (before you render the radios).
# It detects whether an Ollama server is reachable and whether the requested model is installed.
# Then it builds a radio whose options include "Ask LLM" ONLY when available.

import os
import streamlit as st

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


def show_centered_matplotlib(fig, width_ratio=2.4):
    # left : middle : right ratios — tweak width_ratio to change middle width
    left_ , mid, right_ = st.columns([1, width_ratio, 1])
    with mid:
        st.pyplot(fig, use_container_width=True)
        
        
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



def show_home_page(background_image_path):
    """
    Displays the home page with a background image, title, description, and a button to proceed.

    Args:
        background_image_path (str): The path to the background image.
    """
    # NOTE: The user should provide the actual path to the background image.
    # For now, we'll use a placeholder.
    # Example: set_background('assets/background.png')

    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("{background_image_path}");
             background-attachment: fixed;
             background-size: cover
         }}
         .bottom-right-button {{
            position: fixed;
            bottom: 20px;
            right: 20px;
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

    st.title("Dataset analysis playground")
    st.write("This application helps you analyze your dataset. You can clean your data, visualize it, and train machine learning models.")

    # "Let's go" button in the bottom right
    st.markdown('<div class="bottom-right-button">', unsafe_allow_html=True)
    if st.button("Let's go!"):
        st.session_state.home_page_complete = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)