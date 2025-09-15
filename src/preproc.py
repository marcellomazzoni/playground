import numpy as np
import pandas as pd
import seaborn as sns
from groq import Groq
from google import genai
from google.genai import types
from dateutil.parser import parse as dateparse
from dotenv import load_dotenv
import streamlit as st
import re 
import time
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import gaussian_kde
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_halving_search_cv #needed for halving grid
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.util import action_radio_for_column, show_centered_plot, show_plot_and_metrics, get_numeric_x_and_y_from_df

# ----------------------------------------------------------------------------
# Helper: LLM code generator for column-wise transformations
# ----------------------------------------------------------------------------
security_prompt = f"""
Your goal is to ensure that the request that the user inputs is in line with the task of cleaning a column of a dataframe.

RETURN '0' if:
    - The REQUEST is not related in any way to data processing or cleaning a specific column or dataframe.
    - The REQUEST creates code that is harmful to local storage or computer.
    - The REQUEST contains pre-written code of any language.
    
RETURN '1' otherwise
"""

coder_data_cleaner_system_prompt = """
# Role
You are a master in writing python code, with the goal of performing data cleaning

# Instructions
You create a python snippet, which is your only output.
Your snippet will address the requested column of the dataframe, only performing what is asked.
DO NOT CHANGE the original dataframe and column of interest, apply what requested on a newly generated series.
IMPORTANT: The output series MUST be named exactly "lm_transformed_column"
Import the libraries you use.
Never wrap the output in a markdown format.

## Output format
You only output python code.
Your output will be executed with exec(your_answer) in a restricted environment, so code accordingly.
If the user request is to generate more than one new column, simply reply with ```python'raise Exception("You may only generate one column")'```
"""

def ask_llm_data_clean(df, df_name, column_name, request, connectivity='local', model=None):
    """
    Call a local or API LLM endpoint to generate a Pandas snippet
    that produces a Series named `lm_transformed_column`.

    When using connectivity='api', the model parameter must be in the
    format 'provider/model_name' (e.g., 'gemini/gemini-1.5-flash' or 'groq/llama3-8b-8192').

    SECURITY NOTE: Executing code from an LLM is potentially unsafe.
    In this app we execute in a *restricted* local namespace (see below),
    not in globals, and we only expose df/np/pd/re.

    Returns: the raw code string (for auditability) and the produced Series.
    """
    user_prompt = f"""# Request
This is the origin dataframe name: {df_name}
The column you must use to generate the new series: {column_name}
Other columns: {df.columns}
My request: \n{request}
"""

    code = "" # Initialize code variable

    match connectivity:
        case "api":
            # --- API Provider and Model Detection ---
            if not model or '/' not in model:
                raise ValueError("For API connectivity, a model must be specified in the 'provider/model_name' format.")

            provider = st.session_state['llm_choice']
            model_name = model
            load_dotenv()  # Load GEMINI_API_KEY or GROQ_API_KEY from .env

            try:
                match provider.lower():
                    case "gemini":
                        client = genai.Client()
                        # 1. Security check call
                        security_response = client.models.generate_content(
                            model=model_name,
                            contents=request, # Check the raw request
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                                system_instruction=security_prompt
                            ),
                        )

                        if security_response.text and security_response.text.strip() == '1':
                            time.sleep(3)
                            # 2. Main code generation call
                            main_response = client.models.generate_content(
                                model=model_name,
                                contents=user_prompt,
                                config=types.GenerateContentConfig(
                                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                                    system_instruction=coder_data_cleaner_system_prompt
                                ),
                            )
                            code = main_response.text
                        else:
                            code = "print('Security check failed: Request is not valid.')"

                    case "groq":
                        client = Groq() # API key is read from GROQ_API_KEY env var
                        # 1. Security check call
                        security_response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": security_prompt},
                                {"role": "user", "content": request} # Check the raw request
                            ]
                        )
                        security_result = security_response.choices[0].message.content
                        
                        if security_result and security_result.strip() == '1':
                            time.sleep(3)
                            # 2. Main code generation call
                            main_response = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": coder_data_cleaner_system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ]
                            )
                            code = main_response.choices[0].message.content
                        else:
                            code = "print('Security check failed: Request is not valid.')"

                    case _:
                        raise ValueError(f"Unsupported provider: '{provider}'. Please use 'gemini' or 'groq'.")

            except Exception as e:
                print(f"API connection or call failed for '{provider}':\n {e}")
                code = f"print('API call failed: {e}')"


        case "local": # Using OLLAMA for local testing
            import requests # lazy import so the app still runs if requests is missing
            url = "http://localhost:11434/api/generate"
            full_prompt = f"{coder_data_cleaner_system_prompt}\n{user_prompt}"

            data = {
                "model": "qwen2:7b-instruct", # Replace with your model name
                "prompt": full_prompt,
                "stream": False,
            }

            try:
                time.sleep(3)
                response = requests.post(url, json=data, timeout=60)
                response.raise_for_status()
                payload = response.json()
                code = payload.get("response", "")
            except requests.exceptions.RequestException as e:
                print(f"Local connection not working:\n {e}")
                code = f"print('Local LLM call failed: {e}')"


    # --- Post-processing (common for all cases) ---
    # Remove any fenced code blocks if present (``` or ```python)
    if code:
        code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.MULTILINE)
        code = re.sub(r"\s*```$", "", code.strip(), flags=re.MULTILINE)

    return code
# ----------------------------------------------------------------------------
# Helper: Outlier share via IQR (numeric only)
# ----------------------------------------------------------------------------
def iqr_outlier_percent(series: pd.Series):
    if not pd.api.types.is_numeric_dtype(series):
        return np.nan
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    denom = series.notnull().sum()
    if denom == 0:
        return np.nan
    outlier_count = ((series < lower) | (series > upper)).sum()
    return round(100 * (outlier_count / denom), 2)

# ----------------------------------------------------------------------------
# Helper: Missing share for consistency
# ----------------------------------------------------------------------------
def missing_percent(series: pd.Series):
    
    return round(100 * series.isna().sum() / len(series), 2)

# ----------------------------------------------------------------------------
# Helper: Detect boolean-like sets of values (for object dtype columns)
# ----------------------------------------------------------------------------

def is_bool_like(values) -> bool:
    valset = set(str(v).strip().lower() for v in values if pd.notnull(v))
    bool_sets = [
        {"true", "false"}, {"t", "f"}, {"yes", "no"}, {"y", "n"}, {"1", "0"}
    ]
    return any(valset == s for s in bool_sets)

# ----------------------------------------------------------------------------
# Helper: Heuristic datetime format guess (for display only)
# ----------------------------------------------------------------------------

def infer_datetime_format(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return ""
    sample = str(non_null.iloc[0])
    patterns = [
        (r"^\d{4}-\d{2}-\d{2}$", "YYYY-MM-DD"),
        (r"^\d{8}$", "YYYYMMDD"),
        (r"^\d{2}/\d{2}/\d{4}$", "DD/MM/YYYY"),
        (r"^\d{4}/\d{2}/\d{2}$", "YYYY/MM/DD"),
        (r"^\d{2}-\d{2}-\d{4}$", "DD-MM-YYYY"),
        (r"^\d{4}\.\d{2}\.\d{2}$", "YYYY.MM.DD"),
        (r"^\d{2}\.\d{2}\.\d{4}$", "DD.MM.YYYY"),
    ]
    for regex, fmt in patterns:
        if re.match(regex, sample):
            return fmt
    try:
        d = dateparse(sample)
        return d.strftime("%Y-%m-%d %H:%M:%S")  # just an example parse display
    except Exception:
        return "Unknown"

# ----------------------------------------------------------------------------
# Helper: Auto-type assignment for UI logic
# ----------------------------------------------------------------------------
def get_autotype(ser: pd.Series) -> str:
    nunique = ser.nunique(dropna=True)
    dtype = pd.api.types.infer_dtype(ser)
    
    if dtype in ['floating', 'mixed-integer-float']:
        return "Continuous"
    
    elif dtype == 'integer':
        if nunique == 2:
            return "Binary"
        elif 2 < nunique <= 20:
            return "Categorical"
        else:
            return "Continuous"
            
    elif dtype in ['string', 'mixed']:
        unique_vals = ser.dropna().unique()
        if nunique == 2:
            return "Boolean" if is_bool_like(unique_vals) else "Binary"
        elif nunique > 20:
            return "Descriptive"
        else:
            return "Categorical"
            
    elif dtype in ['datetime64', 'datetime', 'date']:
        return "Datetime"
        
    elif dtype == 'boolean':
        return "Boolean"
        
    else:
        return dtype.capitalize()
    
class Summarizer():
    
    def __init__(self, df: pd.DataFrame):
        """Initializes the Grouper class with a DataFrame."""
        self.df = df
        
        # All of your attribute calculations now happen inside __init__
        # and reference self.df
        self.autotype_dict = {col: get_autotype(self.df[col]) for col in self.df.columns}
        self.dtype_dict = {col: str(self.df[col].dtype) for col in self.df.columns}
        
        self.numeric_cols = [
            col
            for col, t in self.autotype_dict.items()
            if t in ["Continuous", "Categorical", "Binary"] and pd.api.types.is_numeric_dtype(self.df[col])
        ]
        self.string_cols = [
            col
            for col, t in self.autotype_dict.items()
            if t in ["Boolean", "Binary", "Categorical", "Descriptive"] and pd.api.types.is_object_dtype(self.df[col])
        ]
        self.datetime_cols = [
            col for col, t in self.autotype_dict.items() if t == "Datetime"
        ]
        
        # create summaries separately
        numeric_df = self.numeric_summary() if self.numeric_cols else pd.DataFrame()
        string_df = self.string_summary() if self.string_cols else pd.DataFrame()
        datetime_df = self.datetime_summary() if self.datetime_cols else pd.DataFrame()
        
        # Finally concatenate only non-empty DataFrames
        dfs_to_concat = [df for df in [numeric_df, string_df, datetime_df] if not df.empty]
        self.summary_df = pd.concat(dfs_to_concat, ignore_index=True) if dfs_to_concat else pd.DataFrame()

    def numeric_summary(self):
        # --- Summary tables ---
        numeric_summary = []
        for col in self.numeric_cols:
            ser = self.df[col]
            vartype = self.autotype_dict[col]
            numeric_summary.append(
                {
                    "Variable": col,
                    "Type": self.dtype_dict[col],
                    "AutoType": vartype,
                    "Min": ser.min(),
                    "Max": ser.max(),
                    "Mean": ser.mean(),
                    "Median": ser.median(),
                    "% Missing": missing_percent(ser),
                    "# Unique": ser.nunique(dropna=True),
                    "% Outliers (± 1.5 IQR)": iqr_outlier_percent(ser) if vartype == "Continuous" else "",
                }
            )
        numeric_summary_df = pd.DataFrame(numeric_summary)
        return numeric_summary_df
    
    def string_summary(self):
        string_summary = []
        for col in self.string_cols:
            ser = self.df[col]
            string_summary.append(
                {
                    "Variable": col,
                    "Type": self.dtype_dict[col],
                    "AutoType":self.autotype_dict[col],
                    "# Unique": ser.nunique(dropna=True),
                    "Most Frequent": ser.mode().iloc[0] if not ser.mode().empty else "",
                    "% Missing": round(100 * ser.isna().sum() / len(ser), 2),
                }
            )
        string_summary_df = pd.DataFrame(string_summary)
        return string_summary_df
    
    def datetime_summary(self):
        datetime_summary = []
        for col in self.datetime_cols:
            ser = self.df[col]
            fmt = infer_datetime_format(ser)
            datetime_summary.append(
                {
                    "Variable": col,
                    "Type": self.dtype_dict[col],
                    "AutoType": "Datetime",
                    "Format (guess)": fmt,
                    "Min": ser.min(),
                    "Max": ser.max(),
                    "% Missing": missing_percent(ser),
                    "# Unique": ser.nunique(dropna=True),
                }
            )
        datetime_summary_df = pd.DataFrame(datetime_summary)
        return datetime_summary_df

class Processor(Summarizer):
    def __init__(self, df: pd.DataFrame):
       super().__init__(df)
       
    def general_actions(self):
        df = self.df
        if st.session_state.last_action_general:
            st.success(f"""{st.session_state.last_action_general}.  
        You can select a new, general, transformation.""")
            st.session_state.last_action_general = None

        # Batch drop logic
        with st.expander("Drop variables"):
            drop_vars = st.multiselect("Select variables to drop", options= list(df.columns))
            if st.button("Drop"):
                if drop_vars:
                    df.drop(columns=drop_vars, inplace=True, errors="ignore")
                    # Keep metadata in sync (recomputed on rerun anyway)
                    st.session_state.last_action_general = f"Dropped variables: {drop_vars}"
                    st.session_state['feature_importance_df'] = None
                    st.rerun()
                else:
                    st.warning("No variables selected for dropping.")
            
        with st.expander("Drop duplicates"):
            subset_cols = st.multiselect("Subset of columns", list(df.columns))
            keep_first = st.checkbox("Keep first occurrence", value=True)
            if st.button("Apply", key="apply_drop_duplicates"):
                before = len(df)
                df.drop_duplicates(subset=subset_cols if subset_cols else None, keep="first" if keep_first else False, inplace=True)
                st.session_state['feature_importance_df'] = None
                after = len(df)
                st.session_state.last_action_general = f"Dropped {before - after} duplicate rows"
                st.rerun()

        # Lowercase column names (safe rename)
        with st.expander("Lowercase variables names"):
            to_lowercase = st.multiselect("Select variables to lowercase", options= list(df.columns))
            if st.button("Apply", key="apply_lowercase_colnames"):
                if not to_lowercase:
                    st.info("Select at least one column to lowercase.")
                else:
                    rename_dict = {c: c.lower() for c in to_lowercase}
                    proposed = {old: new for old, new in rename_dict.items()}
                    # Collision check
                    collision = any(new in set(df.columns) - {old} for old, new in proposed.items())
                    if collision:
                        st.error("Lowercasing would create duplicate column names. Rename or drop manually first.")
                    else:
                        df.rename(columns=proposed, inplace=True)
                        st.session_state.last_action_general = f"Lowercased: {list(proposed.values())}"
                        st.session_state['feature_importance_df'] = None
                        st.rerun()

    def single_actions(self):
        df = self.df
        summary_df = self.summary_df
        autotype_dict = self.autotype_dict
        numeric_cols = self.numeric_cols
        
        if st.session_state.last_action is not None:
            st.success(f"""{st.session_state.last_action}""")
            st.session_state.last_action = None
            st.session_state.var_to_change = None
        
        variable = st.selectbox("Select variable to transform", 
                                [""]+[f"{row.Variable} ({row.AutoType})" for _, row in summary_df.iterrows()], 
                                index=0, key="var_to_change")
        
        if  variable:
            col = variable.split(" (")[0]
            vartype = autotype_dict[col]
            ser = df[col]

            # Summary for selected variable
            summary_row = summary_df[summary_df["Variable"] == col]
            st.dataframe(summary_row, width='stretch', hide_index=True)

            if st.toggle("Show some example observations", key=f"toggle_show_examples_{col}"):
                unique_vals = pd.Series(ser.dropna().unique())
                n = min(20, len(unique_vals))
                sample_obs = unique_vals.sample(n, random_state=42) if len(unique_vals) > n else unique_vals
                st.markdown("Sample values:")
                st.code("\n".join([str(v) for v in sample_obs]), language="text")

            # -------------------- Continuous --------------------
            if vartype == "Continuous":
                action = action_radio_for_column(col = col, coltype = vartype)

                if action == "Rename":
                    new_name = st.text_input("Rename variable", value=col)
                    if new_name != col and st.button("Rename"):
                        if new_name in df.columns:
                            st.error("A column with this name already exists.")
                        else:
                            df.rename(columns={col: new_name}, inplace=True)
                            st.session_state.last_action = f"Renamed {col} to {new_name}"
                            st.session_state['feature_importance_df'] = None
                            st.rerun()

                if action == "Manage outliers":
                    method = st.selectbox("Method", ["Cap to X * IQR bounds", "Cap to percentile bounds"])
                    if method == "Cap to X * IQR bounds":
                        IQR_mult = st.number_input('Inter-Quantile Range multiplier X', min_value=0.0, max_value=2.5, value=1.5, step=0.1, width=300)
                        if st.button("Apply", key=f"apply_outliers_{col}"):
                            q1, q3 = ser.quantile([0.25, 0.75])
                            iqr = q3 - q1
                            lower = q1 - (IQR_mult * iqr)
                            upper = q3 + (IQR_mult * iqr)
                            df[col] = ser.clip(lower, upper)
                            st.session_state.last_action = f"{col} outliers capped to 1.5*IQR"
                            st.session_state['feature_importance_df'] = None
                            st.rerun()
                    else:
                        lower_upper = st.slider(
                            "Select percentile bounds",
                            min_value=0,
                            max_value=100,
                            value=(1, 99),
                            step=1
                        )
                        if st.button("Apply", key=f"apply_percentile_{col}"):
                            lower = ser.quantile(lower_upper[0]/100)
                            upper = ser.quantile(lower_upper[1]/100)
                            df[col] = ser.clip(lower, upper)
                            st.session_state.last_action = f"{col} outliers capped at percentiles {lower}-{upper}"
                            st.session_state['feature_importance_df'] = None
                            st.rerun()

                elif action == "Manage missing values":
                    imp_method = st.selectbox("Impute using", ["Mean", "Median", "KNN"])
                        
                    if imp_method == "Mean":
                        if st.button("Apply", key=f"apply_missing_mean_{col}"):
                            df[col] = ser.fillna(ser.mean())
                            st.session_state.last_action = f"Mean imputation on {col}"
                            st.session_state['feature_importance_df'] = None
                            st.rerun()
                        
                    elif imp_method == "Median":
                        if st.button("Apply", key=f"apply_missing_median_{col}"):
                            df[col] = ser.fillna(ser.median())
                            st.session_state.last_action = f"Median imputation on {col}"
                            st.session_state['feature_importance_df'] = None
                            st.rerun()
                        
                    elif imp_method == "KNN":
                        # Hyperparameter: neighbors
                        k = st.slider("K (neighbors)", min_value=1, max_value=30, value=8, key=f"knn_k_{col}")

                        # Candidates are numeric columns excluding the target
                        candidate_features = [c for c in numeric_cols if c != col]
                        if len(candidate_features) < 1:
                            st.error("At least one other numeric feature is required for KNN imputation.")
                        else:
                            # If the target has no missing values, warn (still allow running if user insists)
                            if not df[col].isna().any():
                                st.info(f"Column '{col}' does not contain missing values; KNN imputation may be unnecessary.")

                            # Let user choose which features to use; defaults to all candidates
                            features = st.multiselect(
                                "Select features to use for imputation",
                                options=candidate_features,         
                                default=candidate_features,
                                help="These features are used to compute distances for nearest neighbors.",
                                key=f"knn_feats_{col}",
                            )

                            if st.button("Apply", key=f"apply_knn_{col}"):
                                if not features:
                                    st.error("Please select at least one feature (other than the target) for KNN imputation.")
                                else:
                                    # Build the matrix for imputation: features + target (target last)
                                    to_impute = df[features + [col]].copy()
                                    # Replace ±inf -> NaN so the imputer won't fail
                                    to_impute = to_impute.replace([np.inf, -np.inf], np.nan)
                                    # Ensure there are enough rows with at least one non-NaN feature
                                    rows_with_some_feature = (~to_impute[features].isna().all(axis=1)).sum()
                                    if rows_with_some_feature < 2:
                                        st.error("Not enough usable rows for KNN (need at least 2 rows with some non-missing feature values).")
                                        st.stop()
        
                                    X = to_impute[features].astype(float)
                                    
                                    mu = X.mean(axis=0, skipna=True)
                                    sigma = X.std(axis=0, ddof=0, skipna=True)
                                    sigma_safe = sigma.replace(0.0, 1.0) # Protect against zero std to avoid division by zero
                                    X_scaled = (X - mu) / sigma_safe  # NaNs preserved
                                    impute_matrix = pd.concat([X_scaled, to_impute[[col]]], axis=1)# Combine scaled features with the (unscaled) target as last column
                                    k_effective = min(k, max(1, rows_with_some_feature - 1)) # Cap k to available neighbor rows

                                    # Cap k to a safe value relative to available neighbor rows
                                    k_effective = min(k, max(1, rows_with_some_feature - 1))

                                    # Optional: if **all** target values are NaN, we cannot impute meaningfully
                                    if to_impute[col].isna().all():
                                        st.error(f"Column '{col}' has all values missing. KNN cannot impute the entire column without any observed targets.")
                                        st.stop()

                                    # Fit/transform with KNNImputer
                                    imputer = KNNImputer(n_neighbors=k_effective)
                                    imputed = imputer.fit_transform(to_impute)  # numpy array
                                    target_imputed = imputed[:, len(features)]  # target is last column

                                    # Safe assignment back: update ONLY where target was missing
                                    missing_mask = to_impute[col].isna().values
                                    if missing_mask.any():
                                        df.loc[to_impute.index[missing_mask], col] = target_imputed[missing_mask]

                                    # UX message + optional UI reset and rerun
                                    n_filled = int(missing_mask.sum())
                                    st.session_state.last_action = (
                                        f"KNN imputation on '{col}': filled {n_filled} missing value(s) "
                                        f"using {len(features)} feature(s), k={k_effective}."
                                    )
                                    st.session_state['feature_importance_df'] = None
                                    st.rerun()

                elif action == "Bucketize (discretize)":
                    buck_type = st.selectbox("Bucketize by", ["Quantiles", "Equal width"])
                    n_buckets = st.slider("Number of buckets", 2, 10, 4)
                    if st.button("Apply", key=f"apply_bucket_{col}"):
                        if buck_type == "Quantiles":
                            df[col] = pd.qcut(ser, n_buckets, labels=False, duplicates="drop")
                        else:
                            df[col] = pd.cut(ser, n_buckets, labels=False, duplicates="drop")
                        st.session_state.last_action = f"Bucketing applied to {col}, now in {n_buckets} groups"
                        st.session_state['feature_importance_df'] = None
                        st.rerun()

                elif action == "Ask LLM":
                    # Initialize session state for LLM results if not exists
                    if 'llm_new_series' not in st.session_state:
                        st.session_state.llm_new_series = None
                    if 'llm_code' not in st.session_state:
                        st.session_state.llm_code = None

                    llm_request = st.text_area(
                        label="""**Describe your request for the LLM**.  
    IMPORTANT: Avoid using transformations that may cause data or label leakage""",
                        placeholder="[EXAMPLE] Change the column from string to numeric by removing the $ dollar sign and keeping its numbers only",
                        key=f"llm_req_{col}"
                    )

                    # Submit button for LLM request
                    if st.button("Submit", key=f"llm_submit_continuous_{col}"):
                        with st.spinner("Generating code..."):
                            if st.session_state.llm_choice == "Ollama":
                                code =  ask_llm_data_clean(df=df, df_name="df", column_name= col, request= llm_request, model = st.session_state['ollama_model_name'])
                            elif st.session_state.llm_choice in ["Gemini","Groq"]:
                                code =  ask_llm_data_clean(df=df, df_name="df", column_name= col, request= llm_request, connectivity = "api", model = st.session_state['llm_api_model_name'])

                            if code:
                                st.session_state.llm_code = code
                                st.markdown("**LLM-proposed code:**")
                                st.code(code, language="python")

                                local_ns = {
                                            "df": df.copy(),  # It's good practice to pass a copy to avoid side effects
                                            "np": np,
                                            "pd": pd,
                                            "re": re
                                        }
                                try:
                                    # exec(code, {}, local_ns)
                                    exec(code, globals(), locals=local_ns)
                                    # Cerca la variabile di output
                                    new_series = local_ns.get("lm_transformed_column")
                                    
                                    if new_series is None:
                                        st.warning("⚠️ The generated code did not create 'lm_transformed_column'.")
                                    elif len(new_series) != len(df):
                                        st.error("❌ The transformed series length does not match the DataFrame.")
                                    else:
                                        st.session_state.llm_new_series = new_series
                                        st.success("✅ Transformation completed successfully!")
                                except Exception as e:
                                    st.error(f"❌ Code execution failed: {str(e)}")

                    # Show results and options if we have a transformed series
                    if st.session_state.llm_new_series is not None:
                        st.markdown("#### Preview of Transformations")
                        
                        # Create comparison DataFrame
                        comparison_df = pd.DataFrame({
                            f"Original ({col})": df[col],
                            "LLM suggestion": st.session_state.llm_new_series
                        }).sample(
                            n=min(10, len(df)), 
                            random_state=42
                        ).reset_index(drop=True)
                        
                        # Show comparison
                        st.dataframe(comparison_df, width='stretch')
                        
                        # Actions for the transformed data
                        action_col1, action_col2 = st.columns([2, 1])
                        with action_col1:
                            choice = st.radio(
                                "Choose what to do with the transformed data:",
                                ["Keep Both (new column)", "Replace Original", "Reject Changes"],
                                key=f"llm_action_choice_{col}",
                                horizontal=True
                            )

                        with action_col2:
                            if choice == "Keep Both (new column)":
                                new_col_name = st.text_input(
                                    "New column name",
                                    value=f"{col}_transformed",
                                    key=f"llm_new_col_name_{col}"
                                )
                                if st.button("✅ Confirm", key=f"llm_confirm_{col}"):
                                    if new_col_name in df.columns:
                                        st.error("Column name already exists!")
                                    else:
                                        df[new_col_name] = st.session_state.llm_new_series
                                        st.session_state.last_action = f"Added transformed data as '{new_col_name}'"
                                        st.session_state.llm_new_series = None
                                        st.session_state.llm_code = None
                                        st.session_state['feature_importance_df'] = None
                                        st.rerun()
                                        
                            elif choice == "Replace Original":
                                if st.button("✅ Confirm Replace", key=f"llm_replace_{col}"):
                                    df[col] = st.session_state.llm_new_series
                                    st.session_state.last_action = f"Replaced '{col}' with transformed data"
                                    st.session_state.llm_new_series = None
                                    st.session_state.llm_code = None
                                    st.session_state['feature_importance_df'] = None
                                    st.rerun()
                                    
                            elif choice == "Reject Changes":
                                if st.button("❌ Reject", key=f"llm_reject_{col}"):
                                    st.session_state.llm_new_series = None
                                    st.session_state.llm_code = None
                                    st.session_state.last_action = "Rejected LLM transformation"
                                    st.rerun()

            # -------------------- Categorical / Binary --------------------
            elif vartype in ["Categorical", "Binary"]:
                action = action_radio_for_column(col = col, coltype = vartype)
                
                if action == "Rename":
                    new_name = st.text_input("Rename variable", value=col)
                    if new_name != col and st.button("Rename"):
                        if new_name in df.columns:
                            st.error("A column with this name already exists.")
                        else:
                            df.rename(columns={col: new_name}, inplace=True)
                            st.session_state.last_action = f"Renamed {col} to {new_name}"
                            st.session_state['feature_importance_df'] = None
                            st.rerun()

                if action == "Impute Missing Values":
                    st.info("Imputation will use the modal class value.")
                    if st.button("Apply", key=f"apply_mode_{col}"):
                        mode = ser.mode().iloc[0] if not ser.mode().empty else None
                        df[col] = ser.fillna(mode)
                        st.session_state.last_action = f"Mode imputation applied to {col}"
                        st.session_state['feature_importance_df'] = None
                        st.rerun()

                elif action == "Label Encoding":
                    if st.button("Apply", key=f"apply_label_{col}"):
                        uniques = ser.dropna().unique()
                        mapping = {v: i for i, v in enumerate(uniques)}
                        df[col] = ser.map(mapping)
                        st.session_state.last_action = f"Label encoding applied to {col}"
                        st.session_state['feature_importance_df'] = None
                        st.rerun()

                elif action == "One-hot Encoding":
                    dummies_preview = pd.get_dummies(ser, prefix=col)
                    st.markdown("**Preview of columns that would be created:**")
                    st.write(",  ".join(map(str, dummies_preview.columns)))
                    if st.button("Apply", key=f"apply_ohe_{col}"):
                        df.drop(columns=[col], inplace=True)
                        for new_col in dummies_preview.columns:
                            df[new_col] = dummies_preview[new_col]
                        st.session_state.last_action = f"One-hot encoding applied to {col}"
                        st.session_state['feature_importance_df'] = None
                        st.rerun()
                        
                elif action == "Ask LLM":
                    # Initialize session state for LLM results if not exists
                    if 'llm_new_series' not in st.session_state:
                        st.session_state.llm_new_series = None
                    if 'llm_code' not in st.session_state:
                        st.session_state.llm_code = None

                    llm_request = st.text_area(
                        label="""**Describe your request for the LLM**.  
    IMPORTANT: Avoid using transformations that may cause data or label leakage""",
                        placeholder="[EXAMPLE] Change the column from string to numeric by removing the $ dollar sign and keeping its numbers only",
                        key=f"llm_req_{col}"
                    )

                    # Submit button for LLM request
                    if st.button("Submit", key=f"llm_submit_categorical_{col}"):
                        with st.spinner("Generating code..."):
                            if st.session_state.llm_choice == "Ollama":
                                code =  ask_llm_data_clean(df=df, df_name="df", column_name= col, request= llm_request, model = st.session_state['ollama_model_name'])
                            elif st.session_state.llm_choice in ["Gemini","Groq"]:
                                code =  ask_llm_data_clean(df=df, df_name="df", column_name= col, request= llm_request, connectivity = "api", model = st.session_state['llm_api_model_name'])

                            if code:
                                st.session_state.llm_code = code
                                st.markdown("**LLM-proposed code:**")
                                st.code(code, language="python")

                                local_ns = {
                                            "df": df.copy(),  # It's good practice to pass a copy to avoid side effects
                                            "np": np,
                                            "pd": pd,
                                            "re": re
                                        }
                                try:
                                    # exec(code, {}, local_ns)
                                    exec(code, globals(), locals=local_ns)
                                    # Cerca la variabile di output
                                    new_series = local_ns.get("lm_transformed_column")
                                    
                                    if new_series is None:
                                        st.warning("⚠️ The generated code did not create 'lm_transformed_column'.")
                                    elif len(new_series) != len(df):
                                        st.error("❌ The transformed series length does not match the DataFrame.")
                                    else:
                                        st.session_state.llm_new_series = new_series
                                        st.success("✅ Transformation completed successfully!")
                                except Exception as e:
                                    st.error(f"❌ Code execution failed: {str(e)}")

                    # Show results and options if we have a transformed series
                    if st.session_state.llm_new_series is not None:
                        st.markdown("#### Preview of Transformations")
                        
                        # Create comparison DataFrame
                        comparison_df = pd.DataFrame({
                            f"Original ({col})": df[col],
                            "LLM suggestion": st.session_state.llm_new_series
                        }).sample(
                            n=min(10, len(df)), 
                            random_state=42
                        ).reset_index(drop=True)
                        
                        # Show comparison
                        st.dataframe(comparison_df, width='stretch')
                        
                        # Actions for the transformed data
                        action_col1, action_col2 = st.columns([2, 1])
                        with action_col1:
                            choice = st.radio(
                                "Choose what to do with the transformed data:",
                                ["Keep Both (new column)", "Replace Original", "Reject Changes"],
                                key=f"llm_action_choice_{col}",
                                horizontal=True
                            )

                        with action_col2:
                            if choice == "Keep Both (new column)":
                                new_col_name = st.text_input(
                                    "New column name",
                                    value=f"{col}_transformed",
                                    key=f"llm_new_col_name_{col}"
                                )
                                if st.button("✅ Confirm", key=f"llm_confirm_{col}"):
                                    if new_col_name in df.columns:
                                        st.error("Column name already exists!")
                                    else:
                                        df[new_col_name] = st.session_state.llm_new_series
                                        st.session_state.last_action = f"Added transformed data as '{new_col_name}'"
                                        st.session_state.llm_new_series = None
                                        st.session_state.llm_code = None
                                        st.session_state['feature_importance_df'] = None
                                        st.rerun()
                                        
                            elif choice == "Replace Original":
                                if st.button("✅ Confirm Replace", key=f"llm_replace_{col}"):
                                    df[col] = st.session_state.llm_new_series
                                    st.session_state.last_action = f"Replaced '{col}' with transformed data"
                                    st.session_state.llm_new_series = None
                                    st.session_state.llm_code = None
                                    st.session_state['feature_importance_df'] = None
                                    st.rerun()
                                    
                            elif choice == "Reject Changes":
                                if st.button("❌ Reject", key=f"llm_reject_{col}"):
                                    st.session_state.llm_new_series = None
                                    st.session_state.llm_code = None
                                    st.session_state.last_action = "Rejected LLM transformation"
                                    st.rerun()
                
            # -------------------- Boolean --------------------
            elif vartype == "Boolean":
                st.info("Convert True/Yes to 1 and False/No to 0 (unrecognized values stay NaN).")
                if st.button("Apply", key=f"apply_bool_{col}"):
                    true_set = {"yes", "true", "y", "1"}
                    false_set = {"no", "false", "n", "0"}

                    # BUGFIX: previously any non-null non-true value became 0; now only explicit false maps to 0, others -> NaN
                    def _to_bin(x):
                        if pd.isna(x):
                            return np.nan
                        s = str(x).strip().lower()
                        if s in true_set:
                            return 1
                        if s in false_set:
                            return 0
                        return np.nan

                    df[col] = ser.map(_to_bin)
                    st.session_state.last_action = f"Boolean conversion applied to {col}"
                    st.session_state['feature_importance_df'] = None
                    st.rerun()

            # -------------------- Descriptive --------------------
            elif vartype == "Descriptive":
                action = action_radio_for_column(col = col, coltype = vartype)
                if action == "Ask LLM":
                    # Initialize session state for LLM results if not exists
                    if 'llm_new_series' not in st.session_state:
                        st.session_state.llm_new_series = None
                    if 'llm_code' not in st.session_state:
                        st.session_state.llm_code = None

                    llm_request = st.text_area(
                        label="""**Describe your request for the LLM**.  
    IMPORTANT: Avoid using transformations that may cause data or label leakage""",
                        placeholder="[EXAMPLE] Change the column from string to numeric by removing the $ dollar sign and keeping its numbers only",
                        key=f"llm_req_{col}"
                    )
                    
                st.info("No pre-made transformations available for descriptive variables yet.")
                
                # SUGGESTION: add text cleanup (strip, lower, remove extra spaces/punctuation) and length stats

            # -------------------- Datetime --------------------
            elif vartype == "Datetime":
                fmt_choices = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y%m%d"]
                new_fmt = st.selectbox("Choose output format (string)", fmt_choices)
                to_string = st.checkbox("Output as string (checked) or keep datetime dtype", value=True)
                if st.button("Apply", key=f"apply_dt_{col}"):
                    try:
                        parsed = pd.to_datetime(ser, errors="coerce")
                        if to_string:
                            df[col] = parsed.dt.strftime(new_fmt)
                        else:
                            df[col] = parsed
                        st.session_state.last_action = f"Datetime formatted/parsed for {col}"
                    except Exception as e:
                        st.error(f"Could not convert: {e}")
                    st.session_state['feature_importance_df'] = None
                    st.rerun()
                                        
class Selector(Summarizer):
    def __init__(self, df: pd.DataFrame):
       super().__init__(df)

    def target_and_problem_selection(self):
        df = self.df
        numeric_summary_df = self.numeric_summary()
        autotype_dict = self.autotype_dict
             
        # Eligible targets (can tweak this)
        show_block_numeric_y = st.toggle("Show Eligible Target Variables Summary", key="toggle_show_block_num_y")
        if show_block_numeric_y and not numeric_summary_df.empty:
            st.markdown("##### Currently Numeric Variables")
            st.dataframe(numeric_summary_df[["Variable", "AutoType", "Type"]], use_container_width=False, hide_index=True)

        eligible_targets = [c for c in df.columns 
                            if autotype_dict[c] in ("Continuous", "Binary", "Categorical", "Boolean")]
        
        target = st.selectbox("Select the target variable:", [""] + eligible_targets)
        
        if target and target != "":
            st.session_state['target'] = target
            t = autotype_dict[target]
            st.success(f"You selected {target}, which is a {t.lower()} variable.")
            
            # Simple mapping based on your auto types
            type_to_problem = {
                "Continuous":  "regression",
                "Binary":      "classification_binary",
                "Categorical": "classification_multi",
                "Boolean":     "classification",
            }
            detected = type_to_problem.get(t)

            if detected is None:
                st.error(f"Target '{target}' is {t} and isn’t supported as a target. "
                        "Transform it or pick another column.")
                st.session_state["problem_type"] = None
            else:
                st.session_state["problem_type"] = detected
        # else:
        #     st.session_state['target'] = None
    
    def feature_selection(self):
        df = self.df
        
        # Make sure a target is already chosen
        if "target" not in st.session_state or st.session_state['target'] is None:
            st.warning("Please select a target variable first in 'Target Variable Selection'")
            return

        target = st.session_state['target']

        # Candidate features = all columns except target
        candidate_vars = [c for c in df.columns if c != target]

        # Multiselect for user choice
        selected_features = st.multiselect(
            "Select variables to include in ML models:",
            options=candidate_vars,
            default=candidate_vars,  # can also set [] if you prefer starting empty
            key = "selected_ml_features"
        )

        # Always include target
        if selected_features:
            final_features = [target] + selected_features
            st.session_state["ml_dataset"] = df[final_features]
            st.success(
                f"Dataset prepared with {len(selected_features)} features + target. "
                f"Shape: {st.session_state['ml_dataset'].shape}"
            )            
            
            
    def univariate_analysis(self):

        # Skip if no target was selected
        if "target" not in st.session_state or st.session_state['target'] is None:
            st.warning("Please select a target variable first")
            return
            
        df = self.df
        summary_df = self.summary_df
        target = st.session_state['target']
        
        # Get non-target variables
        available_vars = [col for col in df.columns if col != target]
        # Variable selection
        selected_var = st.selectbox("Select variable for analysis:", [""] + available_vars)
        
        if selected_var:
            # Get variable type
            vartype = self.autotype_dict[selected_var]
            ser = df[selected_var]
            
            # Basic stats in expander
            if st.checkbox("Show Basic Statistics", key=f"toggle_show_stats_univar_{selected_var}"):
                summary_row = summary_df[summary_df["Variable"] == selected_var]
                st.markdown("**Summary**")
                st.dataframe(summary_row, width='stretch', hide_index=True)
                
                if vartype != "Continuous":
                    # Value counts in expander with minimum width
                    st.markdown("**Values Counts**")
                    value_counts = ser.value_counts(dropna=False)
                    st.dataframe(value_counts, width="content")
                                        

            if st.checkbox("Show Distribution Charts", key=f"toggle_show_distrib_chart_{selected_var}"):
                
                if vartype != "Continuous":
                    value_counts = ser.value_counts(dropna=False)
                    fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
                    fig.update_layout(title=f"Distribution of {selected_var}")
                    st.plotly_chart(fig, use_container_width=False)
                    
                else:
                    # Create the list of metrics in the required format
                    metrics_list = [
                        {'label': 'Kurtosis', 'value': f"{ser.kurtosis():.2f}"},
                        {'label': 'Skewness', 'value': f"{ser.skew():.2f}"},
                        {'label': 'Mean', 'value': f"{ser.mean():.2f}"},
                        {'label': 'Median', 'value': f"{ser.median():.2f}"},
                    ]
                    
                    plot_type = st.selectbox(
                        "Select plot type:",
                        ["Histogram", "Boxplot", "Q-Q Plot"],
                        key=f"plot_type_{selected_var}"
                    )
                    
                    if plot_type == "Histogram":
                        n_bins = st.slider("Number of bins", min_value=2, max_value=min(150, len(df)), value=30)

                        # Create histogram trace (counts on primary y-axis)
                        hist = go.Histogram(
                            x=ser,
                            nbinsx=n_bins,
                            name=selected_var,
                            histnorm=None,
                            opacity=0.8
                        )

                        # Density estimate (secondary y-axis)
                        kde = gaussian_kde(ser.dropna())
                        x_range = np.linspace(ser.min(), ser.max(), 500)
                        y_kde = kde(x_range)

                        density = go.Scatter(
                            x=x_range,
                            y=y_kde,
                            mode="lines",
                            name="Density",
                            yaxis="y2",
                        )

                        # Create figure with secondary y-axis
                        fig = go.Figure(data=[hist, density])
                        fig.update_layout(
                            title=f"Distribution of {selected_var}",
                            xaxis=dict(title=selected_var),
                            yaxis=dict(title="Count"),
                            yaxis2=dict(
                                title="Density",
                                overlaying="y",
                                side="right",
                                showgrid=False
                            ),
                            legend=dict(x=0.8, y=1.05, orientation="h")
                        )

                        show_plot_and_metrics(
                            fig,
                            width_ratio=2.4,
                            plot_type='plotly',
                            list_of_metrics=metrics_list
                        )
                    
                    elif plot_type == "Boxplot":
                        fig = go.Figure()
                        fig.add_trace(go.Box(y=ser, name=selected_var))
                        fig.update_layout(title=f"Boxplot of {selected_var}")
                        show_centered_plot(fig, width_ratio=2.4, plot_type='plotly')
                    
                    elif plot_type == "Q-Q Plot":
                        fig = go.Figure()
                        # Calculate theoretical quantiles
                        theoretical_q = np.quantile(np.random.normal(0, 1, len(ser)), np.linspace(0, 1, len(ser)))
                        sample_q = np.quantile(ser.dropna(), np.linspace(0, 1, len(ser.dropna())))
                        
                        # Calcola i limiti per la linea di riferimento
                        min_val = min(sample_q.min(), theoretical_q.min())
                        max_val = max(sample_q.max(), theoretical_q.max())
                        ref_line = np.array([min_val, max_val])
                        
                        fig.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode='markers', name='Data'))
                        fig.add_trace(go.Scatter(x=ref_line, y=ref_line, mode='lines', name='Reference Line'))
                        
                        fig.update_layout(
                            title=f"Q-Q Plot of {selected_var}",
                            xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles"
                        )
                        show_centered_plot(fig, width_ratio=2.4, plot_type='plotly')
                
    def bivariate_analysis(self):
        
         # Skip if no target was selected
        if "target" not in st.session_state or st.session_state['target'] is None: 
            st.warning("Please select a target variable first")
            return
            
        df = self.df
        target = st.session_state['target']
        x_df = df.drop(columns=[target])
        autotype_dict = self.autotype_dict
        
        # --- Correlation Matrix of Scatterplots ---
        st.markdown("**Comprehensive Correlation Matrix**")
        
        if autotype_dict[target] in ["Categorical", "Binary"]:
            choice = st.radio("Choose nature of correlation plots",
                    ["By Target", "Plain"],
                    key=f"bivariate_superv_unsuperv",
                    horizontal=True
                )
            
            selected_vars = st.multiselect("Select variables for matrix", 
                            x_df.columns.tolist(), 
                            default=x_df.columns[:len(x_df.columns)])

            match choice:
                case "By Target":
                    if selected_vars:
                        plot_vars = selected_vars
                        fig = sns.pairplot(df[plot_vars], 
                                         hue=target,  # Color by target variable
                                         diag_kind="hist",
                                         plot_kws={"s": 20, "alpha": 0.7},)
                        # st.pyplot(fig)
                        show_centered_plot(fig, width_ratio=5, plot_type='pyplot')
                    else:
                        st.warning("Please select at least one variable for analysis")

                case "Plain":
                    fig = sns.pairplot(df[selected_vars], 
                                       diag_kind="hist", 
                                       plot_kws={"s": 20, "alpha": 0.7})
                    # st.pyplot(fig)
                    show_centered_plot(fig, width_ratio=5, plot_type='pyplot')
                    
        else:
            selected_vars = st.multiselect("Select variables for matrix", 
                df.columns.tolist(), 
                default=df.columns[:min(4, len(df.columns))])

            fig = sns.pairplot(df[selected_vars], diag_kind="hist", plot_kws={"s": 20, "alpha": 0.7})
            st.pyplot(fig)

        # --- Custom Bivariate Section ---
        # st.markdown("**Custom Bivariate Plot**")
            # Continuous-Continuous: Hexbin + Pearson
            # Continuous + Categorical/Binary: Boxplot
            # Categorical-Categorical: Heatmap

    def multivariate_analysis(self):
        # Skip if no target was selected
        if "target" not in st.session_state or st.session_state['target'] is None: 
            st.warning("Please select a target variable first")
            return
        
        if st.checkbox("Show Variable Importance", key="toggle_show_var_imp"):
            with st.spinner("Getting variable importance via ML…"):
                if st.session_state['feature_importance_df'] is None:
                    df = self.df
                    target = st.session_state['target']
                    X, y = get_numeric_x_and_y_from_df(dataframe = df, target =  target)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    t = self.autotype_dict[target]            
                    # Simple mapping based on your auto types
                    type_to_problem = {
                            "Continuous":  "regression",
                            "Binary":      "classification_binary",
                            "Categorical": "classification_multi",
                            "Boolean":     "classification",
                        }
                    detected = type_to_problem.get(t)

                    # CLASSIFICATION
                    if detected in ["classification_multi", "classification_binary"]:
                        rf = RandomForestClassifier(random_state=42)
                        param_grid = [{
                            'n_estimators': [10,30,60,100,200],
                            'max_depth': [5, 10, 20, 30],
                            'min_samples_split': [2, 5, 10]
                        }]
                        
                        grid_search = HalvingGridSearchCV(
                            rf,
                            param_grid, 
                            cv=5,
                            scoring='accuracy',
                            refit = True,
                            return_train_score=False  # keep it lean; set True if you want to display train CV too
                        )
                        grid_search.fit(X_scaled, y)
                        # HEre is the variable importance
                        importances = grid_search.best_estimator_.feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)
                            
                    # REGRESSION
                    elif detected == "regression":
                        rf = RandomForestRegressor(random_state=42)
                        param_grid = [{
                            'n_estimators': [10,30,60,100,200],
                            'max_depth': [5, 10, 20, 30],
                            'min_samples_split': [2, 5, 10]
                        }]
                        
                        grid_search = HalvingGridSearchCV(rf,
                                                        param_grid,
                                                        cv=5,
                                                        scoring='neg_root_mean_squared_error', # <-- Use a single string here
                                                        refit=True, # refit will be set to the scoring metric
                                                        return_train_score=False
                                                        )
                        grid_search.fit(X_scaled, y)

                        # HEre is the variable importance
                        importances = grid_search.best_estimator_.feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)
                        
                        st.session_state['feature_importance_df'] = importance_df

                else: # already computed - cache
                    importance_df = st.session_state['feature_importance_df']
                    # Here is the horizontal bar chart of variable importance

                fig = px.bar(
                    title="Feature Importance from Random Forest - optimizing for predictive power",
                    data_frame = importance_df.sort_values(by='Importance', ascending=True),  # reverse order so largest is on top
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    text='Importance',  # Add value labels
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')  # Format and position labels
                fig.update_layout(
                    yaxis=dict(
                        categoryorder='total ascending',  # ensures largest is at top
                        tickfont=dict(size=18)           # make y labels bigger
                    ),
                    xaxis=dict(tickfont=dict(size=16)),  # make x labels bigger
                    bargap=0.5                           # make bars thinner
                )
                                    
            show_centered_plot(fig, width_ratio=5, plot_type='plotly')        

        if st.checkbox("Show 3D PCA Plot", key="toggle_show_3d_pca"):
            with st.spinner("Getting variable importance via ML…"):
                if "target" not in st.session_state or st.session_state['target'] is None:
                    st.warning("Please select a target variable first")
                    return
                df = self.df
                target = st.session_state['target']

                X, y = get_numeric_x_and_y_from_df(dataframe = df, target =  target)

                if X.shape[1] < 3: # Sanity check
                    st.warning("Need at least 3 numeric features for 3D PCA plot.")
                    return
                
                t = self.autotype_dict[target]            
                    # Simple mapping based on your auto types
                type_to_problem = {
                        "Continuous":  "regression",
                        "Binary":      "classification_binary",
                        "Categorical": "classification_multi",
                        "Boolean":     "classification",
                    }
                detected = type_to_problem.get(t)
            

                X_scaled = StandardScaler().fit_transform(X)
                pca = PCA(n_components=3)
                pca_result = pca.fit_transform(X_scaled)

                df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
                df_pca['target'] = y

                # Get loadings (components)
                loadings = pca.components_.T  # shape: (n_features, 3)
                loadings_scaled = pca.components_.T * np.sqrt(pca.explained_variance_)
                feature_names = X.columns

                # Create scatter plot for points
                fig = px.scatter_3d(
                    df_pca,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='target' if detected != "regression" else None,
                    title='3D PCA scores Plot' + (" Colored by Target" if detected != "regression" else "")
                )

                # Add vectors for each regressor
                for i, feature in enumerate(feature_names):
                    fig.add_trace(
                        go.Scatter3d(
                            x=[0, loadings_scaled[i, 0]], #*3],  # scale for visibility
                            y=[0, loadings_scaled[i, 1]], #*3],
                            z=[0, loadings_scaled[i, 2]], #*3],
                            mode='lines+text',
                            line=dict(color='red', width=4),
                            text=[None, feature],
                            textposition='top center',
                            name=feature,
                            showlegend=False
                        )
                    )
            cols = st.columns([2, 1])
            with cols[0]:
                show_centered_plot(fig, width_ratio=3, plot_type='plotly', height=650)
                # df_loadings = pd.DataFrame(loadings, index=X.columns, columns=['PC1', 'PC2', 'PC3'])
                # st.dataframe(df_loadings, width='content')
            # Loadings table (variables x principal components)
            with cols[1]:
                st.markdown("**PCA Raw Loadings**")
                df_loadings = pd.DataFrame(loadings, index=X.columns, columns=['PC1', 'PC2', 'PC3'])
                st.dataframe(df_loadings, width='content')



            










