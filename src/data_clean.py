import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from dateutil.parser import parse as dateparse
from dotenv import load_dotenv
import streamlit as st
import re 
import time

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
You can only use pandas, numpy, math and re.
Never wrap the output in a markdown format.

## Output format
You only output python code, without imports (they are already present).
Your output will be executed with exec(your_answer) in a restricted environment, so code accordingly.
If the user request is to generate more than one new column, simply reply with ```python'raise Exception("You may only generate one column")'```
"""

def  ask_llm_data_clean(df_name, column_name, request, connectivity = 'local'):
    """Call a local LLM endpoint (e.g., Ollama) to generate a Pandas snippet
    that produces a Series named `lm_transformed_column`.

    SECURITY NOTE: Executing code from an LLM is potentially unsafe.
    In this app we execute in a *restricted* local namespace (see below),
    not in globals, and we only expose df/np/pd/re.

    Returns: the raw code string (for auditability) and the produced Series.
    """
    
    user_prompt = f"""# Request
This is the origin dataframe name: {df_name}
The column you must use to generate the new series: {column_name}
My request: \n{request}
"""

    match connectivity:
        
        case "api":
            model_name = 'gemini-2.5-flash'
            load_dotenv() # Verify to have a GEMINI_API_KEY in your .env
            try:
                client = genai.Client()
                
                # Sample call for Gemini API
                response = client.models.generate_content(
                    model= model_name,
                    contents= request,
                    config=types.GenerateContentConfig(
                        thinking_config = types.ThinkingConfig(thinking_budget=0), # Disables thinking
                        system_instruction = security_prompt
                    ),)
                
                if int(response.text) != 0:
                    time.sleep(3)
                    response = client.models.generate_content(
                    model= model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        thinking_config = types.ThinkingConfig(thinking_budget=0), # Disables thinking
                        system_instruction = coder_data_cleaner_system_prompt
                    ),) 
                    code = response.text
                
                else:
                    code = "print('FUCK YOU BITCH')"
                    
            except ConnectionError as e:
                print(f"API connection not working:\n {e}")
                

    
        case "local": # Using OLLAMA for local testing
            import requests  # lazy import so the app still runs if requests is missing
            url = "http://localhost:11434/api/generate"
            
            full_prompt = f"""{coder_data_cleaner_system_prompt}
            {user_prompt}
            """

            data = {
                "model": "qwen2.5-coder:3b",  # Replace with your model name
                "prompt": full_prompt,
                "stream": False,
            }

            try:
                time.sleep(3)
                response = requests.post(url, json=data, timeout=60)
                response.raise_for_status()
                payload = response.json()
                code = payload.get("response", "")

            except ConnectionError as e:
                print(f"API connection not working:\n {e}")

    # Remove any fenced code blocks if present (``` or ```python)
    code = re.sub(r"^```(?:python)?\s*", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"\s*```$", "", code.strip(), flags=re.IGNORECASE)

    return code  # code returned; series will be built where we have access to df

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
    if pd.api.types.is_float_dtype(ser):
        return "Continuous"
    elif pd.api.types.is_integer_dtype(ser):
        if nunique == 2:
            return "Binary"
        elif 2 < nunique <= 20:
            return "Categorical"
        else:
            return "Continuous"
    elif pd.api.types.is_object_dtype(ser):
        unique_vals = ser.dropna().unique()
        if nunique == 2:
            return "Boolean" if is_bool_like(unique_vals) else "Binary"
        elif 2 < nunique <= 20:
            return "Categorical"
        elif nunique > 20:
            return "Descriptive"
        else:
            return "Categorical"
    elif pd.api.types.is_datetime64_any_dtype(ser):
        return "Datetime"
    else:
        return "Other"
