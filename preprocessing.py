# Streamlit – Lightweight Data Cleaning UI (fixed & commented)
# ----------------------------------------------------------------------------
# This file contains:
#   • BUGFIXES – marked with "BUGFIX:" comments where I changed behavior
#   • SUGGESTIONS – marked with "SUGGESTION:" for ideas you may want to add/modify
#   • EXPLANATIONS – comments describing what each block does
# ----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import os
import numpy as np
import re
from src.util import action_radio_for_column
from sklearn.impute import KNNImputer
from src.data_clean import get_autotype, ask_llm_data_clean, iqr_outlier_percent, infer_datetime_format
# import requests  # will import lazily inside ask_llm to avoid import errors if requests not installed

# ----------------------------------------------------------------------------
# UI – Reset and title
# ----------------------------------------------------------------------------

# SUGGESTION: enable wide layout for better tables (must be at top of script normally)
st.set_page_config(layout="wide")

# Top-right reset button to clear state and rerun
a = st.columns([8, 1])
with a[1]:
    if st.button("Reset All"):
        st.session_state.clear()
        st.rerun()

st.title("Data Analysis Tool – Base Cleaning")

# ----------------------------------------------------------------------------
# Session state bootstrap
# ----------------------------------------------------------------------------

for key, default in [
    ("dataframe", None),
    ("target", None),
    ("confirmed", False),
    ("uploaded_file", None),
    ("last_action", None),
    ("last_action_general", None),
    ("llm_output", None),
    ("llm_new_series", None),
    ("var_to_change", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

if st.session_state["dataframe"] is None:
    upload_option = st.radio("Choose input method:", ("Upload CSV")) #, "Select from folder"))

    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file:
            try:
                dataframe = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
            else:
                st.session_state["uploaded_file"] = uploaded_file
                st.session_state["dataframe"] = dataframe
                st.success("File successfully uploaded!")

# ----------------------------------------------------------------------------
# Main UI once we have a DataFrame
# ----------------------------------------------------------------------------

if st.session_state["dataframe"] is not None:
    df: pd.DataFrame = st.session_state["dataframe"]

    # Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.experimental_data_editor(df.head(20))

    # Type detection – recomputed on each run (cheap and consistent)
    autotype_dict = {col: get_autotype(df[col]) for col in df.columns}
    dtype_dict = {col: str(df[col].dtype) for col in df.columns}

    # Column families for summary blocks
    numeric_cols = [
        col
        for col, t in autotype_dict.items()
        if t in ["Continuous", "Categorical", "Binary"] and pd.api.types.is_numeric_dtype(df[col])
    ]
    string_cols = [
        col
        for col, t in autotype_dict.items()
        if t in ["Boolean", "Binary", "Categorical", "Descriptive"] and pd.api.types.is_object_dtype(df[col])
    ]
    datetime_cols = [
        col for col, t in autotype_dict.items() if t == "Datetime"
        ]

    # --- Summary tables ---
    numeric_summary = []
    for col in numeric_cols:
        ser = df[col]
        vartype = autotype_dict[col]
        numeric_summary.append(
            {
                "Variable": col,
                "Type": dtype_dict[col],
                "AutoType": vartype,
                "Min": ser.min(),
                "Max": ser.max(),
                "Mean": ser.mean(),
                "Median": ser.median(),
                "% Missing": round(100 * ser.isna().sum() / len(ser), 2),
                "# Unique": ser.nunique(dropna=True),
                "% Outliers": iqr_outlier_percent(ser) if vartype == "Continuous" else "",
            }
        )
    numeric_summary_df = pd.DataFrame(numeric_summary)

    string_summary = []
    for col in string_cols:
        ser = df[col]
        string_summary.append(
            {
                "Variable": col,
                "Type": dtype_dict[col],
                "AutoType": autotype_dict[col],
                "# Unique": ser.nunique(dropna=True),
                "Most Frequent": ser.mode().iloc[0] if not ser.mode().empty else "",
                "% Missing": round(100 * ser.isna().sum() / len(ser), 2),
            }
        )
    string_summary_df = pd.DataFrame(string_summary)

    datetime_summary = []
    for col in datetime_cols:
        ser = df[col]
        fmt = infer_datetime_format(ser)
        datetime_summary.append(
            {
                "Variable": col,
                "Type": dtype_dict[col],
                "AutoType": "Datetime",
                "Format (guess)": fmt,
                "Min": ser.min(),
                "Max": ser.max(),
                "% Missing": round(100 * ser.isna().sum() / len(ser), 2),
                "# Unique": ser.nunique(dropna=True),
            }
        )
    datetime_summary_df = pd.DataFrame(datetime_summary)

    # --- Show tables ---
    st.markdown("---")
    st.title("Current Variable Type Summary")

    show_block_numeric = st.toggle("Show Numeric Variables Summary", key="toggle_show_block_num")
    if show_block_numeric and not numeric_summary_df.empty:
        st.subheader("Numeric Variables")
        st.dataframe(numeric_summary_df, use_container_width=True, hide_index=True)

    show_block_string = st.toggle("Show String Variables Summary", key="toggle_show_block_string")
    if show_block_string and not string_summary_df.empty:
        st.subheader("String Variables")
        st.dataframe(string_summary_df, use_container_width=True, hide_index=True)

    show_block_datetime = st.toggle("Show Datetime Variables Summary", key="toggle_show_block_datetime")
    if show_block_datetime and not datetime_summary_df.empty:
        st.subheader("Datetime Variables")
        st.dataframe(datetime_summary_df, use_container_width=True, hide_index=True)

    # ---- Interactive cleaning tool ----
    st.markdown("---")
    st.title("Data Cleaning Tool")
    
    summary_df = pd.concat(
        [numeric_summary_df, string_summary_df, datetime_summary_df], ignore_index=True
    )

    st.subheader("General actions:")
    
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
                st.rerun()
            else:
                st.warning("No variables selected for dropping.")
        
    with st.expander("Drop duplicates"):
        subset_cols = st.multiselect("Subset of columns", list(df.columns))
        keep_first = st.checkbox("Keep first occurrence", value=True)
        if st.button("Apply", key="apply_drop_duplicates"):
            before = len(df)
            df.drop_duplicates(subset=subset_cols if subset_cols else None, keep="first" if keep_first else False, inplace=True)
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
                new_names = list(df.columns)
                proposed = {old: new for old, new in rename_dict.items()}
                # Collision check
                collision = any(new in set(df.columns) - {old} for old, new in proposed.items())
                if collision:
                    st.error("Lowercasing would create duplicate column names. Rename or drop manually first.")
                else:
                    df.rename(columns=proposed, inplace=True)
                    st.session_state.last_action_general = f"Lowercased: {list(proposed.values())}"
                    st.rerun()

    # ------------------------ Single-variable actions ------------------------
    st.subheader("Single-variable actions:")
    
    if st.session_state.last_action is not None:
        st.success(f"""{st.session_state.last_action}""")
        st.session_state.last_action = None
        st.session_state.var_to_change = None
    
    variable_choices = [""]+[f"{row.Variable} ({row.AutoType})" for _, row in summary_df.iterrows()]
    variable = st.selectbox("Select variable to transform", variable_choices, index=0, key="var_to_change")
    
    if  variable:
        col = variable.split(" (")[0]
        vartype = autotype_dict[col]
        dtype = dtype_dict[col]
        ser = df[col]

        # Summary for selected variable
        st.markdown("**Summary information for selected variable:**")
        summary_row = summary_df[summary_df["Variable"] == col]
        st.dataframe(summary_row, use_container_width=True, hide_index=True)

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
                        # st.session_state.var_to_change = ""
                        st.rerun()

            if action == "Manage outliers":
                method = st.selectbox("Method", ["Cap to 1.5 IQR bounds", "Cap to percentile bounds"])
                if method == "Cap to 1.5 IQR bounds":
                    if st.button("Apply", key=f"apply_outliers_{col}"):
                        q1, q3 = ser.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        df[col] = ser.clip(lower, upper)
                        st.session_state.last_action = f"{col} outliers capped to 1.5*IQR"
                        # st.session_state.var_to_change = ""
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
                        st.rerun()

            elif action == "Manage missing values":
                imp_method = st.selectbox("Impute using", ["Mean", "Median", "KNN"])
                    
                if imp_method == "Mean":
                    if st.button("Apply", key=f"apply_missing_mean_{col}"):
                        df[col] = ser.fillna(ser.mean())
                        st.session_state.last_action = f"Mean imputation on {col}"
                        st.rerun()
                    
                elif imp_method == "Median":
                    if st.button("Apply", key=f"apply_missing_median_{col}"):
                        df[col] = ser.fillna(ser.median())
                        st.session_state.last_action = f"Median imputation on {col}"
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
                    value="Change the column from string to numeric by removing the $ dollar sign and keeping its numbers only",
                    key=f"llm_req_{col}"
                )

                # Submit button for LLM request
                if st.button("Submit", key=f"llm_submit_{col}"):
                    with st.spinner("Generating code..."):
                        code =  ask_llm_data_clean("df", col, llm_request)# connectivity ="api")
                        if code:
                            st.session_state.llm_code = code
                            st.markdown("**LLM-proposed code:**")
                            st.code(code, language="python")
                            
                            # Execute in isolated namespace
                            local_ns = {"df": df.copy(), "np": np, "pd": pd, "re": re}
                            try:
                                exec(code, {}, local_ns)
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
                    }).drop_duplicates().sample(
                        n=min(10, len(df)), 
                        random_state=42
                    ).reset_index(drop=True)
                    
                    # Show comparison
                    st.dataframe(comparison_df, use_container_width=True)
                    
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
                                    st.rerun()
                                    
                        elif choice == "Replace Original":
                            if st.button("✅ Confirm Replace", key=f"llm_replace_{col}"):
                                df[col] = st.session_state.llm_new_series
                                st.session_state.last_action = f"Replaced '{col}' with transformed data"
                                st.session_state.llm_new_series = None
                                st.session_state.llm_code = None
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
                        st.rerun()

            if action == "Impute Missing Values":
                st.info("Imputation will use the modal class value.")
                if st.button("Apply", key=f"apply_mode_{col}"):
                    mode = ser.mode().iloc[0] if not ser.mode().empty else None
                    df[col] = ser.fillna(mode)
                    st.session_state.last_action = f"Mode imputation applied to {col}"
                    st.rerun()

            elif action == "Label Encoding":
                if st.button("Apply", key=f"apply_label_{col}"):
                    uniques = ser.dropna().unique()
                    mapping = {v: i for i, v in enumerate(uniques)}
                    df[col] = ser.map(mapping)
                    st.session_state.last_action = f"Label encoding applied to {col}"
                    st.rerun()

            elif action == "One-hot Encoding":
                dummies_preview = pd.get_dummies(ser, prefix=col)
                st.markdown("**Preview of columns that would be created:**")
                st.write(", ".join(map(str, dummies_preview.columns)))
                if st.button("Apply", key=f"apply_ohe_{col}"):
                    df.drop(columns=[col], inplace=True)
                    for new_col in dummies_preview.columns:
                        df[new_col] = dummies_preview[new_col]
                    st.session_state.last_action = f"One-hot encoding applied to {col}"
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
                    value="Change the column from string to numeric by removing the $ dollar sign and keeping its numbers only",
                    key=f"llm_req_{col}"
                )
            
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
                    value="Change the column from string to numeric by removing the $ dollar sign and keeping its numbers only",
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
                st.rerun()
                
    # ----------------------------------------------------------------------------
    # Target selection (numeric continuous only for now)
    # ----------------------------------------------------------------------------
    st.markdown("---")
    st.title("Target Variable Selection")
    # Eligible targets (you can tweak this)
    
    show_block_numeric_y = st.toggle("Show Eligible Target Variables Summary", key="toggle_show_block_num_y")
    if show_block_numeric_y and not numeric_summary_df.empty:
        st.subheader("Currently Numeric Variables")
        st.dataframe(numeric_summary_df[["Variable", "AutoType", "Type"]], use_container_width=False, hide_index=True)
        
    eligible_targets = [c for c in df.columns 
                        if autotype_dict[c] in ("Continuous", "Binary", "Categorical", "Boolean")]
    
    target = st.selectbox("Select the target variable:", [""] + eligible_targets)
    
    if target:
        st.session_state["target"] = target
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
            
            
    if st.button("Confirm Dataset and Target Variable"):
        if st.session_state.get("target") and st.session_state.get("problem_type"):
            st.session_state["confirmed"] = True
            st.rerun()
            st.success("Dataset and target confirmed! Go to a model page from the sidebar.")
        else:
            st.error("Please select a valid target and confirm the problem type.")

    st.markdown("---")
    # ----------------------------------------------------------------------------
    # SUGGESTION: Download the cleaned dataset
    # ----------------------------------------------------------------------------
    with st.expander("Export cleaned dataset"):
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv")

# END OF FILE
