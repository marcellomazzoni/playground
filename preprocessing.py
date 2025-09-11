# Streamlit – Lightweight Data Cleaning UI (fixed & commented)
# ----------------------------------------------------------------------------
# This file contains:
#   • BUGFIXES – marked with "BUGFIX:" comments where I changed behavior
#   • SUGGESTIONS – marked with "SUGGESTION:" for ideas you may want to add/modify
#   • EXPLANATIONS – comments describing what each block does
# ----------------------------------------------------------------------------
import code
import subprocess
import sys
import streamlit as st
import pandas as pd
from src.preproc import Summarizer, Processor, Selector

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

st.markdown("# Data Analysis Tool – Base Cleaning")

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
    ("var_to_change", None),
    ("feature_importance_df", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

if st.session_state['dataframe'] is None:
    upload_option = st.radio("Choose input method:", ["Upload CSV","Python script"]) #, "Select from folder"))
    match upload_option:
        case "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded_file:
                try:
                    dataframe = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
                else:
                    st.session_state["uploaded_file"] = uploaded_file
                    st.session_state['dataframe'] = dataframe
                    st.rerun()
                    st.success("File successfully uploaded!")

        case "Python script":
            install_code = st.text_area(label = "Your requirements, if any", 
                                        placeholder  = """Just like you would write a requirements file! Like this:
'''
streamlit
scikit-learn
scipy
'''
""",
                                        key="install_code")
            import_data_code = st.text_area(label = "Your code to retrieve data", 
                                        placeholder  = """Must start with imports, must end with a pandas DataFrame named 'starting_dataframe'
e.g.

import pandas as pd
[...]
starting_dataframe = a Pandas DataFrame object""",
                                        key="uploader_code")
            
            if st.button("Run all", key=f"run_import_code"):
                        with st.spinner("Installing dependencies..."):
                            # Split the install_code string into lines, filter out comments and empty lines
                            packages = [line.strip() for line in install_code.splitlines() if line.strip() and not line.startswith("#")]
                            if packages:
                                subprocess.run([sys.executable, "-m", "pip", "install", *packages])
                        with st.spinner("Running code..."):
                            local_ns = {"pd": pd}
                            exec(import_data_code, {}, local_ns)
                            if "starting_dataframe" in local_ns and isinstance(local_ns["starting_dataframe"], pd.DataFrame):
                                st.session_state['dataframe'] = local_ns["starting_dataframe"]
                                st.rerun()
                                st.success("Code ran successfully and DataFrame found!")
                            else:
                                st.error("Code did not produce a DataFrame named 'starting_dataframe'.")

# ----------------------------------------------------------------------------
# Main UI once we have a DataFrame
# ----------------------------------------------------------------------------

if st.session_state['dataframe'] is not None:
    df: pd.DataFrame = st.session_state['dataframe']

    # Preview
    st.markdown("### Dataset Preview")
    st.dataframe(pd.concat([df.head(10), df.tail(10)]), use_container_width=True)

    # Type detection – recomputed on each run 
    summary_obj = Summarizer(df)
    numeric_summary_df = summary_obj.numeric_summary()
    string_summary_df = summary_obj.string_summary()
    datetime_summary_df = summary_obj.datetime_summary()

    # --- Show tables ---
    st.markdown("---")
    st.markdown("## Current Variable Type Summary")

    show_block_numeric = st.toggle("Show Numeric Variables Summary", key="toggle_show_block_num")
    if show_block_numeric and not numeric_summary_df.empty:
        st.markdown("### Numeric Variables")
        st.dataframe(numeric_summary_df, use_container_width=True, hide_index=True)

    show_block_string = st.toggle("Show String Variables Summary", key="toggle_show_block_string")
    if show_block_string and not string_summary_df.empty:
        st.markdown("### String Variables")
        st.dataframe(string_summary_df, use_container_width=True, hide_index=True)

    show_block_datetime = st.toggle("Show Datetime Variables Summary", key="toggle_show_block_datetime")
    if show_block_datetime and not datetime_summary_df.empty:
        st.markdown("### Datetime Variables")
        st.dataframe(datetime_summary_df, use_container_width=True, hide_index=True)

    # ---- Interactive cleaning tool ----
    st.markdown("---")
    st.markdown("## Data Cleaning Tool")
    
    # General Actions
    st.markdown("### General actions:")
    processor_obj = Processor(df)
    processor_obj.general_actions()
    
    # ------------------------ Single-variable actions ------------------------
    st.markdown("### Single-variable actions:")
    processor_obj.single_actions()

    # ----------------------------------------------------------------------------
    # Target selection (numeric continuous only for now)
    # ----------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("## Variables Selection")
    
    st.markdown("### Target Variable Selection")
    selector_obj = Selector(df)
    selector_obj.target_and_problem_selection()
                
    st.write(st.session_state["target"])
    
    st.markdown("### Variable Selection")
    
    st.markdown("#### Analysis")
    if st.toggle("Univariate"):
        selector_obj.univariate_analysis()
        
    if st.toggle("Bivariate"):
        selector_obj.bivariate_analysis()
            
    if st.toggle("Multivariate"):
        st.markdown("#### Multivariate")
        selector_obj.multivariate_analysis()

    st.markdown("#### X variables")
    selector_obj.feature_selection()
        
    if st.button("Confirm Dataset and Target Variable"):
        if st.session_state.get("target") and st.session_state.get("problem_type") and (
            st.session_state["ml_dataset"] is not None and
            not st.session_state["ml_dataset"].empty):
            st.session_state["confirmed"] = True
            st.rerun()
            st.success("Dataset and target confirmed! Go to a model page from the sidebar.")
        else:
            st.error("Please select a valid target and confirm the problem type.")

    st.markdown("---")
    # ----------------------------------------------------------------------------
    # SUGGESTION: Download the cleaned dataset
    # ----------------------------------------------------------------------------
    st.markdown("## Export resulting dataset")
    with st.expander("Export cleaned dataset"):
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv")

# END OF FILE
