# app.py
import streamlit as st
from src.util import show_home_page

# Initialize session state for home page
if 'home_page_complete' not in st.session_state:
    st.session_state.home_page_complete = False

# Show home page if not complete
if not st.session_state.home_page_complete:
    # NOTE: The user can change this URL to a local path or another URL.
    show_home_page(background_image_path="https://img.freepik.com/free-vector/background-gradient-line-digital-abstract_483537-2954.jpg")
else:
    st.set_page_config(layout="wide")

    pages = {
        "Data": [st.Page("preprocessing.py", title="Prepare / Clean", icon=":material/build:")]
    }

    ptype = st.session_state.get("problem_type")  # set in preprocessing.py
    # Show models only after the user confirmed target/problem type
    if st.session_state.get("confirmed") and ptype:

        if ptype in ["classification_binary", "classification_multi"]:
            pages["Models"] = [
                st.Page("models/KNN.py",            title="KNN (Classifier)",  icon=":material/science:"),
                st.Page("models/Random Forest.py",  title="Random Forest (Clf)", icon=":material/forest:"),
                st.Page("models/SVM.py",            title="SVM (SVC)",         icon=":material/bolt:"),
                st.Page("models/Logistic Regression.py", title="Logistic Regression", icon=":material/linear_scale:"),
                st.Page("models/Naive Bayes.py",    title="Naive Bayes",       icon=":material/inbox:"),
                st.Page("models/XGBoost.py",        title="XGBoost",           icon=":material/rocket_launch:"),
            ]

        elif ptype == "regression":
            pages["Models"] = [
                st.Page("models/KNN.py",            title="KNN (Regressor)",   icon=":material/science:"),
                st.Page("models/Random Forest.py",  title="Random Forest (Reg)", icon=":material/forest:"),
                st.Page("models/SVM.py",            title="SVM (SVR)",         icon=":material/bolt:"),
                st.Page("models/Linear Models.py",  title="Linear Models",     icon=":material/straighten:"),
                st.Page("models/XGBoost.py",        title="XGBoost",           icon=":material/rocket_launch:"),
            ]
        
        elif ptype == "forecasting":
            print("ouch, gotta work on that")

    pg = st.navigation(pages, position="sidebar", expanded=True)
    pg.run()
