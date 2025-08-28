# app.py
import streamlit as st

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
            st.Page("models/SVM.py",            title="SVM (SVC)",         icon=":material/bolt:")
        ]
        
    elif ptype == "regression":
        pages["Models"] = [
#            st.Page("models/KNN.py",            title="KNN (Regressor)",   icon=":material/science:"),
            st.Page("models/Random Forest.py",  title="Random Forest (Reg)", icon=":material/forest:"),
#            st.Page("models/SVM.py",            title="SVM (SVR)",         icon=":material/bolt:")
        ]
    
    elif ptype == "forecasting":
        print("ouch, gotta work on that")

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
