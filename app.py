# app.py
import streamlit as st

st.set_page_config(layout="wide")

pages = {
    "Data": [st.Page("preprocessing.py", title="Prepare / Clean", icon=":material/build:")]
}


ptype = st.session_state.get("problem_type")  # set in preprocessing.py
# Show models only after the user confirmed target/problem type
if st.session_state.get("sup_unsup_button") == "Supervised":
    
    if st.session_state.get("confirmed") and ptype:
        
        if ptype in ["classification_binary", "classification_multi"]:
            pages["Classification Models"] = [
                st.Page("supervised_models/KNN.py",            title="KNN (Classifier)",  icon=":material/science:"),
                st.Page("supervised_models/Random Forest.py",  title="Random Forest (Clf)", icon=":material/forest:"),
                st.Page("supervised_models/SVM.py",            title="SVM (SVC)",         icon=":material/bolt:"),
                st.Page("supervised_models/Logistic Regression.py", title="Logistic Regression", icon=":material/linear_scale:"),
                st.Page("supervised_models/Naive Bayes.py",    title="Naive Bayes",       icon=":material/inbox:"),
                st.Page("supervised_models/XGBoost.py",        title="XGBoost",           icon=":material/rocket_launch:"),
            ]
            
        elif ptype == "regression":
            pages["Regression Models"] = [
                st.Page("supervised_models/KNN.py",            title="KNN (Regressor)",   icon=":material/science:"),
                st.Page("supervised_models/Random Forest.py",  title="Random Forest (Reg)", icon=":material/forest:"),
                st.Page("supervised_models/SVM.py",            title="SVM (SVR)",         icon=":material/bolt:"),
                st.Page("supervised_models/Linear Models.py",  title="Linear Models",     icon=":material/straighten:"),
                st.Page("supervised_models/XGBoost.py",        title="XGBoost",           icon=":material/rocket_launch:"),
            ]
        
        elif ptype == "forecasting":
            print("ouch, gotta work on that")

elif st.session_state.get("sup_unsup_button") == "Unsupervised":   
    if st.session_state.get("confirmed"):
         pages["Clustering methods"] = [
                st.Page("unsupervised_models/Kmeans.py",           title="K-Means",  icon=":material/science:"),]


pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()