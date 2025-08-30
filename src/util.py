import streamlit as st 

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
# for col in some_numeric_columns:
#     action = action_radio_for_column(col, llm_model="qwen2.5-coder:3b", llm_url="http://localhost:11434")
#     if action == "Ask LLM":
#         # call your ask_llm(...) helper
#         pass


def show_centered_matplotlib(fig, width_ratio=2.4):
    # left : middle : right ratios — tweak width_ratio to change middle width
    left_ , mid, right_ = st.columns([1, width_ratio, 1])
    with mid:
        st.pyplot(fig, use_container_width=True)