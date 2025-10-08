# hierarchical_clustering.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Assuming these helpers are in src/util.py
from src.util import show_centered_plot, load_descriptions

# Load tooltips if available
tooltips = load_descriptions()

# ------------------------ Page Configuration ------------------------
st.title("Hierarchical Agglomerative Clustering")

# ------------------------ Initial Checks ------------------------
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first on the Home page.")
    st.stop()

if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.header(f"Analysis of: ` {st.session_state.uploaded_file.name} `")
    

# ------------------------ Session State Initialization ------------------------
if 'HIERARCHICAL_first_entered' not in st.session_state:
    st.session_state.HIERARCHICAL_first_entered = True
if 'HIERARCHICAL_params_changed' not in st.session_state:
    st.session_state.HIERARCHICAL_params_changed = False
if 'HIERARCHICAL_to_analyze' not in st.session_state:
    st.session_state.HIERARCHICAL_to_analyze = False
if 'HIERARCHICAL_analyzed' not in st.session_state:
    st.session_state.HIERARCHICAL_analyzed = False
if 'HIERARCHICAL_final_model_trained' not in st.session_state:
    st.session_state.HIERARCHICAL_final_model_trained = False

# ------------------------ UI & Parameter Selection ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['ml_dataset']
    numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = numerical_cols # Using all numerical features by default

    if st.session_state.HIERARCHICAL_first_entered:
        st.session_state.HIERARCHICAL_last_params = {}

    st.sidebar.header('Parameters')
    
    # Hierarchical Clustering algorithm parameters
    linkage_method = st.sidebar.selectbox(
        'Linkage Method',
        ['ward', 'complete', 'average', 'single'],
        help=tooltips['hierarchical']['linkage_method']
    )

    # Note: 'ward' linkage only works with 'euclidean' distance.
    if linkage_method == 'ward':
        metric = 'euclidean'
        st.sidebar.warning("'ward' linkage requires the 'euclidean' metric.")
        
    else:
        metric = st.sidebar.selectbox(
            'Distance Metric',
            ['euclidean', 'cosine'],
            help=tooltips['hierarchical']['distance_metric']
        )
        
    st.sidebar.subheader("Dendrogram Parameters")
    truncate_mode = st.sidebar.selectbox(
        'Truncate Mode',
        ['lastp', 'level', None],
        index=0,
        help=tooltips['hierarchical']['truncate_mode']
    )
    p_truncate = st.sidebar.number_input(
        'Levels/Clusters to Show (p)',
        min_value=2,
        max_value=50,
        value=15,
        help=tooltips['hierarchical']['p_truncate']
    )

    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State', 0, 9999, 42, help=tooltips['general']['random_state'])

    # Store current parameters to detect changes
    HIERARCHICAL_current_params = {
        'linkage': linkage_method,
        'metric': metric,
        'truncate_mode': truncate_mode,
        'p_truncate': p_truncate,
        'random_state': seed
    }

    # --- Button to start the analysis ---
    if st.button("📊 Generate Dendrogram and Analyze"):
        st.session_state.HIERARCHICAL_to_analyze = True
        st.session_state.HIERARCHICAL_first_entered = False
        st.session_state.HIERARCHICAL_params_changed = False
        st.session_state.HIERARCHICAL_last_params = HIERARCHICAL_current_params
        # Reset downstream states
        st.session_state.HIERARCHICAL_analyzed = False
        st.session_state.HIERARCHICAL_final_model_trained = False

    # Detect parameter changes
    if not st.session_state.HIERARCHICAL_first_entered and (HIERARCHICAL_current_params != st.session_state.HIERARCHICAL_last_params):
        st.session_state.HIERARCHICAL_params_changed = True
        st.session_state.HIERARCHICAL_to_analyze = False
        st.session_state.HIERARCHICAL_analyzed = False
        st.session_state.HIERARCHICAL_final_model_trained = False
        st.session_state.pop('HIERARCHICAL_linkage_matrix', None)
        st.session_state.pop('HIERARCHICAL_final_model', None)

    if st.session_state.HIERARCHICAL_params_changed:
        st.warning("⚠️ Parameters have changed. Please re-run the analysis.")


    # ------------------------ Stage 1: Dendrogram Generation (Compute) ------------------------
    if st.session_state.HIERARCHICAL_to_analyze:
        with st.spinner("Generating linkage matrix and dendrogram..."):
            # --- Data Preparation ---
            data_subset = dataframe[selected_features].dropna()
            st.session_state.HIERARCHICAL_data_subset = data_subset # Save for later

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_subset)
            st.session_state.HIERARCHICAL_data_scaled = data_scaled # Save for later

            # --- Compute Linkage Matrix ---
            linkage_matrix = linkage(
                data_scaled,
                method=st.session_state.HIERARCHICAL_last_params['linkage'],
                metric=st.session_state.HIERARCHICAL_last_params['metric']
            )
            st.session_state.HIERARCHICAL_linkage_matrix = linkage_matrix
            st.session_state.HIERARCHICAL_analyzed = True
            st.session_state.HIERARCHICAL_to_analyze = False
            st.success("✅ Dendrogram analysis is ready.")

    # ------------------------ Stage 1: Display Dendrogram ------------------------
    if st.session_state.HIERARCHICAL_analyzed:
        st.markdown("---")
        st.markdown("### 🌳 Dendrogram", help=tooltips['hierarchical']['dendrogram_help'])

        linkage_matrix = st.session_state.HIERARCHICAL_linkage_matrix
        params = st.session_state.HIERARCHICAL_last_params
        
        fig, ax = plt.subplots(figsize=(12, 7))
        dendrogram(
            linkage_matrix,
            ax=ax,
            truncate_mode=params['truncate_mode'],
            p=params['p_truncate'],
            leaf_rotation=90.,
            leaf_font_size=8.,
        )
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Sample index or (cluster size)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        show_centered_plot(fig, width_ratio=7.5)
        
        st.markdown("---")

        # ------------------------ Stage 2: Final Model Training & Interpretation ------------------------
        st.markdown("### 🚀 Final Model and Clusters Analysis")
        final_k = st.number_input(
            "Choose a final number of Clusters (based on the dendrogram)",
            min_value=2,
            max_value=params['p_truncate'] * 2, # A reasonable upper limit
            value=3, # A sensible default
            step=1,
            help="Look at the dendrogram above. A good number of clusters is often where you can make a horizontal cut that crosses a large vertical distance without merging clusters."
        )

        if st.button(f"Visualize Clusterization, K = {final_k}"):
            with st.spinner("Running final model and profiling clusters..."):
                final_model = AgglomerativeClustering(
                    n_clusters=final_k,
                    metric=params['metric'],
                    linkage=params['linkage']
                )
                
                data_scaled = st.session_state.HIERARCHICAL_data_scaled
                data_subset = st.session_state.HIERARCHICAL_data_subset.copy()
                
                # Fit and assign cluster labels
                clusters = final_model.fit_predict(data_scaled)
                data_subset['cluster'] = clusters
                
                st.session_state.HIERARCHICAL_clustered_data = data_subset
                st.session_state.HIERARCHICAL_final_model_trained = True
                st.success(f"✅ Final model trained with {final_k} clusters.")
    
    # ------------------------ Stage 2: Display Final Model Results ------------------------
    if st.session_state.HIERARCHICAL_final_model_trained:
        clustered_data = st.session_state.HIERARCHICAL_clustered_data
                
        st.markdown("#### 📊 Cluster Sizes")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Pie chart for cluster distribution
            cluster_counts = clustered_data['cluster'].value_counts().sort_index()
            pie_data = cluster_counts.reset_index()
            pie_data.columns = ['cluster', 'count']
            
            fig_pie = px.pie(
                pie_data,
                names='cluster',
                values='count',
                title='Data Points Distribution'
            )
            fig_pie.update_traces(
                textinfo='percent+label',
                hovertemplate='<b>Cluster %{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
            )
            show_centered_plot(fig_pie, plot_type='plotly')
            
        with col2:
            st.markdown("**Cluster Profiles (Mean Values)**", help=tooltips['general_unsupervised']['cluster_profiles_mean'])
            cluster_profile = clustered_data.groupby('cluster').mean()
            st.dataframe(cluster_profile.style.background_gradient(cmap='Blues', axis=0))

        st.markdown("#### 🎨 Cluster Visualization (via PCA)")
    
        if len(selected_features) > 1:
            data_scaled = st.session_state.HIERARCHICAL_data_scaled
            
            # --- 2D PCA ---
            pca_2 = PCA(n_components=2, random_state=seed)
            data_pca_2 = pca_2.fit_transform(data_scaled)
            df_pca_2d = pd.DataFrame(data_pca_2, columns=['PC1', 'PC2'])
            df_pca_2d['cluster'] = clustered_data['cluster'].astype(str)

            # --- 3D PCA ---
            pca_3 = PCA(n_components=3, random_state=seed)
            data_pca_3 = pca_3.fit_transform(data_scaled)
            df_pca_3d = pd.DataFrame(data_pca_3, columns=['PC1', 'PC2', 'PC3'])
            df_pca_3d['cluster'] = clustered_data['cluster'].astype(str)

            if st.checkbox("Show 2D visualization", key="hierarchical_pca_2d"):
                fig_2d = px.scatter(
                    df_pca_2d,
                    x='PC1',
                    y='PC2',
                    color='cluster',
                    title="Clusters projected onto 2 Principal Components",
                    labels={
                        "PC1": f"PC 1 ({pca_2.explained_variance_ratio_[0]*100:.2f}%)",
                        "PC2": f"PC 2 ({pca_2.explained_variance_ratio_[1]*100:.2f}%)"
                    }
                )
                show_centered_plot(fig_2d, plot_type='plotly')

            if st.checkbox("Show 3D visualization", key="hierarchical_pca_3d"):
                fig_3d = px.scatter_3d(
                    df_pca_3d,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='cluster',
                    title="Clusters projected onto 3 Principal Components",
                    labels={
                        "PC1": f"PC 1 ({pca_3.explained_variance_ratio_[0]*100:.2f}%)",
                        "PC2": f"PC 2 ({pca_3.explained_variance_ratio_[1]*100:.2f}%)",
                        "PC3": f"PC 3 ({pca_3.explained_variance_ratio_[2]*100:.2f}%)"
                    }
                )
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
                show_centered_plot(fig_3d, plot_type='plotly')
        else:
            st.warning("PCA Visualization requires at least 2 features.")