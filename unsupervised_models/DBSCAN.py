import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from src.util import show_centered_plot, load_descriptions

# Load tooltips if available
tooltips = load_descriptions()

# ------------------------ Page Configuration ------------------------
st.title("DBSCAN Clustering")

# ------------------------ Initial Checks ------------------------
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first on the Home page.")
    st.stop()

if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.header(f"Analysis of: ` {st.session_state.uploaded_file.name} `")

# ------------------------ Session State Initialization ------------------------
if 'DBSCAN_analyzed' not in st.session_state:
    st.session_state.DBSCAN_analyzed = False

# ------------------------ UI & Parameter Selection ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['ml_dataset']
    numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = numerical_cols

    st.sidebar.header('Parameters')
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, help=tooltips.get('dbscan', {}).get('eps', ''))
    min_samples = st.sidebar.slider('Minimum Samples', 2, 20, 5, help=tooltips.get('dbscan', {}).get('min_samples', ''))

    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State (seed)', 0, 9999, 42, help=tooltips['general']['random_state'])

    if st.button("Analyze Clusters"):
        st.session_state.DBSCAN_analyzed = False

        # --- Data Preparation ---
        data_subset = dataframe[selected_features].dropna()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_subset)

        # --- DBSCAN Algorithm ---
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data_scaled)

        data_subset['cluster'] = clusters
        st.session_state.DBSCAN_clustered_data = data_subset

        # --- Metrics ---
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)

        metrics = {
            "Number of clusters": n_clusters,
            "Number of noise points": n_noise,
        }

        if n_clusters > 1:
            metrics["Silhouette Score"] = silhouette_score(data_scaled, clusters)
            metrics["Calinski-Harabasz Score"] = calinski_harabasz_score(data_scaled, clusters)

        st.session_state.DBSCAN_metrics = metrics
        st.session_state.DBSCAN_analyzed = True
        st.success("✅ DBSCAN analysis complete.")

    # ------------------------ Display Results ------------------------
    if st.session_state.DBSCAN_analyzed:
        clustered_data = st.session_state.DBSCAN_clustered_data
        metrics = st.session_state.DBSCAN_metrics

        st.markdown("---")
        st.markdown("### 📊 Analysis Results")

        # --- Metrics Display ---
        st.write("#### Metrics")
        st.json(metrics)

        st.markdown("#### 📊 Cluster Sizes")
        cluster_counts = clustered_data['cluster'].value_counts().sort_index()

        col1, col2 = st.columns(2)

        with col1:
            pie_data = cluster_counts.reset_index()
            pie_data.columns = ['cluster', 'count']
            pie_data['cluster'] = pie_data['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')

            fig_pie = px.pie(
                pie_data,
                names='cluster',
                values='count',
                title='Data Points Distribution',
            )
            fig_pie.update_traces(
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
            )
            show_centered_plot(fig_pie, plot_type='plotly')

        with col2:
            st.markdown("**Cluster Profiles (Mean Values)**", help=tooltips.get('general_unsupervised', {}).get('cluster_profiles_mean', ''))
            cluster_profile = clustered_data[clustered_data['cluster'] != -1].groupby('cluster').mean()
            st.dataframe(cluster_profile.style.background_gradient(cmap='Blues', axis=0))

        st.markdown("#### 🎨 Cluster Visualization via PCA")

        if len(selected_features) > 1:
            data_scaled = StandardScaler().fit_transform(dataframe[selected_features].dropna())

            pca_2 = PCA(n_components=2, random_state=seed)
            data_pca_2 = pca_2.fit_transform(data_scaled)

            pca_3 = PCA(n_components=3, random_state=seed)
            data_pca_3 = pca_3.fit_transform(data_scaled)

            df_pca_3d = pd.DataFrame(data_pca_3, columns=['PC1', 'PC2', 'PC3'])
            df_pca_3d['cluster'] = clustered_data['cluster'].astype(str)

            if st.checkbox("2D visualization", key="cluster_pca_2d"):
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    x=data_pca_2[:, 0],
                    y=data_pca_2[:, 1],
                    hue=clustered_data['cluster'],
                    palette=sns.color_palette("hsv", len(set(clustered_data['cluster']))),
                    ax=ax
                )
                ax.set_title("Clusters projected onto 2 Principal Components")
                ax.set_xlabel(f"Principal Component 1 ({pca_2.explained_variance_ratio_[0]*100:.2f}% variance)")
                ax.set_ylabel(f"Principal Component 2 ({pca_2.explained_variance_ratio_[1]*100:.2f}% variance)")
                ax.grid(True)
                show_centered_plot(fig)

            if st.checkbox("3D visualization", key="cluster_pca_3d"):
                fig_3d_interactive = px.scatter_3d(
                    df_pca_3d,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='cluster',
                    title="Clusters projected onto 3 Principal Components",
                )
                fig_3d_interactive.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                show_centered_plot(fig_3d_interactive, plot_type='plotly')
        else:
            st.warning("Visualization requires at least 2 features to be selected.")