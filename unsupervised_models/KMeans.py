import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from src.util import show_centered_plot, load_descriptions, generate_blue_color_map# Assuming you have these helpers

# Load tooltips if available
tooltips = load_descriptions() 

# ------------------------ Page Configuration ------------------------
st.title("K-Means Clustering")

# ------------------------ Initial Checks ------------------------
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first on the Home page.")
    st.stop()

if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.header(f"Analysis of: ` {st.session_state.uploaded_file.name} `")

# ------------------------ Session State Initialization ------------------------
if 'KMEANS_analyzed' not in st.session_state:
    st.session_state.KMEANS_analyzed = False
if 'KMEANS_to_analyze' not in st.session_state:
    st.session_state.KMEANS_to_analyze = False
if 'KMEANS_params_changed' not in st.session_state:
    st.session_state.KMEANS_params_changed = False
if 'KMEANS_first_entered' not in st.session_state:
    st.session_state.KMEANS_first_entered = True
if 'KMEANS_final_model_trained' not in st.session_state:
    st.session_state.KMEANS_final_model_trained = False

# ------------------------ UI & Parameter Selection ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['ml_dataset']
    numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = numerical_cols
    
    if st.session_state.KMEANS_first_entered:
        st.session_state.KMEANS_last_params = {}

    st.sidebar.header('Parameters')
    # Parameters for finding optimal K
    # K-Means algorithm parameters
    min_max = st.sidebar.slider("Select K range",
                            min_value=0,
                            max_value=40,
                            value=(2, 20),
                            step=1,
                            help = tooltips['kmeans']['k_range'])    
    
    k_min = min_max[0]
    k_max = min_max[1]
    
    init_method = st.sidebar.selectbox('Initialization Method', ['k-means++', 'random'], help = tooltips['kmeans']['initialization_method'])
    n_init = st.sidebar.slider('Number of Initializations', 2, 20, 10, help = tooltips['kmeans']['number_of_initializations'])
    max_iter = st.sidebar.number_input('Max Iterations per Initialization', 100, 1000, 300, help = tooltips['kmeans']['max_iterations'])
    
    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State (seed)', 0, 9999, 42, help = tooltips['general']['random_state'])

    # Store current parameters to detect changes
    KMEANS_current_params = {
        'features': selected_features,
        'k_range': list(range(k_min, k_max + 1)),
        'init': init_method,
        'n_init': n_init,
        'max_iter': max_iter,
        'random_state': seed,
    }
    
    # --- Button to start the analysis to find optimal K ---
    if st.button("📊 Find Optimal Number of Clusters (K)"):
        if not selected_features:
            st.error("Please select at least one feature for clustering.")
        else:
            st.session_state.KMEANS_to_analyze = True
            st.session_state.KMEANS_first_entered = False
            st.session_state.KMEANS_params_changed = False
            st.session_state.KMEANS_last_params = KMEANS_current_params
            st.session_state.KMEANS_final_model_trained = False # Reset final model

    # Detect parameter changes
    if not st.session_state.KMEANS_first_entered and (KMEANS_current_params != st.session_state.KMEANS_last_params):
        st.session_state.KMEANS_params_changed = True
        st.session_state.KMEANS_to_analyze = False
        st.session_state.KMEANS_analyzed = False
        st.session_state.KMEANS_final_model_trained = False
        st.session_state.pop('KMEANS_metrics', None)
        st.session_state.pop('KMEANS_final_model', None)

    if st.session_state.KMEANS_params_changed:
        st.warning("⚠️ Parameters have changed. Please re-run the analysis.")

    # ------------------------ Step 1: Find Optimal K (Compute) ------------------------
    if st.session_state.KMEANS_to_analyze:
        with st.spinner("Analyzing different values of K..."):
            
            # --- Data Preparation ---
            data_subset = dataframe[selected_features].dropna()
            st.session_state.KMEANS_data_subset = data_subset # Save for later use
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_subset)
            st.session_state.KMEANS_data_scaled = data_scaled # Save for later use

            # --- K-Means Loop ---
            metrics = {
                "k": [],
                "inertia": [],
                "silhouette": [],
                "calinski_harabasz": []
            }
            k_range = st.session_state.KMEANS_last_params['k_range']

            for k in k_range:
                kmeans = KMeans(
                    n_clusters=k,
                    init=st.session_state.KMEANS_last_params['init'],
                    n_init=st.session_state.KMEANS_last_params['n_init'],
                    max_iter=st.session_state.KMEANS_last_params['max_iter'],
                    random_state=st.session_state.KMEANS_last_params['random_state']
                )
                kmeans.fit(data_scaled)
                
                metrics['k'].append(k)
                metrics['inertia'].append(kmeans.inertia_)
                metrics['silhouette'].append(silhouette_score(data_scaled, kmeans.labels_))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(data_scaled, kmeans.labels_))

            st.session_state.KMEANS_metrics = pd.DataFrame(metrics)
            st.session_state.KMEANS_analyzed = True
            st.session_state.KMEANS_to_analyze = False
            st.success("✅ Analysis for optimal K complete.")
            
    # ------------------------ Step 2: Display Optimal K Results ------------------------
    if st.session_state.KMEANS_analyzed:
        st.markdown("")
        st.markdown("---")
        st.markdown("### 📈 Analysis Plots for Optimal K")
        metrics_df = st.session_state.KMEANS_metrics
        
        # --- Plotting ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**The Elbow Method (Inertia)**", help = tooltips['general_unsupervised']['elbow_inertia'])
            fig, ax = plt.subplots()
            ax.plot(metrics_df['k'], metrics_df['inertia'], marker='o')
            ax.set_xlabel("Number of clusters (k)")
            ax.set_ylabel("Inertia (WCSS)")
            ax.grid(True)
            show_centered_plot(fig, width_ratio=7)

        with col2:
            st.markdown("**Silhouette Score**", help = tooltips['general_unsupervised']['silhouette'])
            fig, ax = plt.subplots()
            ax.plot(metrics_df['k'], metrics_df['silhouette'], marker='o')
            ax.set_xlabel("Number of clusters (k)")
            ax.set_ylabel("Silhouette Score")
            ax.grid(True)
            show_centered_plot(fig, width_ratio=7)
            
        st.markdown("")
        st.markdown("**Metrics Table**", help = tooltips['general_unsupervised']['metrics_table'])
        st.dataframe(metrics_df.set_index('k').style.background_gradient(
            subset=['silhouette', 'calinski_harabasz']).format("{:.2f}"))
        
        st.markdown("---")

        # ------------------------ Step 3: Final Model Training & Interpretation ------------------------
        st.markdown("### 🚀 Final Model and Clusters Analysis")
        final_k = st.number_input(
            "Choose a final number of Clusters",
            min_value=min(metrics_df['k']),
            max_value=max(metrics_df['k']),
            value=int(metrics_df.loc[metrics_df['silhouette'].idxmax()]['k']), # Suggest best k
            width=250
        )

        if st.button(f"Visualize clusterization, **K = {final_k}**"):
            
            with st.spinner("Running final model and profiling clusters..."):
                final_kmeans = KMeans(
                    n_clusters=final_k,
                    init = st.session_state.KMEANS_last_params['init'],
                    n_init = st.session_state.KMEANS_last_params['n_init'],
                    max_iter = st.session_state.KMEANS_last_params['max_iter'],
                    random_state = st.session_state.KMEANS_last_params['random_state']
                )
                
                data_scaled = st.session_state.KMEANS_data_scaled
                data_subset = st.session_state.KMEANS_data_subset.copy()
                
                # Fit and assign cluster labels
                clusters = final_kmeans.fit_predict(data_scaled)
                data_subset['cluster'] = clusters
                
                
                st.session_state.KMEANS_final_model = final_kmeans
                st.session_state.KMEANS_clustered_data = data_subset
                st.session_state.KMEANS_final_model_trained = True
                st.success(f"✅ Final model trained with {final_k} clusters.")

    # ------------------------ Step 4: Display Final Model Results ------------------------
    if st.session_state.KMEANS_final_model_trained:
        clustered_data = st.session_state.KMEANS_clustered_data
                
        st.markdown("#### 📊 Cluster Sizes")
        cluster_counts = clustered_data['cluster'].value_counts().sort_index()
        # st.bar_chart(cluster_counts)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # We need the data in a DataFrame format for Plotly Express
            pie_data = cluster_counts.reset_index()
            pie_data.columns = ['cluster', 'count']
            # Create the pie chart
            fig_pie = px.pie(
                pie_data,
                names='cluster',
                values='count',
                title='Data Points distributions',
            )
            # Customize the hover information and the text on the slices
            fig_pie.update_traces(
                textinfo='percent+label',  # Show percentage and label on the chart
                hovertemplate='<b>Cluster %{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
            )
            show_centered_plot(fig_pie, plot_type='plotly')
            
            
        with col2:
            st.markdown("**Cluster Profiles (Mean Values)**", help=tooltips['general_unsupervised']['cluster_profiles_mean'])
            cluster_profile = clustered_data.groupby('cluster').mean()
            st.dataframe(cluster_profile.style.background_gradient(cmap='Blues', axis=0))


        st.markdown("#### 🎨 Cluster Visualization via PCA)")
    
        if len(selected_features) > 1:
            data_scaled = st.session_state.KMEANS_data_scaled
            pca_2 = PCA(n_components=2, random_state=seed)
            data_pca_2 = pca_2.fit_transform(data_scaled)
            
            pca_3 = PCA(n_components=3, random_state=seed)
            data_pca_3 = pca_3.fit_transform(data_scaled)
            df_pca_3d = pd.DataFrame(
                    data_pca_3, 
                    columns=['PC1', 'PC2', 'PC3']
                )
            # Add cluster information. Convert to string to ensure discrete colors.
            df_pca_3d['cluster'] = clustered_data['cluster'].astype(str)


            if st.checkbox("2D visualization", key="cluster_pca_2d"):                
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = sns.scatterplot(
                    x=data_pca_2[:, 0], 
                    y=data_pca_2[:, 1], 
                    hue=clustered_data['cluster'],
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
                        # labels={
                        #     "PC1": f"PC 1 ({pca_3.explained_variance_ratio_[0]*100:.2f}%)",
                        #     "PC2": f"PC 2 ({pca_3.explained_variance_ratio_[1]*100:.2f}%)",
                        #     "PC3": f"PC 3 ({pca_3.explained_variance_ratio_[2]*100:.2f}%)"
                        # },
                    )
                
                # Improve layout
                fig_3d_interactive.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                
                # Display the interactive plot in Streamlit
                show_centered_plot(fig_3d_interactive, plot_type='plotly')
                
                
            
            
            
            
            
            
        else:
            st.warning("Visualization requires at least 2 features to be selected.")