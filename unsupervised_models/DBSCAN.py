import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from itertools import product
from src.util import show_centered_plot, load_descriptions # Assuming you have these helpers

# Load tooltips if available
tooltips = load_descriptions() 

# ------------------------ Page Configuration ------------------------
st.title("DBSCAN Clustering with Grid Search")

# ------------------------ Initial Checks ------------------------
if not st.session_state.get('confirmed', False):
    st.warning("Please upload and confirm your dataset first on the Home page.")
    st.stop()

if hasattr(st.session_state.get('uploaded_file', None), 'name'):
    st.header(f"Analysis of: ` {st.session_state.uploaded_file.name} `")

# ------------------------ Session State Initialization for the new workflow ------------------------
if 'DBSCAN_first_entered' not in st.session_state:
    st.session_state.DBSCAN_first_entered = True
if 'DBSCAN_params_changed' not in st.session_state:
    st.session_state.DBSCAN_params_changed = False
# Stage 1: Grid Search
if 'DBSCAN_to_grid_search' not in st.session_state:
    st.session_state.DBSCAN_to_grid_search = False
if 'DBSCAN_grid_search_complete' not in st.session_state:
    st.session_state.DBSCAN_grid_search_complete = False
# Stage 2: Final Model
if 'DBSCAN_final_model_complete' not in st.session_state:
    st.session_state.DBSCAN_final_model_complete = False


# ------------------------ UI & Parameter Selection ------------------------
if st.session_state.confirmed:
    dataframe = st.session_state['ml_dataset']
    numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = numerical_cols
    
    if st.session_state.DBSCAN_first_entered:
        st.session_state.DBSCAN_last_params = {}

    st.sidebar.header('Parameters', help = "Values here are being evaluated via gridsearch.\nYou should then explore which combination suits your interests.")

    # --- UI for selecting multiple parameters ---
    eps_range = st.sidebar.slider(
        "Select range for Epsilon",
        min_value=0.1, max_value=5.0, value=(0.3, 0.7), step=0.1,
        help=tooltips['dbscan']['epsilon']
    )
    eps_steps = st.sidebar.number_input("Number of `eps` values to test", min_value=2, max_value=20, value=5)
    eps_values = np.linspace(eps_range[0], eps_range[1], eps_steps)

    min_samples_values_pre = st.sidebar.multiselect(
        'Minimum Samples',
        options=[5,10,15,20,25,30],
        default=[5, 10, 15],
        accept_new_options=True,
        help=tooltips['dbscan']['min_samples']
    )
    if min_samples_values_pre: # Ensure the list is not empty
        min_samples_values = [int(x) for x in min_samples_values_pre]

    st.sidebar.markdown('---')
    seed = st.sidebar.number_input('Random State', 0, 9999, 42, help=tooltips['general']['random_state'])

    total_combinations = len(eps_values) * len(min_samples_values)
    
    # Store current parameters to detect changes
    DBSCAN_current_params = {
        'eps_values': list(eps_values),
        'min_samples_values': min_samples_values,
    }
    
    # --- Button to start the grid search ---
    if st.button("🔍 Run Parameter Grid Search", disabled=(total_combinations == 0)):
        st.session_state.DBSCAN_to_grid_search = True
        st.session_state.DBSCAN_first_entered = False
        st.session_state.DBSCAN_params_changed = False
        st.session_state.DBSCAN_last_params = DBSCAN_current_params
        # Reset downstream states
        st.session_state.DBSCAN_grid_search_complete = False
        st.session_state.DBSCAN_final_model_complete = False


    # Detect parameter changes and reset the entire workflow
    if not st.session_state.DBSCAN_first_entered and (DBSCAN_current_params != st.session_state.DBSCAN_last_params):
        st.session_state.DBSCAN_params_changed = True
        st.session_state.DBSCAN_to_grid_search = False
        st.session_state.DBSCAN_grid_search_complete = False
        st.session_state.DBSCAN_final_model_complete = False
        st.session_state.pop('DBSCAN_grid_search_results', None)
        st.session_state.pop('DBSCAN_final_params', None)

    if st.session_state.DBSCAN_params_changed:
        st.warning("⚠️ Parameters have changed. Please re-run the grid search.")

    # ------------------------ STAGE 1: Grid Search Computation ------------------------
    if st.session_state.DBSCAN_to_grid_search:
        with st.spinner(f"Running DBSCAN for {total_combinations} combinations..."):
            
            data_subset = dataframe[selected_features].dropna()
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_subset)
            st.session_state.DBSCAN_data_scaled = data_scaled # Save for final model
            st.session_state.DBSCAN_data_subset = data_subset # Save for final model

            grid_search_results = []
            param_combinations = list(product(
                st.session_state.DBSCAN_last_params['eps_values'],
                st.session_state.DBSCAN_last_params['min_samples_values']
            ))

            for eps, min_samples in param_combinations:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                labels = dbscan.fit_predict(data_scaled)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = np.count_nonzero(labels == -1)
                noise_percentage = (n_noise / len(labels)) * 100 if len(labels) > 0 else 0
                
                silhouette = None
                if n_clusters > 1:
                    clean_labels = labels[labels != -1]
                    clean_data = data_scaled[labels != -1]
                    if len(set(clean_labels)) > 1:
                        silhouette = silhouette_score(clean_data, clean_labels)
                
                grid_search_results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'noise_%': noise_percentage,
                    'silhouette': silhouette
                })
            
            st.session_state.DBSCAN_grid_search_results = pd.DataFrame(grid_search_results)
            st.session_state.DBSCAN_grid_search_complete = True
            st.session_state.DBSCAN_to_grid_search = False
            st.success(f"✅ Grid search complete. Found results for {total_combinations} combinations.")

    # ------------------------ STAGE 1: Display Grid Search Results ------------------------
    if st.session_state.DBSCAN_grid_search_complete:
        st.markdown("---")
        st.markdown("### 📊 Grid Search Results", help= tooltips['dbscan']['grid_search_result'])
        
        results_df = st.session_state.DBSCAN_grid_search_results
        
        # Display styled dataframe
        st.dataframe(results_df.style
            .format({'eps': '{:.3f}', 'noise_%': '{:.2f}%', 'silhouette': '{:.3f}'})
            .background_gradient(cmap='viridis', subset=['silhouette'])
            .background_gradient(cmap='Reds', subset=['noise_%'])
        )

        # Heatmap Visualization
        if len(results_df['eps'].unique()) > 1 and len(results_df['min_samples'].unique()) > 1:
            st.markdown("#### Heatmap of Silhouette Scores")
            # Pivot table for heatmap
            pivot_silhouette = results_df.pivot(index='min_samples', columns='eps', values='silhouette')
            
            # Round the epsilon column labels to 3 decimal places
            pivot_silhouette.columns = pivot_silhouette.columns.to_series().round(3)
                        
            fig, ax = plt.subplots(figsize=(10, 6))
            # Added annot=True and a proper format string to display values in the heatmap
            sns.heatmap(pivot_silhouette, annot=True, fmt='.2f', cmap='viridis', ax=ax, cbar_kws={'label': 'Silhouette Score'})
            ax.set_title('Silhouette Score')
            ax.set_xlabel('Epsilon (eps)')
            ax.set_ylabel('Minimum Samples')
            show_centered_plot(fig)

        # ------------------------ STAGE 2: Select Final Parameters ------------------------
        st.markdown("---")
        st.markdown("### 🚀 Final Model Selection")
        st.write("Based on the results above, select the final parameters to visualize the clusters.")

        # Suggest best parameters based on silhouette score (with low noise as a tie-breaker)
        best_params = results_df.loc[results_df['silhouette'].idxmax()]

        col1, col2 = st.columns(2)
        final_eps = col1.number_input(
            "Final Epsilon (eps)", 
            value=best_params['eps'], format="%.3f"
        )
        final_min_samples = col2.number_input(
            "Final Minimum Samples", 
            value=int(best_params['min_samples'])
        )

        if st.button("Visualize Clusters with Selected Parameters"):
            st.session_state.DBSCAN_final_params = {'eps': final_eps, 'min_samples': final_min_samples}
            
            # --- Run Final Model ---
            with st.spinner("Running final DBSCAN model..."):
                data_scaled = st.session_state.DBSCAN_data_scaled
                data_subset = st.session_state.DBSCAN_data_subset
                
                dbscan = DBSCAN(eps=final_eps, min_samples=final_min_samples)
                clusters = dbscan.fit_predict(data_scaled)
                
                clustered_data = data_subset.copy()
                clustered_data['cluster'] = clusters
                st.session_state.DBSCAN_clustered_data = clustered_data
                st.session_state.DBSCAN_final_model_complete = True
            st.success("✅ Final model is ready.")
            
    # ------------------------ STAGE 2: Display Final Analysis ------------------------
    if st.session_state.DBSCAN_final_model_complete:
        st.markdown("---")
        st.markdown(f"### 📈 Detailed Analysis for `eps={st.session_state.DBSCAN_final_params['eps']:.3f}` and `min_samples={st.session_state.DBSCAN_final_params['min_samples']}`")

        clustered_data = st.session_state.DBSCAN_clustered_data
        
        # --- Display Cluster Profiles ---
        st.markdown("#### 📊 Cluster Sizes")
        col1, col2 = st.columns(2)
        with col1:
            cluster_counts = clustered_data['cluster'].value_counts().sort_index()
            pie_data = cluster_counts.reset_index()
            pie_data.columns = ['cluster', 'count']
            pie_data['cluster'] = pie_data['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
            
            fig_pie = px.pie(pie_data, names='cluster', values='count', title='Data Points Distribution')
            fig_pie.update_traces(textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}')
            show_centered_plot(fig_pie, plot_type='plotly')
            
        with col2:
            st.markdown("**Cluster Profiles (Mean Values)**", help=tooltips['general_unsupervised']['cluster_profiles_mean'])
            if -1 in clustered_data['cluster'].unique():
                 st.caption("Noise points (cluster -1) are excluded from this summary.")
            cluster_profile = clustered_data[clustered_data['cluster'] != -1].groupby('cluster').mean()
            if not cluster_profile.empty:
                st.dataframe(cluster_profile.style.background_gradient(cmap='Blues', axis=0))
            else:
                st.warning("No clusters were found (only noise). Cannot generate profiles.")

        # --- Display PCA Visualization ---
        st.markdown("---")
        st.markdown("#### 🎨 Cluster Visualization (via PCA)")
        if len(selected_features) > 1:
            data_scaled = st.session_state.DBSCAN_data_scaled
            
            pca_2 = PCA(n_components=2, random_state=seed)
            data_pca_2 = pca_2.fit_transform(data_scaled)
            df_pca_2d = pd.DataFrame(data_pca_2, columns=['PC1', 'PC2'])
            df_pca_2d['cluster'] = clustered_data['cluster'].astype(str)

            pca_3 = PCA(n_components=3, random_state=seed)
            data_pca_3 = pca_3.fit_transform(data_scaled)
            df_pca_3d = pd.DataFrame(data_pca_3, columns=['PC1', 'PC2', 'PC3'])
            df_pca_3d['cluster'] = clustered_data['cluster'].astype(str)

            if st.checkbox("Show 2D visualization", key="cluster_pca_2d_final_dbscan"):
                fig = px.scatter(
                    df_pca_2d, x='PC1', y='PC2', color='cluster',
                    title="Clusters projected onto 2 Principal Components",
                    labels={"PC1": f"PC 1 ({pca_2.explained_variance_ratio_[0]*100:.2f}%)", "PC2": f"PC 2 ({pca_2.explained_variance_ratio_[1]*100:.2f}%)"},
                    color_discrete_map={'-1': 'grey'}
                )
                show_centered_plot(fig, plot_type='plotly')
            
            if st.checkbox("Show 3D visualization", key="cluster_pca_3d_final_dbscan"):
                fig_3d = px.scatter_3d(
                    df_pca_3d, x='PC1', y='PC2', z='PC3', color='cluster',
                    title="Clusters projected onto 3 Principal Components",
                    labels={"PC1": f"PC 1 ({pca_3.explained_variance_ratio_[0]*100:.2f}%)", "PC2": f"PC 2 ({pca_3.explained_variance_ratio_[1]*100:.2f}%)", "PC3": f"PC 3 ({pca_3.explained_variance_ratio_[2]*100:.2f}%)"},
                    color_discrete_map={'-1': 'grey'}
                )
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
                show_centered_plot(fig_3d, plot_type='plotly')
        else:
            st.warning("PCA Visualization requires at least 2 features.")