import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.mixture import GaussianMixture

import utils
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

if __name__ == '__main__':
    train_data, test_data, y_train, y_test = utils.get_data()
    # train_data = pd.read_csv('train_data.csv')
    # y_train = pd.read_csv('y_train.csv')
    # train_data = train_data.set_index('file_key')
    # y_train = y_train.set_index('file_key')
    text_feature_names = [col for col in train_data.columns if col.startswith('text_feature_')]
    audio_feature_names = [col for col in train_data.columns if col.startswith('audio_feature_')]
    standard_scaler = StandardScaler()
    scaled_train_data = standard_scaler.fit_transform(train_data)
    scaled_train_data = pd.DataFrame(scaled_train_data, columns=train_data.columns)
    X_audio = scaled_train_data[audio_feature_names]
    X_text = scaled_train_data[text_feature_names]
    sse = []  # Sum of squared errors
    for k in range(1, 11):  # Test different numbers of clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(train_data)
        sse.append(kmeans.inertia_)

    # Plot SSE for each *k*
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.xticks(range(1, 11))
    plt.show()

    kmeans_t = KMeans(n_clusters=3, random_state=42)
    kmeans_a = KMeans(n_clusters=3, random_state=42)
    kmeans_audio = kmeans_a.fit(X_audio)
    kmeans_text = kmeans_t.fit(X_text)
    cm = confusion_matrix(kmeans_audio.labels_, kmeans_text.labels_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.title('Confusion Matrix')
    plt.xlabel('Audio Labels')
    plt.ylabel('Text Labels')
    plt.show()
    acc = accuracy_score(kmeans_audio.labels_, kmeans_text.labels_)
    print(f'Accuracy: {acc}')

    # Apply PCA to reduce dimensions to 2 for visualization
    pca = PCA(n_components=2)
    principal_components_audio = pca.fit_transform(X_audio)

    # Create a scatter plot of the two PCA components
    plt.figure(figsize=(10, 6))
    scatter_audio = plt.scatter(principal_components_audio[:, 0], principal_components_audio[:, 1], c=y_train, cmap='viridis',
                          marker='o', alpha=0.5)
    plt.title('Cluster Visualization after PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Adding a color bar to show cluster colors
    plt.colorbar(scatter_audio, label='Cluster')

    # # Optionally, you can add annotations or additional information
    # for i, txt in enumerate(df.index):
    #     plt.annotate(txt, (principal_components[i, 0], principal_components[i, 1]), textcoords="offset points",
    #                  xytext=(0, 10), ha='center')

    plt.show()

    df_audio_cluster = pd.DataFrame({'cluster_column': kmeans_audio.labels_,
                                     'target': y_train})
    df_text_cluster = pd.DataFrame({'cluster_column': kmeans_text.labels_,
                                    'target': y_train})

    df_text_cluster['cluster_column'] = pd.Categorical(df_text_cluster['cluster_column'])
    df_audio_cluster['cluster_column'] = pd.Categorical(df_audio_cluster['cluster_column'])

    def plot_dist_of_cluster(df, title):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='target', hue='cluster_column', multiple='dodge', shrink=0.8, bins=2, discrete=True)
        plt.title(f'Distribution of Target by Cluster In {title} Features')
        plt.xlabel('Target')
        plt.ylabel('Count')
        plt.xticks([0, 1, 2], labels=['Negative', 'Neutral', 'Positive'])
        plt.show()


    plot_dist_of_cluster(df_audio_cluster, 'Audio')
    plot_dist_of_cluster(df_text_cluster, 'Text')
    df_plot_audio_cluster = df_audio_cluster.reset_index()
    df_plot_audio_cluster[audio_feature_names] = X_audio[audio_feature_names]
    import seaborn as sns
    import matplotlib.pyplot as plt

    for f in audio_feature_names:
        sns.boxplot(x='cluster_column', y=f, hue='target', data=df_plot_audio_cluster)
        plt.title('Box Plot of Values by Category')
        plt.show()

    import scipy.stats as stats
    selected_features = []
    homogeneity = []
    for feature in audio_feature_names:
        # Test between clusters
        clusters = df_plot_audio_cluster['cluster_column'].unique()
        cluster_groups = [df_plot_audio_cluster[df_plot_audio_cluster['cluster_column'] == cluster][feature] for cluster in clusters]
        stat, p_between = stats.kruskal(*cluster_groups)

        # If significant differences between clusters
        if p_between < 0.1:
            homogeneity_within_cluster = True
            homogeneity.append(feature)

            # Test within each cluster for target homogeneity
            for cluster in clusters:
                subdf = df_plot_audio_cluster[df_plot_audio_cluster['cluster_column'] == cluster]
                target_groups = [subdf[subdf['target'] == target][feature] for target in subdf['target'].unique()]
                if len(target_groups) > 1:
                    stat, p_within = stats.kruskal(*target_groups)
                    if p_within >= 0.05:
                        continue
                    else:
                        homogeneity_within_cluster = False
                        break

            if homogeneity_within_cluster:
                selected_features.append(feature)

    print("Selected features:", selected_features)



class ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_type='kmeans', n_clusters=3):
        self.cluster_type = cluster_type
        self.n_clusters = n_clusters
        self.clusterer = None

    def fit(self, X, y=None):
        if self.cluster_type == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.n_clusters)
        elif self.cluster_type == 'dbscan':
            self.clusterer = DBSCAN(min_samples=self.n_clusters)
        elif self.cluster_type == 'gmm':
            self.clusterer = GaussianMixture(n_components=self.n_clusters)
        else:
            raise ValueError("Unsupported cluster type. Choose from 'kmeans', 'dbscan', or 'gmm'")

        self.clusterer.fit(X)
        return self

    def transform(self, X):
        # Predict the clusters
        if hasattr(self.clusterer, 'labels_'):
            labels = self.clusterer.labels_
        else:
            labels = self.clusterer.predict(X)

        # Return the original DataFrame with an added column for the cluster labels
        X_transformed = X.copy()  # Ensure we don't modify the original DataFrame
        X_transformed['cluster_label'] = labels
        return X_transformed
