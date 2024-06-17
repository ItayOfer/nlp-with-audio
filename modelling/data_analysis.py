import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

audio_features = ['audio_feature_mfccs_mean_7', 'audio_feature_mfccs_mean_2', 'audio_feature_mfccs_mean_9',
                  'audio_feature_mfccs_mean_8', 'audio_feature_mfccs_mean_5', 'audio_feature_mfccs_mean_10',
                  'audio_feature_mfccs_mean_0', 'audio_feature_rms_energy_mean', 'audio_feature_zcr_mean',
                  'audio_feature_mfccs_mean_14', 'audio_feature_centroid_mean', 'audio_feature_tempogram_mean',
                  'audio_feature_mfccs_mean_12', 'audio_feature_mfccs_mean_15',
                  'audio_feature_tempogram_ratio_mean']
text_features = [col for col in X_train.columns if col.startswith('text_feature_')]
features_engineered = [col for col in X_train.columns if col.startswith('fe_features_')]

correlation_audio = X_train[audio_features].corr()
correlation_text = X_train[text_features].corr()
correlation_audio_text = X_train[audio_features + text_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_audio, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Test Features')
plt.show()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
data_scaled = scaler.fit_transform(X_train[text_features])

inertias = []
K_range = range(1, 11)  # Usually, we test from 1 to 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.xticks(K_range)
plt.show()