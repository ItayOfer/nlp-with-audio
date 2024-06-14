from feature_engineering import FeatureEngineering
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import utils
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from modelling.scaler import CustomMinMaxScaler
from xgboost import XGBClassifier


if __name__ == '__main__':
    train_data, test_data, y_train, y_test = utils.get_data()
    audio_feature_name = [col for col in train_data.columns if col.startswith('audio_feature_')]
    # audio_feature_name = ['audio_feature_mfccs_mean_7', 'audio_feature_mfccs_mean_2', 'audio_feature_mfccs_mean_9',
    #                       'audio_feature_mfccs_mean_8', 'audio_feature_mfccs_mean_5', 'audio_feature_mfccs_mean_10',
    #                       'audio_feature_mfccs_mean_0', 'audio_feature_rms_energy_mean', 'audio_feature_zcr_mean',
    #                       'audio_feature_mfccs_mean_14', 'audio_feature_centroid_mean', 'audio_feature_tempogram_mean',
    #                       'audio_feature_mfccs_mean_12', 'audio_feature_mfccs_mean_15',
    #                       'audio_feature_tempogram_ratio_mean']
    text_feature_name = [col for col in train_data.columns if col.startswith('text_feature_')]
    fe = FeatureEngineering()
    test_meld = pd.read_csv('../MELD.Raw/test_sent_emo.csv')
    train_meld = pd.read_csv('../MELD.Raw/train/train_sent_emo.csv')
    fe_train_data = fe.run(train_meld)
    fe_test_data = fe.run(test_meld, train=False)
    fe_features_name = fe_train_data.columns
    common_indices_train = fe_train_data.index.intersection(y_train.index)
    fe_train_data = fe_train_data.loc[common_indices_train]
    common_indices_test = fe_test_data.index.intersection(y_test.index)
    fe_test_data = fe_test_data.loc[common_indices_test]
    X_train_fe = fe_train_data[fe_features_name]
    X_test_fe = fe_test_data[fe_features_name]
    X_train_audio_text = train_data[audio_feature_name + text_feature_name]
    X_test_audio_text = test_data[audio_feature_name + text_feature_name]
    X_train = pd.merge(X_train_fe, X_train_audio_text, left_index=True, right_index=True)
    X_test = pd.merge(X_test_fe, X_test_audio_text, left_index=True, right_index=True)
    pipeline = Pipeline([
        ('scaler', CustomMinMaxScaler()),
        ('clustering', KMeans(n_clusters=2)),
        ('feature_selection', RFE(estimator=lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42))),
        ('model', XGBClassifier(random_state=42))
    ])
    param_grid = {
        'feature_selection__n_features_to_select': [10, 20, 30, 50, 70],
        'clustering__n_clusters': [2, 3, 4],
        'model__max_depth': [3, 5, 7],
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
    print("Training model...")
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, grid.predict(X_test)))
