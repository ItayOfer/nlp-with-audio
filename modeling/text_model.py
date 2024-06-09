import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

import utils
from sklearn.model_selection import GridSearchCV
from feature_engineering import FeatureEngineering
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from modeling.clustering import ClusterTransformer
from modeling.scaler import CustomMinMaxScaler

if __name__ == '__main__':
    train_data, test_data, y_train, y_test = utils.get_data()
    text_feature_name = [col for col in train_data.columns if col.startswith('text_feature_')]
    fe = FeatureEngineering('../MELD.Raw/train/train_sent_emo.csv')
    fe_train_data = fe.run()
    test_meld = pd.read_csv('../MELD.Raw/test_sent_emo.csv')
    dev_meld = pd.read_csv('../MELD.Raw/dev_sent_emo.csv')
    fe_dev_data = fe.run(dev_meld, train=False)
    fe_test_data = fe.run(test_meld, train=False)
    # fe_train_data = pd.concat([fe_train_data, fe_dev_data])
    common_indices = fe_train_data.index.intersection(y_train.index)
    fe_train_data = fe_train_data.loc[common_indices]
    X_train = train_data[text_feature_name]
    X_test = test_data[text_feature_name]
    X_train = pd.merge(X_train, fe_train_data, left_index=True, right_index=True)
    X_test = pd.merge(X_test, fe_test_data, left_index=True, right_index=True)

    pipeline = Pipeline([
        ('scaler', CustomMinMaxScaler()),
        ('clustering', ClusterTransformer(cluster_type='kmeans')),
        ('feature_selection', RFE(estimator=lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42))),
        ('model',  RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'feature_selection__n_features_to_select': [10, 20, 30, 70],
        'model__max_depth': [5, 10, 15],
        'clustering__n_clusters': [2, 3, 4],
        # 'model__max_depth': [5, 10, 15],
        # 'model__learning_rate': [0.01, 0.05, 0.1],
        # 'model__n_estimators': [50, 100, 200]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
    print("Training model...")
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, grid.predict(X_test)))
