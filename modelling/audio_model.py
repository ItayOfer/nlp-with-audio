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


if __name__ == '__main__':
    train_data, test_data, y_train, y_test = utils.get_data()
    audio_feature_name = [col for col in train_data.columns if col.startswith('audio_feature_')]
    test_meld = pd.read_csv('../MELD.Raw/test_sent_emo.csv')
    X_train = train_data[audio_feature_name]
    X_test = test_data[audio_feature_name]
    scaler = MinMaxScaler()
    pipeline = Pipeline([
        ('scaler', scaler),
        ('feature_selection', RFE(estimator=lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42))),
        ('model', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'feature_selection__n_features_to_select': [10, 20, 30, 70],
        'model__max_depth': [5, 10, 15]
        # 'model__max_depth': [5, 10, 15],
        # 'model__learning_rate': [0.01, 0.05, 0.1],
        # 'model__n_estimators': [50, 100, 200]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)
    print("Training model...")
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)  #
    print("Accuracy:", accuracy_score(y_test, grid.predict(X_test)))  #
