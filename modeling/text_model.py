import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import utils
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from feature_engineering import FeatureEngineering
from mrmr import mrmr_classif

if __name__ == '__main__':
    train_data, test_data, y_train, y_test = utils.get_data()
    text_feature_name = [col for col in train_data.columns if col.startswith('text_feature_')]
    fe = FeatureEngineering('../MELD.Raw/train/train_sent_emo.csv')
    fe_train_data = fe.run()
    test_meld = pd.read_csv('../MELD.Raw/test_sent_emo.csv')
    dev_meld = pd.read_csv('../MELD.Raw/dev_sent_emo.csv')
    fe_dev_data = fe.run(dev_meld, train=False)
    fe_test_data = fe.run(test_meld, train=False)
    fe_train_data = pd.concat([fe_train_data, fe_dev_data])
    common_indices = fe_train_data.index.intersection(y_train.index)
    fe_train_data = fe_train_data.loc[common_indices]
    X_train = train_data[text_feature_name]
    X_test = test_data[text_feature_name]
    X_train = pd.merge(X_train, fe_train_data, left_index=True, right_index=True)
    X_test = pd.merge(X_test, fe_test_data, left_index=True, right_index=True)
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

