from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
import utils
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    train_data, test_data, y_train, y_test = utils.get_data()
    text_feature_name = [col for col in train_data.columns if col.startswith('text_feature_')]
    X_train = train_data[text_feature_name]
    X_test = test_data[text_feature_name]
    param_grid = {
        'max_depth': [3, 6, 9],
        'eta': [0.1, 0.3, 0.5],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'objective': ['binary:logistic']  # set the objective for binary classification
    }
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # fit model
    grid_search_text = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
    grid_search_text.fit(X_train, y_train)
    best_xgb_model_text = grid_search_text.best_estimator_
    preds_text = best_xgb_model_text.predict(X_test)
    accuracy_text = accuracy_score(y_test, preds_text)
