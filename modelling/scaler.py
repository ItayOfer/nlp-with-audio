from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class CustomMinMaxScaler(MinMaxScaler):
    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__(feature_range=feature_range, copy=copy)

    def transform(self, X):
        # Perform the scaling
        result = super().transform(X)
        # Convert numpy array back to DataFrame, preserving column names
        if isinstance(X, pd.DataFrame):
            result_df = pd.DataFrame(result, index=X.index, columns=X.columns)
            return result_df
        return result

    def fit_transform(self, X, y=None):
        # Fit the scaling
        self.fit(X)
        # Transform the data and convert to DataFrame
        return self.transform(X)
