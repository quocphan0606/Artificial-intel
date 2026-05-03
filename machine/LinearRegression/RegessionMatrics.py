import numpy as np

class RegressionMetrics:

    @staticmethod
    def mse(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mae(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(RegressionMetrics.mse(y_true, y_pred))

    @staticmethod
    def r2(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def adjusted_r2(y_true, y_pred, num_features):
        n = len(y_true)
        r2 = RegressionMetrics.r2(y_true, y_pred)
        return 1 - (1 - r2) * (n - 1) / (n - num_features - 1)