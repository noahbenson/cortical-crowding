"""Code related to regression for the corticalcrowding library.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def fit_and_evaluate(X, Y, logerror=True):
    # set intercept = 0
    model = LinearRegression(fit_intercept = False)
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    y_pred = model.predict(X)
    # residual sum of squares
    if logerror:
        rss = np.sum((np.log10(Y) - np.log10(y_pred)) ** 2)
    else:
        rss = np.sum((Y - y_pred) ** 2)
    return rss, model.coef_[0]


