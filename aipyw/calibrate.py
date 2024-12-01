import xgboost as xgb
import numpy as np


def isoreg_with_xgboost(x, y, max_depth=15, min_child_weight=20, weights=None):
    """
    Fits isotonic regression using XGBoost with monotonic constraints to ensure
    non-decreasing predictions as the predictor variable increases.

    Args:
        x (np.array): A vector or matrix of predictor variables.
        y (np.array): A vector of response variables.
        max_depth (int, optional): Maximum depth of the trees in XGBoost.
                                   Default is 15.
        min_child_weight (float, optional): Minimum sum of instance weights
                                            needed in a child node. Default is 20.
        weights (np.array, optional): A vector of weights for each instance.
                                      If None, all instances are equally weighted.

    Returns:
        function: A prediction function that takes a new predictor variable x
                  and returns the model's predicted values.

    h/t Lars van der Laan https://github.com/Larsvanderlaan/CDML/

    """

    # Create an XGBoost DMatrix object from the data with optional weights
    data = xgb.DMatrix(
        data=np.asarray(x).reshape(len(y), -1), label=np.asarray(y), weight=weights
    )

    # Set parameters for the monotonic XGBoost model
    params = {
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "monotone_constraints": "(1)",  # Enforce monotonic increase
        "eta": 1,
        "gamma": 0,
        "lambda": 0,
    }

    # Train the model with one boosting round
    iso_fit = xgb.train(params=params, dtrain=data, num_boost_round=1)

    # Prediction function for new data
    def predict_fn(x):
        """
        Predicts output for new input data using the trained isotonic regression model.

        Args:
            x (np.array): New predictor variables as a vector or matrix.

        Returns:
            np.array: Predicted values.
        """
        x = np.atleast_2d(x).T if x.ndim == 1 else x
        data_pred = xgb.DMatrix(data=x)
        pred = iso_fit.predict(data_pred)
        return pred

    return predict_fn
