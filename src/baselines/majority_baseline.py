import pandas as pd

class MajorityBaseline:
    """
    5-fold out-of-fold evaluation on 200 hand-labelled examples.
    """
    def __init__(self):
        self.majority_intent = None

    def fit(self, y_train):
        """
        Finds majority intent from the provided training labels.
        """
        y_series = pd.Series(y_train)
        self.majority_intent = y_series.mode()[0]

    def predict(self, X_test):
        """
        Predicts the majority intent for all test examples.
        """
        return [self.majority_intent] * len(X_test)
