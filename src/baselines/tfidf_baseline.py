from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class TfidfBaseline:
    """
    5-fold out-of-fold evaluation on 200 hand-labelled examples.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)

    def fit(self, X_train, y_train):
        """
        Trains TF-IDF vectorizer and LogisticRegression on provided training data.
        """
        X_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_vec, y_train)

    def predict(self, X_test):
        """
        Predicts intent for test data.
        """
        X_vec = self.vectorizer.transform(X_test)
        preds = self.classifier.predict(X_vec)
        return preds
