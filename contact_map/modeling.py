# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""Train and score few-shot L1-logit model predicting contacts using
self-attention maps.
"""

from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import metrics


def train_logistic_regression(X, y):
    """Train a l1-logit classifier on dataset.
    
    Args:
        X: np.array of shape [n_samples, n_features]
        y: np.array of shape [n_samples, ]

    Returns:
        Trained sklear Logistic Regression model instance
    """
    l1_logit = LogisticRegression(
        penalty="l1", solver="saga", C=0.15, random_state=42, n_jobs=-1
    )
    l1_logit.fit(X, y)
    return l1_logit


def save_logistic_regression(model, fp):
    """Save sklearn logistic regression model
    
    Args:
        model: Model instnace to be stored.
        fp: String specifying the filepath where the model will be stored.
    
    Returns:
        None
    """
    pickle.dump(model, open(fp, "wb"))


def load_logistic_regression(fp):
    """Load sklearn logistic regression model
    
    Args:
        fp: String specifying the filepath where the model will be stored.
    
    Returns:
        Loaded model instnace.
    """
    l1_logit = pickle.load(open(fp, "rb"))
    return l1_logit


def compute_scores(model, X, y):
    """
    Compute average scores: precision score and average precision score.
    
    Args:
        model: trained l1-logit classifier
        X: np.array of shape [n_samples, n_features]
        y: np.array of shape [n_samples, ]
    
    Returns:
        Tuple whith the precision score in the first elemnt and the average precision score in the second element. 

    """
    y_hat_prob = model.predict_proba(X)
    y_hat = model.predict(X)

    prec_score = metrics.precision_score(y, y_hat)
    avg_prec_score = metrics.average_precision_score(y, y_hat_prob[:, 1])

    return prec_score, avg_prec_score
