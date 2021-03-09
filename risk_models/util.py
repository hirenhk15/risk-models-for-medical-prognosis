# All the methods related to load and split the data

import os
import lifelines
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)


def cindex(y_true: np.array, scores: np.array) -> float:
    """AI is creating summary for cindex

    Args:
        y_true (np.array): An array of actual values of target
        scores (np.array): An array of predicted score of target

    Returns:
        [float]: Returns C-Index score
    """
    return lifelines.utils.concordance_index(y_true, scores)


def load_data(threshold: int):
    """This method loads and split the data into dev and test

    Args:
        threshold (int): Time threshold value to calulate risk for

    Returns:
        [pd.DataFrame]: Returns splited dev and test data
    """
    X, y = nhanesi()
    df = X.drop([X.columns[0]], axis=1)
    df.loc[:, 'time'] = y
    df.loc[:, 'death'] = np.ones(len(X))
    df.loc[df.time < 0, 'death'] = 0
    df.loc[:, 'time'] = np.abs(df.time)
    df = df.dropna(axis='rows')
    mask = (df.time > threshold) | (df.death == 1)
    df = df[mask]
    X = df.drop(['time', 'death'], axis='columns')
    y = df.time < threshold

    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    feature_y = 'Systolic BP'
    frac = 0.7

    drop_rows = X_dev.sample(frac=frac, replace=False,
                             weights=[prob_drop(X_dev.loc[i, 'Age']) for i in
                                      X_dev.index], random_state=10)
    drop_rows.loc[:, feature_y] = None
    drop_y = y_dev[drop_rows.index]
    X_dev.loc[drop_rows.index, feature_y] = None

    return X_dev, X_test, y_dev, y_test


def prob_drop(age: float):
    """
    Returns specific probability distribution
    """
    return 1 - (np.exp(0.25 * age - 5) / (1 + np.exp(0.25 * age - 5)))


def nhanesi(display=False):
    """
    Read raw csv data and separate features and target
    """
    data = pd.read_csv(os.path.join(__location__, './data/NHANES_I_epidemiology.csv'))
    X = data.drop('y', axis=1)
    y = data['y']

    if display:
        X_display = X.copy()
        X_display["Sex"] = ["Male" if v == 1 else "Female" for v in X["Sex"]]
        return X_display, np.array(y)
    return X, np.array(y)
    

class DataLoader:
    """
    This class loads and splits the data into training, validation and test sets.
    """
    def __init__(self):
        pass

    def load_and_split(self, threshold: int, test_size: float, random_state: int):
        # Load the data
        X_dev, X_test, y_dev, y_test = load_data(threshold)

        # Split the data according to the test size
        X_train, X_val, y_train, y_val = train_test_split(
            X_dev, y_dev, test_size=test_size, random_state=random_state
            )

        return X_train, X_val, X_test, y_train, y_val, y_test