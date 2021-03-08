# This file contains the NON-SENSITIVE, source controlled configuration variables that your app needs.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Data loading parameters
TIME_THRESHOLD = 10
TEST_SIZE = 0.25
RANDOM_STATE = 10

# Raw data path
RAW_DATA_PATH = './risk_models/data/NHANES_I_epidemiology.csv'

# Model configuration for training
MODEL = RandomForestClassifier(random_state=RANDOM_STATE)
HYPERPARAMS = {
                'n_estimators': [150, 200, 250, 300],
                'max_depth': [3, 4, 5, 6],
                'min_samples_leaf': [1, 2, 3, 4],
            }

CROSS_VALIDATION = 5

# Risk score threshold which used to classify target class
RISK_THRESHOLD = 0.5