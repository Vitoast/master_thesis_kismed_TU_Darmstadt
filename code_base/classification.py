from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np


outcome_descriptors = ["AKI3", "AKD1", "LCOS", "AF"]


def split_maps(train_map, test_map, outcome):
    feature_count = 0
    x_train, y_train = [], []
    for feature_name, feature_data in train_map.items():
        if feature_count == outcome:
            y_train = feature_data
        if feature_count > 3:
            x_train.append(feature_data)
        feature_count += 1

    feature_count = 0
    x_test, y_test = [], []
    for feature_name, feature_data in test_map.items():
        if feature_count == outcome:
            y_test = feature_data
        if feature_count > 3:
            x_test.append(feature_data)
        feature_count += 1

    return x_train, x_test, y_train, y_test


def classify_naive_bayes(train_data_map, test_data_map, result_path):
    for outcome in range(4):
        x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map, outcome)
        classifier = GaussianNB()
        classifier.fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
        y_pred = classifier.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        print(outcome_descriptors[outcome], " prediction accuracy of naive Bayes: ", accuracy_score(y_test, y_pred))
        print("    Class prior probabilities:", classifier.class_prior_)


def classify_linear_regression(train_data_map, test_data_map, result_path):
    for outcome in range(4):
        x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map, outcome)
        model = LinearRegression().fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
        y_pred = model.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(outcome_descriptors[outcome], " prediction accuracy of Linear Regression: MSE ", mse, "R^2 ", r2)
