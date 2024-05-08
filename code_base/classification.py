from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np

# Values to print names of adverse outcomes
outcome_descriptors = ["AKI3", "AKD1", "LCOS", "AF"]
# Name for Excel file with results
result_file_name = "classification_results.xlsx"


# Provides functionality to extract x and y train and test series from dictionaries
def split_maps(train_map, test_map, outcome):
    feature_count = 0
    x_train, y_train = [], []
    for feature_name, feature_data in train_map.items():
        # Outcomes are first 4 columns, use them as y
        if feature_count == outcome:
            y_train = feature_data
        # Rest is X
        if feature_count > 3:
            x_train.append(feature_data)
        feature_count += 1

    feature_count = 0
    x_test, y_test = [], []
    for feature_name, feature_data in test_map.items():
        # Outcomes are first 4 columns, use them as y
        if feature_count == outcome:
            y_test = feature_data
        # Rest is X
        if feature_count > 3:
            x_test.append(feature_data)
        feature_count += 1

    return x_train, x_test, y_train, y_test


# Provides printout for ordered feature importance in a decision tree or random forest
def print_feature_importances_of_forests(train_data_map, feature_importances):
    # Remove outcomes from key list to print feature importance
    feature_names = list(train_data_map.keys())
    for outcome_index in range(4):
        feature_names.pop(outcome_index)
    # Tie features together with their importance and print the sorted list, leaving out neglected features
    features_with_importance = list(zip(feature_names, feature_importances))
    features_with_importance_filtered = [(feature, importance) for feature, importance in features_with_importance
                                         if importance != 0]
    features_with_importance_sorted = sorted(features_with_importance_filtered, key=lambda x: x[1], reverse=True)
    print("    Features Importance:")
    for feature, importance in features_with_importance_sorted:
        print(f"        {feature}: {importance}")


# Save results to Excel file
def save_results_to_file(accuracy, result_path, classification_descriptor, parameter_descriptor):
    os.makedirs(result_path, exist_ok=True)
    result_file = os.path.join(result_path, result_file_name)
    # If the file does not exist, create it and add a first column with descriptors of the rows plus data column
    if not os.path.exists(result_file):
        df = pd.DataFrame({
            'Configuration': (['Classifier', 'Standardized', 'Imputed', 'Outlier_Filtered'] + outcome_descriptors),
            classification_descriptor: (parameter_descriptor + accuracy),
        })
    # Otherwise read file only add new column
    else:
        df = pd.read_excel(result_file)
        # Create column matching current configuration if non existent
        if classification_descriptor not in df.columns:
            df[classification_descriptor] = pd.NA
        # Overwrite data in column
        df[classification_descriptor][:len(parameter_descriptor + accuracy)] = (parameter_descriptor + accuracy)
    df.to_excel(result_file, index=False)


# Classify for each outcome with a naive Bayesian classifier
def classify(train_data_map, test_data_map, result_path, parameter_descriptor, classification_descriptor):
    accuracy_results = []
    parameter_descriptor = [classification_descriptor] + parameter_descriptor
    model_descriptor = ''
    for parameter in parameter_descriptor: model_descriptor += str(parameter)
    # Fit model, predict and evaluate accuracy for each outcome
    for outcome in range(4):
        x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map, outcome)

        classifier = GaussianNB()
        if classification_descriptor == 'LinearRegression': classifier = LinearRegression()
        elif classification_descriptor == 'DecisionTree': classifier = DecisionTreeClassifier()
        elif classification_descriptor == 'SVM': classifier = SVC()
        elif classification_descriptor == 'RandomForest': classifier = RandomForestClassifier()

        classifier.fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
        y_pred = classifier.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        if classification_descriptor == 'LinearRegression':
            predictions_class = [1 if pred >= 0.5 else 0 for pred in y_pred]
            accuracy_results.append(accuracy_score(y_test, predictions_class))
        else:
            accuracy_results.append(accuracy_score(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if classification_descriptor == 'NaiveBayes':
            print(outcome_descriptors[outcome], " prediction accuracy of naive Bayes: ", accuracy_results[outcome])
            print("    Class prior probabilities:", classifier.class_prior_)
        elif classification_descriptor == 'LinearRegression':
            print(outcome_descriptors[outcome], " Linear Regression: MSE ", mse, "R^2 ", r2, " prediction accuracy: ",
                  accuracy_results[outcome])
        elif classification_descriptor == 'DecisionTree':
            print(outcome_descriptors[outcome], " prediction accuracy of Decision Tree: ", accuracy_results[outcome])
            print("    Maximum Depth: ", classifier.get_depth(), "Number of Leaves:", classifier.get_n_leaves())
            feature_importances = classifier.feature_importances_
            print_feature_importances_of_forests(train_data_map, feature_importances)
            print("")
        elif classification_descriptor == 'SVM':
            print(outcome_descriptors[outcome], " prediction accuracy of SVM: ", accuracy_score(y_test, y_pred))
        elif classification_descriptor == 'RandomForest':
            print(outcome_descriptors[outcome], " prediction accuracy of Random Forest: ",
                  accuracy_results[outcome])
            print("    Max Depth of Trees:", classifier.max_depth, "Number of Trees:", len(classifier.estimators_))
            feature_importances = classifier.feature_importances_
            print_feature_importances_of_forests(train_data_map, feature_importances)

    save_results_to_file(accuracy_results, result_path, model_descriptor, parameter_descriptor)
    return accuracy_results


# Classify for each outcome with a naive Bayesian classifier
def classify_naive_bayes(train_data_map, test_data_map, result_path, parameter_descriptor):
    accuracy_results = []
    classification_descriptor = 'NaiveBayes'
    parameter_descriptor = [classification_descriptor] + parameter_descriptor
    for parameter in parameter_descriptor: classification_descriptor += str(parameter)
    # Fit model, predict and evaluate accuracy for each outcome
    for outcome in range(4):
        x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map, outcome)
        classifier = GaussianNB()
        classifier.fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
        y_pred = classifier.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        accuracy_results.append(accuracy_score(y_test, y_pred))
        print(outcome_descriptors[outcome], " prediction accuracy of naive Bayes: ", accuracy_score(y_test, y_pred))
        print("    Class prior probabilities:", classifier.class_prior_)
    save_results_to_file(accuracy_results, result_path, classification_descriptor, parameter_descriptor)
    return accuracy_results


# Classify for each outcome using linear regression
def classify_linear_regression(train_data_map, test_data_map, result_path, parameter_descriptor):
    accuracy_results = []
    classification_descriptor = 'LinearRegression'
    parameter_descriptor = [classification_descriptor] + parameter_descriptor
    for parameter in parameter_descriptor: classification_descriptor += str(parameter)
    # Fit model, predict and evaluate accuracy for each outcome
    for outcome in range(4):
        x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map, outcome)
        model = LinearRegression().fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
        y_pred = model.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        predictions_class = [1 if pred >= 0.5 else 0 for pred in y_pred]
        accuracy_results.append(accuracy_score(y_test, predictions_class))
        print(outcome_descriptors[outcome], " Linear Regression: MSE ", mse, "R^2 ", r2, " prediction accuracy: ",
              accuracy_score(y_test, predictions_class))
    save_results_to_file(accuracy_results, result_path, classification_descriptor, parameter_descriptor)
    return accuracy_results


# Classify for each outcome using a simple decision tree
def classify_decision_tree(train_data_map, test_data_map, result_path, parameter_descriptor):
    accuracy_results = []
    classification_descriptor = 'DecisionTree'
    parameter_descriptor = [classification_descriptor] + parameter_descriptor
    for parameter in parameter_descriptor: classification_descriptor += str(parameter)
    # Fit model, predict and evaluate accuracy for each outcome
    for outcome in range(4):
        x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map, outcome)
        classifier = DecisionTreeClassifier()
        classifier.fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
        y_pred = classifier.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_results.append(accuracy)
        print(outcome_descriptors[outcome], " prediction accuracy of Decision Tree: ", accuracy)
        print("    Maximum Depth: ", classifier.get_depth(), "Number of Leaves:", classifier.get_n_leaves())
        feature_importances = classifier.feature_importances_
        print_feature_importances_of_forests(train_data_map, feature_importances)
        print("")
    save_results_to_file(accuracy_results, result_path, classification_descriptor, parameter_descriptor)
    return accuracy_results


# Classify outcomes by SVM
def classify_svm(train_data_map, test_data_map, result_path, parameter_descriptor):
    accuracy_results = []
    classification_descriptor = 'SVM'
    parameter_descriptor = [classification_descriptor] + parameter_descriptor
    for parameter in parameter_descriptor: classification_descriptor += str(parameter)
    # Fit model, predict and evaluate accuracy for each outcome
    for outcome in range(4):
        x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map, outcome)
        classifier = SVC()
        classifier.fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
        y_pred = classifier.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        accuracy_results.append(accuracy_score(y_test, y_pred))
        print(outcome_descriptors[outcome], " prediction accuracy of SVM: ", accuracy_score(y_test, y_pred))
    save_results_to_file(accuracy_results, result_path, classification_descriptor, parameter_descriptor)
    return accuracy_results


# Classify outcomes by Random Forest
def classify_random_forest(train_data_map, test_data_map, result_path, parameter_descriptor):
    accuracy_results = []
    classification_descriptor = 'RandomForest'
    parameter_descriptor = [classification_descriptor] + parameter_descriptor
    for parameter in parameter_descriptor: classification_descriptor += str(parameter)
    # Fit model, predict and evaluate accuracy for each outcome
    for outcome in range(4):
        x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map, outcome)
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
        y_pred = classifier.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        feature_importances = classifier.feature_importances_
        accuracy_results.append(accuracy_score(y_test, y_pred))
        print(outcome_descriptors[outcome], " prediction accuracy of Random Forest: ", accuracy_score(y_test, y_pred))
        print("    Max Depth of Trees:", classifier.max_depth, "Number of Trees:", len(classifier.estimators_))
        print_feature_importances_of_forests(train_data_map, feature_importances)
    save_results_to_file(accuracy_results, result_path, classification_descriptor, parameter_descriptor)
    return accuracy_results
