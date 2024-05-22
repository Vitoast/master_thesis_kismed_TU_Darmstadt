from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import pandas as pd
import os
import numpy as np
import preprocess_data as pre
import global_variables as gl

# Name for Excel file with results
result_file_name = "classification_results.xlsx"


# Provides functionality to extract x and y train and test series from dictionaries
def split_maps(train_map, test_map):
    feature_count = 0
    x_train, y_train = [], []
    for feature_name, feature_data in train_map.items():
        # Outcome is in first column, use them as y
        if feature_count == 0:
            y_train.append(feature_data)
        # Rest is X
        else:
            x_train.append(feature_data)
        feature_count += 1
    feature_count = 0
    x_test, y_test = [], []
    for feature_name, feature_data in test_map.items():
        # Outcome is in first column, use them as y
        if feature_count == 0:
            y_test.append(feature_data)
        # Rest is X
        else:
            x_test.append(feature_data)
        feature_count += 1

    return x_train, x_test, y_train, y_test


# Provides printout for ordered feature importance in a decision tree or random forest
def print_feature_importances_of_forests(train_data_map, feature_importances):
    # Remove outcomes from key list to print feature importance
    feature_names = list(train_data_map.keys())
    for outcome_index in range(gl.number_outcomes):
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
def save_results_to_file(accuracy, f1_scores, result_path, classification_descriptor, parameter_descriptor):
    os.makedirs(result_path, exist_ok=True)
    result_file = os.path.join(result_path, result_file_name)
    str_parameters = [str(num) for num in parameter_descriptor]
    str_accuracies = [str(num) for num in accuracy]
    str_f1_scores = [str(num) for num in f1_scores]
    # If the file does not exist, create it and add a first column with descriptors of the rows plus data column
    if not os.path.exists(result_file):
        # accuracy_descriptors = [s + '_accuracy'for s in gl.outcome_descriptors]
        # f1_descriptors = [s + '_f1_score' for s in gl.outcome_descriptors]
        df = pd.DataFrame({
            'Configuration': (['Outcome', 'Classifier', 'Standardized', 'Imputer', 'Z-Score Threshold', 'Oversampling',
                               'K-folds', 'Accuracy', 'F1-Score']),
            classification_descriptor: (str_parameters + str_accuracies + str_f1_scores),
        })
    # Otherwise read file only add new column
    else:
        df = pd.read_excel(result_file)
        # Create column matching current configuration if non existent
        if classification_descriptor not in df.columns:
            df[classification_descriptor] = pd.NA
        # Overwrite data in column
        new_column_content = str_parameters + str_accuracies + str_f1_scores
        df[classification_descriptor][:len(new_column_content)] = new_column_content
    df.to_excel(result_file, index=False)


# Classify for one outcome with a naive Bayesian classifier
def classify(train_data_map, test_data_map, outcome_target_index, result_path, parameter_descriptor,
             classification_descriptor, print_model_details=False, save_model_details=True):
    train_data_map, test_data_map = pre.preprocess_data(train_data_map, test_data_map, outcome_target_index,
                                                        standardize=gl.standardize,
                                                        impute=gl.impute, z_score_threshold=gl.filter_outliers_z_score,
                                                        oversample_rate=gl.oversample)
    x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map)
    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()
    # Create unique identifier for current model
    model_descriptor = ''
    parameter_descriptor = [classification_descriptor] + [
        gl.outcome_descriptors[outcome_target_index]] + parameter_descriptor + [0]
    for parameter in parameter_descriptor: model_descriptor += str(parameter)
    # Train model and predict
    accuracy_results, f1_scores = classify_internal(x_train, x_test, y_train, y_test, train_data_map, test_data_map,
                                                    outcome_target_index, result_path, parameter_descriptor,
                                                    classification_descriptor, print_model_details)
    # Save results and return accuracy
    if save_model_details:
        save_results_to_file(accuracy_results, f1_scores, result_path, model_descriptor, parameter_descriptor)
    return accuracy_results, f1_scores


def classify_internal(x_train, x_test, y_train, y_test, train_data_map, test_data_map, outcome_target_index,
                      result_path, parameter_descriptor, classification_descriptor, print_model_details=False):
    accuracy_results, f1_scores = [], []
    # Fit model, predict and evaluate accuracy for the desired outcome
    # Choose desired classifier
    classifier = GaussianNB()
    if classification_descriptor == 'LogisticRegression':
        classifier = LogisticRegression()
    elif classification_descriptor == 'DecisionTree':
        classifier = DecisionTreeClassifier()
    elif classification_descriptor == 'SVM':
        classifier = SVC()
    elif classification_descriptor == 'RandomForest':
        classifier = RandomForestClassifier()
    # Fit model and predict on test set
    classifier.fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train)
    y_pred = classifier.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
    # Compute prediction accuracy
    accuracy_results.append(accuracy_score(y_test, y_pred))
    # Compute error measures
    f1_scores.append(f1_score(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Print out information about the classification model
    if print_model_details:
        if classification_descriptor == 'NaiveBayes':
            print(gl.outcome_descriptors[outcome_target_index], " prediction accuracy of naive Bayes: ",
                  accuracy_results[outcome_target_index])
            print("    Class prior probabilities:", classifier.class_prior_)
        elif classification_descriptor == 'LogisticRegression':
            print(gl.outcome_descriptors[outcome_target_index], " Linear Regression: MSE ", mse, "R^2 ", r2,
                  " prediction accuracy: ",
                  accuracy_results[outcome_target_index])
        elif classification_descriptor == 'DecisionTree':
            print(gl.outcome_descriptors[outcome_target_index], " prediction accuracy of Decision Tree: ",
                  accuracy_results[outcome_target_index])
            print("    Maximum Depth: ", classifier.get_depth(), "Number of Leaves:", classifier.get_n_leaves())
            feature_importances = classifier.feature_importances_
            print_feature_importances_of_forests(train_data_map, feature_importances)
            print("")
        elif classification_descriptor == 'SVM':
            print(gl.outcome_descriptors[outcome_target_index], " prediction accuracy of SVM: ",
                  accuracy_results[outcome_target_index])
        elif classification_descriptor == 'RandomForest':
            print(gl.outcome_descriptors[outcome_target_index], " prediction accuracy of Random Forest: ",
                  accuracy_results[outcome_target_index])
            print("    Max Depth of Trees:", classifier.max_depth, "Number of Trees:", len(classifier.estimators_))
            feature_importances = classifier.feature_importances_
            print_feature_importances_of_forests(train_data_map, feature_importances)
    return accuracy_results, f1_scores


# Classify with using k-fold cross validation
def classify_k_fold(data_map, outcome, result_path, parameter_descriptor, classification_descriptor,
                    print_model_details=False, save_model_details=True):
    # Prepare k split
    k_fold_split = 5
    kf = KFold(n_splits=k_fold_split, shuffle=True)
    # Create unique identifier for current model
    model_descriptor = ''
    parameter_descriptor = [gl.outcome_descriptors[outcome]] + [classification_descriptor] + parameter_descriptor + [
        k_fold_split]
    for parameter in parameter_descriptor: model_descriptor += str(parameter)
    # Store accuracy results here
    accuracy_results, f1_score_results = [], []
    # Perform classification for each split
    for train_index, test_index in kf.split(next(iter(data_map.values()))):
        # Get train and test sets from the complete data set
        train_data_map = {}
        test_data_map = {}
        for feature_name, feature_map in data_map.items():
            train_data_map[feature_name] = [feature_map[i] for i in train_index]
            test_data_map[feature_name] = [feature_map[i] for i in test_index]
        # Train and evaluate the model for each fold
        accuracy_value, f1_score_value = classify(train_data_map, test_data_map, outcome, result_path,
                                                  parameter_descriptor, classification_descriptor,
                                                  print_model_details, False)
        accuracy_results.append(accuracy_value)
        f1_score_results.append(f1_score_value)
    # Save average accuracy to file
    if save_model_details:
        save_results_to_file(np.mean(accuracy_results, axis=0), np.mean(f1_score_results, axis=0), result_path,
                             model_descriptor, parameter_descriptor)
    return np.mean(accuracy_results, axis=0), np.mean(f1_score_results, axis=0)
