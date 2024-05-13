from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
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

# Values to print names of adverse outcomes
outcome_descriptors = ["AKI3", "AKD1", "LCOS", "AF", "Any"]
# Name for Excel file with results
result_file_name = "classification_results.xlsx"
# Number of adverse outcomes in data set
number_outcomes = 5


# Provides functionality to extract x and y train and test series from dictionaries
def split_maps(train_map, test_map):
    feature_count = 0
    x_train, y_train = [], []
    for feature_name, feature_data in train_map.items():
        # Outcomes are first 5 columns, use them as y
        if feature_count in range(number_outcomes):
            y_train.append(feature_data)
        # Rest is X
        else:
            x_train.append(feature_data)
        feature_count += 1

    feature_count = 0
    x_test, y_test = [], []
    for feature_name, feature_data in test_map.items():
        # Outcomes are first 5 columns, use them as y
        if feature_count in range(number_outcomes):
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
    for outcome_index in range(number_outcomes):
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
        accuracy_descriptors = [s + '_accuracy' for s in outcome_descriptors]
        f1_descriptors = [s + '_f1_score' for s in outcome_descriptors]
        df = pd.DataFrame({
            'Configuration': (['Classifier', 'Standardized', 'Imputed', 'Outlier_Filtered', 'k-folds']
                              + accuracy_descriptors + f1_descriptors),
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


# Classify for each outcome with a naive Bayesian classifier
def classify(train_data_map, test_data_map, result_path, parameter_descriptor, classification_descriptor,
             print_model_details=False):
    x_train, x_test, y_train, y_test = split_maps(train_data_map, test_data_map)
    # Create unique identifier for current model
    model_descriptor = ''
    parameter_descriptor = [classification_descriptor] + parameter_descriptor + [0]
    for parameter in parameter_descriptor: model_descriptor += str(parameter)
    # Train model and predict
    accuracy_results, f1_scores = classify_internal(x_train, x_test, y_train, y_test, train_data_map, test_data_map,
                                                    result_path, parameter_descriptor, classification_descriptor,
                                                    print_model_details)
    # Save results and return accuracy
    save_results_to_file(accuracy_results, f1_scores, result_path, model_descriptor, parameter_descriptor)


def classify_internal(x_train, x_test, y_train, y_test, train_data_map, test_data_map,
                      result_path, parameter_descriptor, classification_descriptor, print_model_details=False):
    accuracy_results, f1_scores = [], []
    # Fit model, predict and evaluate accuracy for each outcome
    for outcome in range(number_outcomes):
        # Choose desired classifier
        classifier = GaussianNB()
        if classification_descriptor == 'LinearRegression':
            classifier = LinearRegression()
        elif classification_descriptor == 'DecisionTree':
            classifier = DecisionTreeClassifier()
        elif classification_descriptor == 'SVM':
            classifier = SVC()
        elif classification_descriptor == 'RandomForest':
            classifier = RandomForestClassifier()
        # Fit model and predict on test set
        classifier.fit(np.reshape(x_train, (len(x_train[0]), len(x_train))), y_train[outcome])
        y_pred = classifier.predict(np.reshape(x_test, (len(x_test[0]), len(x_test))))
        # Compute prediction accuracy
        # In case of regression extract classes from prediction
        if classification_descriptor == 'LinearRegression':
            y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred]
        accuracy_results.append(accuracy_score(y_test[outcome], y_pred))
        # Compute error measures
        f1_scores.append(f1_score(y_test[outcome], y_pred))
        mse = mean_squared_error(y_test[outcome], y_pred)
        r2 = r2_score(y_test[outcome], y_pred)
        # Print out information about the classification model
        if print_model_details:
            if classification_descriptor == 'NaiveBayes':
                print(outcome_descriptors[outcome], " prediction accuracy of naive Bayes: ", accuracy_results[outcome])
                print("    Class prior probabilities:", classifier.class_prior_)
            elif classification_descriptor == 'LinearRegression':
                print(outcome_descriptors[outcome], " Linear Regression: MSE ", mse, "R^2 ", r2,
                      " prediction accuracy: ",
                      accuracy_results[outcome])
            elif classification_descriptor == 'DecisionTree':
                print(outcome_descriptors[outcome], " prediction accuracy of Decision Tree: ",
                      accuracy_results[outcome])
                print("    Maximum Depth: ", classifier.get_depth(), "Number of Leaves:", classifier.get_n_leaves())
                feature_importances = classifier.feature_importances_
                print_feature_importances_of_forests(train_data_map, feature_importances)
                print("")
            elif classification_descriptor == 'SVM':
                print(outcome_descriptors[outcome], " prediction accuracy of SVM: ", accuracy_results[outcome])
            elif classification_descriptor == 'RandomForest':
                print(outcome_descriptors[outcome], " prediction accuracy of Random Forest: ",
                      accuracy_results[outcome])
                print("    Max Depth of Trees:", classifier.max_depth, "Number of Trees:", len(classifier.estimators_))
                feature_importances = classifier.feature_importances_
                print_feature_importances_of_forests(train_data_map, feature_importances)
    return accuracy_results, f1_scores


# Classify with using k-fold cross validation
def classify_k_fold(data_map, result_path, parameter_descriptor, classification_descriptor, print_model_details=False):
    k_fold_split = 5
    kf = KFold(n_splits=k_fold_split)
    # Extract x and y from map
    x, y = [], []
    # Convert map into processable x and y
    feature_count = 0
    for feature_name, feature_data in data_map.items():
        # Outcomes are first 4 columns, use them as y
        if feature_count in range(number_outcomes):
            y.append(feature_data)
        # Rest is X
        else:
            x.append(feature_data)
        feature_count += 1
    # Create unique identifier for current model
    model_descriptor = ''
    parameter_descriptor = [classification_descriptor] + parameter_descriptor + [k_fold_split]
    for parameter in parameter_descriptor: model_descriptor += str(parameter)
    # Store accuracy results here
    accuracy_results, f1_score_results = [], []
    # Train and evaluate the model for each fold
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        # current_x = x[:, train_index]
        current_train_x = [[arr[i] for i in train_index] for arr in x]
        current_train_y = [[arr[i] for i in train_index] for arr in y]
        current_test_x = [[arr[i] for i in test_index] for arr in x]
        current_test_y = [[arr[i] for i in test_index] for arr in y]
        accuracy_value, f1_score_value = classify_internal(current_train_x, current_test_x, current_train_y,
                                                           current_test_y,
                                                           data_map, [],
                                                           result_path, parameter_descriptor, classification_descriptor,
                                                           print_model_details)
        accuracy_results.append(accuracy_value)
        f1_score_results.append(f1_score_value)
    # Save average accuracy to file
    save_results_to_file(np.mean(accuracy_results, axis=0), np.mean(f1_score_results, axis=0), result_path, model_descriptor,
                         parameter_descriptor)
