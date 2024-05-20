import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import global_variables as gl


# Returns an array of booleans that tell if a point is outside the threshold * std_deviation from the median
def filter_by_z_score(data, center, threshold):
    z_scores = np.abs((data - center) / np.std(data))
    return z_scores < threshold


# Preprocess data to make it usable in classification
# Involves: Imputation of missing values, standardization and outlier filtering
def preprocess_data(train_data_dictionary, test_data_dictionary, outcome_target_index, standardize, impute,
                    z_score_threshold):
    # Create imputation instance to replace nan values in data
    if 'mean' in impute:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif 'median' in impute:
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    else:
        print('Invalid imputation method')
        return
    scaler = StandardScaler()
    # Threshold for outlier filtering, all values further away than threshold * std_deviation from median are abandoned
    feature_to_remove = []
    # Count features but leave out ID in first column
    feature_count = 0
    remove_points_mask = np.ones(len(train_data_dictionary['ID']), dtype=bool)
    outcome_descriptor = list(train_data_dictionary.keys())[outcome_target_index + 1]
    train_outcome_values = train_data_dictionary[outcome_descriptor]
    # If requested impute missing values in set by treating each outcome class separately
    if 'group' in impute:
        imputed_train_data = []
        for class_value in np.unique(train_outcome_values):
            subset_train = train_outcome_values == class_value
            class_subset = pd.DataFrame(train_data_dictionary)[subset_train]
            # imputer.fit_transform(np.reshape(class_subset, (-1, 1)))
            t = class_subset.median()
            idf = pd.DataFrame(class_subset.fillna(class_subset.median()))
            imputed_train_data.append(idf)
        train_data_dictionary = pd.concat(imputed_train_data)
        new_data_dictionary = {}
        for feature_name, feature_data in train_data_dictionary.items():
            tmp = feature_data.tolist()
            new_data_dictionary[feature_name] = tmp
        train_data_dictionary = new_data_dictionary
    # For Test set do imputation and standardization with parameter from training set if requested
    for (train_feature_name, train_feature_data), (test_feature_name, test_feature_data) in zip(
            train_data_dictionary.items(), test_data_dictionary.items()):
        # Preprocess only numerical data, remove all features that are strings
        if not any(isinstance(elem, str) for elem in train_feature_data):
            # Remove first columns with outcomes
            if feature_count <= gl.number_outcomes and feature_count != outcome_target_index + 1:
                feature_to_remove.append(train_feature_name)
            else:
                if 'group' in impute:
                    test_data_dictionary[train_feature_name] = np.reshape(imputer.fit_transform(
                        np.reshape(test_feature_data, (-1, 1))), (-1,))
                else:
                    train_data_dictionary[train_feature_name] = np.reshape(imputer.fit_transform(
                        np.reshape(train_feature_data, (-1, 1))), (-1,))
                    test_data_dictionary[train_feature_name] = np.reshape(imputer.transform(
                        np.reshape(test_feature_data, (-1, 1))), (-1,))
                # If there are still nan values after imputation then there were no values in the first place
                if any(np.isnan(test_feature_data)) or any(np.isnan(test_feature_data)):
                    feature_to_remove.append(train_feature_name)
                    feature_count += 1
                    # Skip standardization here to avoid errors
                    continue
                # Standardize and filter only if requested and non-categorical
                if not np.all(np.isin(train_feature_data, [0, 1])):
                    if standardize:
                        train_data_dictionary[train_feature_name] = np.reshape(scaler.fit_transform(
                            np.reshape(train_data_dictionary[train_feature_name], (-1, 1))), (-1,))
                        test_data_dictionary[test_feature_name] = np.reshape(scaler.transform(
                            np.reshape(test_data_dictionary[test_feature_name], (-1, 1))), (-1,))
                    if z_score_threshold > 0:
                        remove_points_mask &= filter_by_z_score(train_data_dictionary[train_feature_name],
                                                                np.median(train_data_dictionary[train_feature_name]),
                                                                z_score_threshold)
        else:
            feature_to_remove.append(train_feature_name)
        feature_count += 1
    # Remove all the features that are not suitable for classification
    for name in feature_to_remove:
        train_data_dictionary.pop(name)
        test_data_dictionary.pop(name)
    # Remove patients with invalid outlying points from the training feature set
    if z_score_threshold > 0:
        for feature_name, feature_data in train_data_dictionary.items():
            train_data_dictionary[feature_name] = feature_data[remove_points_mask]
        # print('number of points removed', np.count_nonzero(np.invert(remove_points_mask)), 'with threshold', z_score_threshold)
    return train_data_dictionary, test_data_dictionary
