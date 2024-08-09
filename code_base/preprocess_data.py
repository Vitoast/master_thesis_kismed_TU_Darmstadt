import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import global_variables as gl


# Returns an array of booleans that tell if a point is outside the threshold * std_deviation from the median
def filter_by_z_score(data, center, threshold):
    if np.std(data) != 0:
        z_scores = np.abs((data - center) / np.std(data))
    else:
        z_scores = np.zeros(len(data))
    return z_scores <= threshold


# Preprocess data to make it usable in classification
# Involves: Imputation of missing values, standardization, outlier filtering and oversampling
# Note: If grouped imputation or oversampling is enabled, order of data points will change
def preprocess_data(train_data_dictionary, test_data_dictionary, outcome_target_index, standardize=gl.standardize,
                    impute=gl.impute, z_score_threshold=gl.filter_outliers_z_score, oversample_rate=gl.oversample):
    # Create imputation instance to replace nan values in data
    if 'mean' in impute:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    elif 'median' in impute:
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    else:
        print('Invalid imputation method')
        return

    # Create copies of the raw input to keep a reference
    ref_train_data_dictionary = train_data_dictionary.copy()
    ref_test_data_dictionary = test_data_dictionary.copy()

    # Filter data subsets to keep only requested parts
    train_data_dictionary = filter_data_sub_sets(train_data_dictionary)
    test_data_dictionary = filter_data_sub_sets(test_data_dictionary)

    # Setup scaling instance
    scaler = StandardScaler()
    # Threshold for outlier filtering, all values further away than threshold * std_deviation from median are abandoned
    feature_to_remove = []
    # Count features but leave out ID in first column
    feature_count = 0
    outcome_descriptor = gl.original_outcome_strings[outcome_target_index]
    train_outcome_values = train_data_dictionary[outcome_descriptor]
    train_data_frame = pd.DataFrame(train_data_dictionary)
    test_data_frame = pd.DataFrame(test_data_dictionary)
    train_data_medians = train_data_frame.median()

    # If requested impute missing values in set by treating each outcome class separately
    if 'group' in impute:
        imputed_train_data = []
        # Separate classes by outcome
        for class_value in np.unique(train_outcome_values):
            subset_train = train_outcome_values == class_value
            class_subset = train_data_frame[subset_train]
            # In case of a complete class being nan, group based imputation is not possible,
            # it has to be switched to imputation of the whole set
            if class_subset.isnull().all().any():
                impute = 'std_median'
                break
            idf_train = pd.DataFrame(class_subset.fillna(class_subset.median()))
            imputed_train_data.append(idf_train)
        # Only continue with grouped imputation if not aborted earlier due to nan problems
        if 'group' in impute:
            train_data_dictionary = pd.concat(imputed_train_data)
            new_data_dictionary = {}
            for feature_name, feature_data in train_data_dictionary.items():
                tmp = feature_data.tolist()
                new_data_dictionary[feature_name] = tmp
            train_data_dictionary = new_data_dictionary
            # Impute test set with values from overall train set
            idf_test = pd.DataFrame(test_data_frame.fillna(train_data_medians))
            new_data_dictionary = {}
            for feature_name, feature_data in idf_test.items():
                tmp = feature_data.tolist()
                new_data_dictionary[feature_name] = tmp
            test_data_dictionary = new_data_dictionary

    # Do standardization and imputation if requested
    for (train_feature_name, train_feature_data), (test_feature_name, test_feature_data) in zip(
            train_data_dictionary.items(), test_data_dictionary.items()):
        # Preprocess only numerical data, remove all features that are strings or do not contain information
        if not any(isinstance(elem, str) for elem in train_feature_data) and len(set(train_feature_data)) > 1:
            # Remove first columns with outcomes
            if feature_count < gl.number_outcomes and feature_count != outcome_target_index:
                feature_to_remove.append(train_feature_name)
            else:

                # With grouped imputation impute test set with overall training mean to avoid data leakage
                if 'group' in impute:
                    test_data_dictionary[test_feature_name] = pd.DataFrame(test_feature_data).fillna(train_data_medians)
                else:
                    train_data_dictionary[train_feature_name] = np.reshape(imputer.fit_transform(
                        np.reshape(train_feature_data, (-1, 1))), (-1,))
                    test_data_dictionary[train_feature_name] = np.reshape(imputer.transform(
                        np.reshape(test_feature_data, (-1, 1))), (-1,))

                # If there are still nan values after imputation then there were no values in the first place
                if (any(np.isnan(train_data_dictionary[train_feature_name]))
                        or any(np.isnan(test_data_dictionary[test_feature_name]))):
                    feature_to_remove.append(train_feature_name)
                    feature_count += 1
                    # Skip standardization here to avoid errors
                    continue

                # Standardize only if requested and non-categorical
                if standardize and not np.all(np.isin(train_feature_data, [0, 1])):
                    train_data_dictionary[train_feature_name] = np.reshape(scaler.fit_transform(
                        np.reshape(train_data_dictionary[train_feature_name], (-1, 1))), (-1,))
                    if len(test_data_dictionary[test_feature_name]) == 0:
                        whatever = 0
                    test_data_dictionary[test_feature_name] = np.reshape(scaler.transform(
                        np.reshape(test_data_dictionary[test_feature_name], (-1, 1))), (-1,))
        else:
            feature_to_remove.append(train_feature_name)
        feature_count += 1

    # Remove all the features that are not suitable for classification
    for name in feature_to_remove:
        train_data_dictionary.pop(name)
        test_data_dictionary.pop(name)

    # Do SMOTE oversampling
    if oversample_rate > 0:
        smote = SMOTE(random_state=42)
        train_set = pd.DataFrame(train_data_dictionary)
        train_set.drop(columns=outcome_descriptor)
        train_set, train_outcome_values = smote.fit_resample(train_set, train_outcome_values)
        # Convert the resampled data back to a dictionary
        for feature_name in train_data_dictionary.keys():
            train_data_dictionary[feature_name] = train_set[feature_name]
        train_data_dictionary[outcome_descriptor] = train_outcome_values

    # Create boolean mask to filter points
    t = list(train_data_dictionary.keys())
    remove_points_mask = np.ones(len(train_data_dictionary[list(train_data_dictionary.keys())[0]]), dtype=bool)
    # Do z-score outlier filtering if requested
    if z_score_threshold > 0:
        for feature_name, feature_data in train_data_dictionary.items():
            z_score = filter_by_z_score(feature_data, np.median(feature_data), z_score_threshold)
            remove_points_mask &= z_score
        # Remove patients with invalid outlying points from the training feature set
        for feature_name, feature_data in train_data_dictionary.items():
            train_data_dictionary[feature_name] = np.array(feature_data)[remove_points_mask]
    # print('number of points removed', np.count_nonzero(np.invert(remove_points_mask)), 'with threshold', z_score_threshold)
    # print('Oversampled', multiplier, 'times, achieved class rate: ', current_class_rate)

    if 'PMP' in gl.feature_blocks_to_use:
        for (train_key, train_value), (test_key, test_value) \
                in zip(ref_train_data_dictionary.items(), ref_test_data_dictionary.items()):
            if 'DIF' in train_key:
                train_data_dictionary[train_key] = np.abs(
                    np.subtract(ref_train_data_dictionary[train_key[:-3] + 'PRE'],
                                ref_train_data_dictionary[train_key[:-4] + 'POST']))
                if test_key is not None:
                    test_data_dictionary[test_key] = np.abs(
                        np.subtract(ref_test_data_dictionary[test_key[:-3] + 'PRE'],
                                    ref_test_data_dictionary[test_key[:-4] + 'POST']))

    if len(train_data_dictionary[list(train_data_dictionary.keys())[0]]) == 0:
        return None, None
    return train_data_dictionary, test_data_dictionary


def filter_data_sub_sets(data_map):
    # According to the global settings remove data if it should be neglected in this run
    tmp_data_map = data_map.copy()
    before_data_reached, during_data_reached, after_data_reached = False, False, False

    for key, value in data_map.items():
        # Remove String data
        if not all(isinstance(x, (int, float)) for x in data_map[key]):
            tmp_data_map.pop(key)
            continue

        # If PMP is used add new field to map
        if 'PRE' in key and 'PMP' in gl.feature_blocks_to_use:
            tmp_data_map[key[:-3] + 'DIF'] = value

        # Remove SAP and HP markers
        if (('PRE' in key and 'PRE' not in gl.feature_blocks_to_use)
                or ('POST' in key and 'POST' not in gl.feature_blocks_to_use)):
            tmp_data_map.pop(key)
            continue

        # Blocks are sorted together, so always the first feature determines a new block
        if key == 'Age':
            before_data_reached = True
        elif key == 'CPB_yes/no':
            during_data_reached = True
            before_data_reached = False
        elif key == 'ICUStay(Days)':
            after_data_reached = True
            during_data_reached = False

        # Remove clinical data
        if ((before_data_reached and 'BEFORE' not in gl.feature_blocks_to_use)
                or (during_data_reached and 'DURING' not in gl.feature_blocks_to_use)
                or (after_data_reached and 'AFTER' not in gl.feature_blocks_to_use)):
            tmp_data_map.pop(key)

    return tmp_data_map
