import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# Returns an array of booleans that tell if a point is outside the threshold * std_deviation from the median
def filter_by_z_score(data, center, threshold):
    z_scores = np.abs((data - center) / np.std(data))
    return z_scores < threshold


# Preprocess data to make it usable in classification
# Involves: Imputation of missing values, standardization and (outlier filtering)
def preprocess_data(data_dictionary, standardize, impute, z_score_threshold):
    # Create imputation instance to replace nan values in data
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = StandardScaler()
    # Threshold for outlier filtering, all values further away than threshold * std_deviation from median are abandoned
    feature_to_remove = []
    remove_points_mask = np.ones(len(data_dictionary['ID']), dtype=bool)
    for feature_name, feature_data in data_dictionary.items():
        if not any(isinstance(elem, str) for elem in feature_data):
            if impute:
                data_dictionary[feature_name] = np.reshape(imputer.fit_transform(np.reshape(feature_data, (-1, 1))), (-1,))
            # Standardize and filter only if requested and non-categorical
            if not np.all(np.isin(feature_data, [0, 1])):
                if standardize:
                    data_dictionary[feature_name] = np.reshape(scaler.fit_transform(np.reshape(data_dictionary[feature_name], (-1, 1))), (-1,))
                if z_score_threshold > 0:
                    remove_points_mask &= filter_by_z_score(data_dictionary[feature_name], np.median(data_dictionary[feature_name]), z_score_threshold)
        else:
            feature_to_remove.append(feature_name)
    # Remove all the features that are not suitable for classification
    for name in feature_to_remove:
        data_dictionary.pop(name)
    # Remove patients with invalid outlying points from the feature set
    if z_score_threshold > 0:
        for feature_name, feature_data in data_dictionary.items():
            data_dictionary[feature_name] = feature_data[remove_points_mask]
        # print('number of points removed', np.count_nonzero(np.invert(remove_points_mask)), 'with threshold', z_score_threshold)
