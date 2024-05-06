import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def filter_by_z_score(data, center, threshold):
    z_scores = np.abs((data - center) / np.std(data))
    return z_scores < threshold


# Preprocess data to make it usable in classification
# Involves: Imputation of missing values, standardization and (outlier filtering)
def preprocess_data(data_dictionary, preprocessed_train_data, standardize, impute, filter_outliers):
    # Create imputation instance to replace nan values in data
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = StandardScaler()
    filter_threshold = 3
    feature_to_remove, points_to_remove = [], []
    remove_points_mask = np.ones(len(data_dictionary['ID']), dtype=bool)
    for feature_name, feature_data in data_dictionary.items():
        if not any(isinstance(elem, str) for elem in feature_data):
            if impute:
                data_dictionary[feature_name] = np.reshape(imputer.fit_transform(np.reshape(feature_data, (-1, 1))), (-1,))
            else:
                # Standardize only if requested and non-categorical
                if not np.all(np.isin(feature_data, [0, 1])):
                    if standardize:
                        data_dictionary[feature_name] = np.reshape(scaler.fit_transform(np.reshape(data_dictionary[feature_name], (-1, 1))), (-1,))
                    if filter_outliers:
                        remove_points_mask &= filter_by_z_score(data_dictionary[feature_name], np.median(data_dictionary[feature_name]), filter_threshold)
            # print(data_dictionary[feature_name])
        else:
            feature_to_remove.append(feature_name)

    # Remove all the features that are not suitable
    for name in feature_to_remove:
        data_dictionary.pop(name)
    # Remove patients with invalid outlying points from the features
    if filter_outliers:
        for feature_name, feature_data in data_dictionary.items():
            data_dictionary[feature_name] = feature_data[remove_points_mask]
        print('number of points removed', np.count_nonzero(np.invert(remove_points_mask)), 'with threshold', filter_threshold)
