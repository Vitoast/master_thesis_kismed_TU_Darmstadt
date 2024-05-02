import numpy as np
from sklearn.impute import SimpleImputer


# Preprocess data to make it usable in classification
# Involves: Imputation of missing values, standardization and outlier filtering
def preprocess_data(data_dictionary):
    # Create imputer instance to replace nan values in data
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    for feature_name, feature_data in data_dictionary.items():
        # Check if the feature data contains only numeric values
        if all(isinstance(x, (int, float)) for x in feature_data):
            data_dictionary[feature_name] = imputer.fit_transform(np.array(feature_data).reshape(-1, 1))
