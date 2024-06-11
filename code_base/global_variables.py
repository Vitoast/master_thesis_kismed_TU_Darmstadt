# String descriptors of used classifiers
classifiers = ['NaiveBayes', 'LogisticRegression', 'DecisionTree', 'SVM', 'RandomForest', 'XGBoost']
# String descriptors and count of adverse outcomes for easy use
outcome_descriptors = ["AKI3", "AKD1", "LCOS", "AF", "Any"]
number_outcomes = 5
# List used to save complete descriptors of adverse outcomes in data
original_outcome_strings = []

# Uniform colors for plots of used models
classifier_colors = ['red', 'orange', 'green', 'purple', 'black', 'blue']

# Adjust preprocessing of data for exploration and classification
#   Standardization can be turned on and off with True and False
#   For imputation a method can be chosen
#       'mean_std', 'median_std', 'mean_group' and 'median_group'
#   For z-score outlier filtering a threshold can be chosen
#       Reasonable values range from 4 to 16 times the standard deviation
#   For validation you can choose 'hold_out' for standard 80/20 distribution,
#       'k_fold' for k-set cross validation
#   For oversampling of imbalanced classes set a value.
#       Choose 0 to not oversample or 1 to enable
standardize, impute, filter_outliers_z_score, validation_method, oversample = True, 'mean_group', 13, 'k_fold', 0
standardization_parameters = [True, False]
imputation_parameters = ['median_std', 'mean_std', 'median_group', 'mean_group']
max_test_threshold = 16
min_test_threshold = 4
filter_outliers_parameters = range(min_test_threshold, max_test_threshold)
oversampling_parameters = [1, 0]
# When facing problems with oversampling see that threadpoolctl is properly installed

# These parameter combinations for preprocessing where computed to be the best for each outcome and model combination
preprocess_parameters = {
    'AKI3,NaiveBayes': [False, 'mean_std', 6, 0],
    'AKI3,LogisticRegression': [True, 'mean_std', 10, 1],
    'AKI3,DecisionTree': [False, 'mean_group', 11, 1],
    'AKI3,SVM': [True, 'mean_std', 4, 1],
    'AKI3,RandomForest': [False, 'mean_std', 11, 1],
    'AKI3,XGBoost': [False, 'mean_std', 11, 1],
    'AKD1,NaiveBayes': [True, 'mean_std', 4, 1],
    'AKD1,LogisticRegression': [True, 'median_std', 14, 0],
    'AKD1,DecisionTree': [False, 'mean_group', 14, 0],
    'AKD1,SVM': [True, 'mean_group', 5, 1],
    'AKD1,RandomForest': [True, 'median_std', 10, 0],
    'AKD1,XGBoost': [True, 'mean_std', 11, 0],
    'LCOS,NaiveBayes': [False, 'mean_std', 15, 1],
    'LCOS,LogisticRegression': [False, 'mean_std', 16, 0],
    'LCOS,DecisionTree': [False, 'median_std', 16, 0],
    'LCOS,SVM': [False, 'median_std', 10, 1],
    'LCOS,RandomForest': [False, 'mean_std', 16, 1],
    'LCOS,XGBoost': [False, 'median_std', 5, 1],
    'AF,NaiveBayes': [True, 'median_std', 10, 0],
    'AF,LogisticRegression': [True, 'mean_group', 5, 0],
    'AF,DecisionTree': [True, 'mean_group', 11, 0],
    'AF,SVM': [True, 'median_group', 7, 1],
    'AF,RandomForest': [False, 'mean_std', 14, 0],
    'AF,XGBoost': [True, 'median_std', 12, 1],
    'Any,NaiveBayes': [True, 'mean_std', 16, 0],
    'Any,LogisticRegression': [True, 'mean_group', 4, 0],
    'Any,DecisionTree': [False, 'median_std', 16, 1],
    'Any,SVM': [True, 'median_std', 14, 0],
    'Any,RandomForest': [False, 'median_std', 16, 1],
    'Any,XGBoost': [False, 'mean_std', 12, 0],
}
