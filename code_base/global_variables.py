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



