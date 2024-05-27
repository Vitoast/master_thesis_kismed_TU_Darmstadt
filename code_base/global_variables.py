# String descriptors of the used classifiers
classifiers = ['NaiveBayes', 'LogisticRegression', 'DecisionTree', 'SVM', 'RandomForest', 'XGBoost']
# String descriptors and count of adverse outcomes
outcome_descriptors = ["AKI3", "AKD1", "LCOS", "AF", "Any"]
number_outcomes = 5

# Preprocess data for exploration and classification
#   Standardization can be turned on and off with True and False
#   For imputation a method can be chosen
#       'mean_std', 'median_std', 'mean_group' and 'median_group' can be chosen
#   For outlier filtering a threshold can be chosen
#       Reasonable values range from 4 to 16 times the standard deviation
#   For validation you can choose 'hold_out' for standard 80/20 distribution,
#       'k_fold' for k-set cross validation (and 'leave_one_out' for n-point cross validation)
#   For oversampling of imbalanced classes set a value.
#       Choose 0 to not oversample or the desired class distribution
standardize, impute, filter_outliers_z_score, validation_method, oversample = True, 'median_group', 7, 'k_fold', 1
feature_ablation_strategy = 'performance' # 'vif'
imputation_parameters = ['median_std', 'mean_std', 'median_group', 'mean_group']
#   Maximum and minimum constants for z-score outlier filtering
max_test_threshold = 16
min_test_threshold = 4
# Variable to save used class distributions
original_outcome_strings = []
# Uniform colors for plots
classifier_colors = ['red', 'orange', 'green', 'purple', 'black', 'blue']

