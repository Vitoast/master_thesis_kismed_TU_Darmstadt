# String descriptors of used classifiers
# Only the two best performing are used in the final version, uncomment the rest to use them
classifiers = ['NaiveBayes', 'SVM']#, 'LogisticRegression', 'DecisionTree',  'RandomForest', 'XGBoost']
# String descriptors and count of adverse outcomes for easy use
outcome_descriptors = ["AKD", "AKI1", "LCOS", "AF"]#, "Any"]
number_outcomes = len(outcome_descriptors)
# List used to save complete descriptors of adverse outcomes in data
original_outcome_strings = []

# Descriptors used to divide the data into subsets
# Add the specific string to the general descriptor to use that block of data
#   'PRE' for computed clinical markers before anesthesia
#   'POST' for computed clinical markers after anesthesia
#   'PMP' for absolute PRE minus POST values, only usable standalone
#   'BEFORE' for demographic data and data collected before surgery
#   'DURING' for data about the surgery process
#   'AFTER' for data collected after surgery
feature_blocks_to_use = 'PRE_POST_BEFORE_DURING_AFTER'
# Subsets that make sense to split from the data set
possible_feature_combinations = ['PRE',
                                 'POST',
                                 'PRE_POST',
                                 #'PMP',
                                 'BEFORE_DURING',
                                 'PRE_POST_BEFORE_DURING']
                                 #'PRE_POST_BEFORE_DURING_AFTER']
# These are the combinations of subsets that should be compared
combinations_to_test = [['PRE', 'PRE_POST'],
                        ['POST', 'PRE_POST'],
                        ['PRE', 'PRE_PMP'],
                        ['POST', 'POST_PMP'],
                        ['PRE_POST', 'PRE_POST_BEFORE_DURING'],
                        ['BEFORE_DURING', 'PRE_POST_BEFORE_DURING']]

# Uniform colors for plots of used models
classifier_colors = ['red', 'purple', 'orange', 'green', 'black', 'blue']

# Adjust preprocessing of data for exploration and classification
#   Standardization can be turned on and off with True and False
#   For imputation a method can be chosen
#       'mean_std', 'median_std', 'mean_group' and 'median_group'
#   For z-score outlier filtering a threshold can be chosen, pick 0 to turn off
#       Reasonable values range from 4 to 16 times the standard deviation
#   For validation you can choose 'hold_out' for standard 80/20 distribution,
#       'k_fold' for k-set cross validation
#   For oversampling of imbalanced classes set a value.
#       Choose 0 to not oversample or 1 to enable
standardize, impute, filter_outliers_z_score, validation_method, oversample = False, 'mean_group', 0, 'k_fold', 1
standardization_parameters = [True, False]
imputation_parameters = ['median_std', 'mean_std', 'median_group', 'mean_group']
max_test_threshold = 16
min_test_threshold = 4
filter_outliers_parameters = range(min_test_threshold, max_test_threshold)
# HINT: When facing problems with oversampling see that package threadpoolctl is properly installed
oversampling_parameters = [1, 0]

# Define how many splits in CV should be used
k_fold_split = 5

# Parameter combinations for preprocessing to be the best for each outcome and model combination (precomputed)
preprocess_parameters = {
    'AKD,NaiveBayes': [True, 'mean_std', 6, 0],
    'AKD,LogisticRegression': [True, 'mean_std', 10, 1],
    'AKD,DecisionTree': [False, 'mean_group', 11, 1],
    'AKD,SVM': [True, 'mean_std', 10, 1],
    'AKD,RandomForest': [False, 'mean_std', 11, 1],
    'AKD,XGBoost': [False, 'mean_std', 11, 1],
    'AKI1,NaiveBayes': [True, 'mean_std', 4, 1],
    'AKI1,LogisticRegression': [True, 'median_std', 14, 0],
    'AKI1,DecisionTree': [False, 'mean_group', 14, 0],
    'AKI1,SVM': [True, 'mean_group', 13, 1],
    'AKI1,RandomForest': [True, 'median_std', 10, 0],
    'AKI1,XGBoost': [True, 'mean_std', 11, 0],
    'LCOS,NaiveBayes': [True, 'mean_std', 15, 1],
    'LCOS,LogisticRegression': [False, 'mean_std', 16, 0],
    'LCOS,DecisionTree': [False, 'median_std', 16, 0],
    'LCOS,SVM': [True, 'median_group', 11, 1],
    'LCOS,RandomForest': [False, 'mean_std', 16, 1],
    'LCOS,XGBoost': [False, 'median_std', 5, 1],
    'AF,NaiveBayes': [True, 'median_std', 10, 0],
    'AF,LogisticRegression': [True, 'mean_group', 5, 0],
    'AF,DecisionTree': [True, 'mean_group', 11, 0],
    'AF,SVM': [True, 'median_group', 13, 0],
    'AF,RandomForest': [False, 'mean_std', 14, 0],
    'AF,XGBoost': [True, 'median_std', 12, 1],
    'Any,NaiveBayes': [True, 'mean_std', 16, 0],
    'Any,LogisticRegression': [True, 'mean_group', 4, 0],
    'Any,DecisionTree': [False, 'median_std', 16, 1],
    'Any,SVM': [True, 'median_group', 14, 0],
    'Any,RandomForest': [False, 'median_std', 16, 1],
    'Any,XGBoost': [False, 'mean_std', 12, 0],
}

# These parameters are to set up the prediction models in the best way corresponding to the outcomes (precomputed)
model_parameters = {
    'AKD,NaiveBayes': {'var_smoothing': 1.0},
    'AKI1,NaiveBayes': {'var_smoothing': 1.0},
    'LCOS,NaiveBayes': {'var_smoothing': 1.0},
    'AF,NaiveBayes': {'var_smoothing': 0.0001},
    'Any,NaiveBayes': {'var_smoothing': 1.0},
    'AKD,SVM': {'C': 0.01, 'class_weight': None, 'coef0': 0.0, 'degree': 3, 'kernel': 'rbf',
                'probability': True, 'shrinking': True, 'tol': 0.0001},
    'AKI1,SVM': {'C': 0.01, 'class_weight': None, 'coef0': 0.0, 'degree': 2, 'kernel': 'rbf',
                 'probability': True, 'shrinking': False, 'tol': 0.0001},
    'LCOS,SVM': {'C': 1, 'class_weight': 'balanced', 'coef0': 0.1, 'degree': 5, 'kernel': 'sigmoid',
                 'probability': True, 'shrinking': False, 'tol': 0.0001},
    'AF,SVM': {'C': 1, 'class_weight': None, 'coef0': 0.0, 'degree': 2, 'kernel': 'poly',
               'probability': True, 'shrinking': True, 'tol': 0.001},
    'Any,SVM': {'C': 0.1, 'class_weight': None, 'coef0': 0.1, 'degree': 5, 'kernel': 'sigmoid',
                'probability': True, 'shrinking': False, 'tol': 0.01},
}

# Threshold for variance inflation based feature ablation, usually 5 (below vif is insignificant)
vif_threshold = 5

# If this is set SHAP is used with each classification to explain the result
explain_prediction = False

# This is used to tell the classification if a feature study or a statistical test is performed
# In feature study a not learning model has to be punished more
scale_bad_performance_results = True
