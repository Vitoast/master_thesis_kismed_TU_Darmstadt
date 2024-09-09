import numpy as np
import os
import matplotlib.pyplot as plt
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import global_variables as gl
from code_base.prediction import preprocess_data as pre, classification as clf


# Initialize data structure to save results
def create_result_structure():
    accuracy_results, f1_score_results = [], []
    for i in range(gl.number_outcomes):
        accuracy_results.append([])
        f1_score_results.append([])
        for j in range(len(gl.classifiers)):
            accuracy_results[i].append([])
            f1_score_results[i].append([])
    return accuracy_results, f1_score_results


# save results as plot
def plot_parameter_evaluation(accuracy_results, f1_score_results, x_axis, result_path, title, xlabel, ylabel,
                              plot_name):
    for outcome in range(len(gl.outcome_descriptors)):
        ax = plt.subplot(111)

        # Plot a scatter plot of the data including a regression line
        for model in range(len(gl.classifiers)):
            plt.scatter(x=x_axis[outcome],  # fix this for imp and z !!!
                        y=f1_score_results[outcome][model],
                        color=gl.classifier_colors[model], label=gl.classifiers[model])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(title + gl.outcome_descriptors[outcome])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, gl.outcome_descriptors[outcome] + plot_name))
        plt.close()


# Try classification with and without SMOTE and plot performance difference
def find_best_oversampling(data_map, result_path):
    save_result_path = os.path.join(result_path, 'oversampling_evaluation')
    os.makedirs(save_result_path, exist_ok=True)
    accuracy_results, f1_score_results = create_result_structure()
    class_distributions = []

    # Test influence for each outcome
    for outcome_value in range(gl.number_outcomes):
        class_distributions.append([])

        # Try for oversampling and without
        for oversampling_rate in [0, 1]:
            # Try for each model separately
            for model in gl.classifiers:
                parameter_descriptor = [gl.standardize, gl.impute, gl.filter_outliers_z_score, oversampling_rate]
                accuracy, acc_var, f1_scores, f1_var, tmp0, tmp1 = clf.classify_k_fold(data_map, outcome_value,
                                                                                       save_result_path,
                                                                                       parameter_descriptor, model,
                                                                                       False, True)
                accuracy_results[outcome_value][gl.classifiers.index(model)].append(accuracy)
                f1_score_results[outcome_value][gl.classifiers.index(model)].append(f1_scores)

            # Calculate the original and oversampled class distributions in the data
            positive_class_count = data_map[gl.original_outcome_strings[outcome_value]].count(1)
            total_class_count = len(data_map[gl.original_outcome_strings[outcome_value]])
            class_ratio = positive_class_count / total_class_count
            multiplier = int(np.floor(oversampling_rate / class_ratio)) + 1
            class_ratio = positive_class_count * multiplier / total_class_count
            class_distributions[outcome_value].append(class_ratio)

    # Plot resulting f1-scores
    plot_parameter_evaluation(accuracy_results, f1_score_results, class_distributions,
                              save_result_path, 'Oversampling test study ',
                              'class distribution', 'F1-score', '_oversampling_plot')


# Test every imputation strategy according to classification performance
def find_best_imputation(data_map, result_path):
    save_result_path = os.path.join(result_path, 'imputation_evaluation')
    os.makedirs(save_result_path, exist_ok=True)
    accuracy_results, f1_score_results = create_result_structure()

    # Test each parameter
    for parameter in gl.imputation_parameters:
        for model in gl.classifiers:
            for outcome_value in range(gl.number_outcomes):
                parameter_descriptor = [gl.standardize, parameter, gl.filter_outliers_z_score]
                accuracy, acc_var, f1_scores, f1_var, tmp0, tmp1 = clf.classify_k_fold(data_map, outcome_value,
                                                                                       result_path,
                                                                                       parameter_descriptor, model,
                                                                                       False, True)
                accuracy_results[outcome_value][gl.classifiers.index(model)].append(accuracy)
                f1_score_results[outcome_value][gl.classifiers.index(model)].append(f1_scores)

    # Plot evaluation results
    plot_parameter_evaluation(accuracy_results, f1_score_results, list(range(len(gl.imputation_parameters))),
                              save_result_path, 'Imputation methods test study ',
                              'strategy', 'F1-score', '_imputation_methods_plot')


# Train classifiers with differently outlier-filtered data and evaluate the performance
def find_best_z_score_filter(data_map, result_path):
    save_result_path = os.path.join(result_path, 'z_score_evaluation')
    os.makedirs(save_result_path, exist_ok=True)
    accuracy_results, f1_score_results = create_result_structure()

    # Try thresholds for z score filtering from 0 to max_test_threshold
    for model in gl.classifiers:
        for test_z in range(gl.min_test_threshold, gl.max_test_threshold):
            parameter_descriptor = [True, gl.imputation_parameters[2], test_z]
            for outcome_value in range(gl.number_outcomes):
                accuracy, acc_var, f1_scores, f1_var, tmp0, tmp1 = clf.classify_k_fold(data_map, outcome_value,
                                                                                       result_path,
                                                                                       parameter_descriptor, model,
                                                                                       False, True)
                accuracy_results[outcome_value][gl.classifiers.index(model)].append(accuracy)
                f1_score_results[outcome_value][gl.classifiers.index(model)].append(f1_scores)

    # Plot evaluation of different z-scores
    plot_parameter_evaluation(accuracy_results, f1_score_results,
                              list(range(gl.min_test_threshold, gl.max_test_threshold)),
                              save_result_path, 'Z-score outlier filter test study ',
                              'Z-score threshold', 'F1-score', '_z_score_plot')


# Perform a bayesian optimization on the hyperparameter of the data preprocessing
def bayesian_parameter_optimization_preprocessing(train_data_map, test_data_map, result_path):
    stats_directory_path = os.path.join(result_path, 'bayesian_parameter_optimization')
    os.makedirs(stats_directory_path, exist_ok=True)
    stats_file_path = os.path.join(stats_directory_path, 'preprocessing_parameter_optimization.txt')
    # Define the hyperparameter space
    param_space = [
        Categorical([True, False], name='standardize'),
        Categorical(['mean_std', 'median_std', 'mean_group', 'median_group'], name='imputation_strategy'),
        Integer(gl.min_test_threshold, gl.max_test_threshold, name='z_score'),
        Integer(0, 1, name='oversampling'),
    ]
    current_outcome = 0
    current_model = ''
    parameter_descriptor = [gl.standardize, gl.impute, gl.filter_outliers_z_score, gl.oversample]

    # Define the objective function to wrap classification process
    @use_named_args(param_space)
    def objective(**params):
        # Apply parameters
        gl.standardize = params['standardize']
        gl.impute = params['imputation_strategy']
        gl.filter_outliers_z_score = params['z_score']
        gl.oversample = params['oversampling']
        # Classify and return f1 score
        tmp = \
            clf.classify(train_data_map.copy(), test_data_map.copy(), current_outcome, result_path,
                         parameter_descriptor,
                         current_model, False, False)[1]
        return - tmp[0]

    with open(stats_file_path, 'w') as stats_file:

        # Do optimization separately for each outcome and model
        for outcome in range(0, gl.number_outcomes):
            current_outcome = outcome
            for model in gl.classifiers:
                current_model = model
                # Minimize objective function
                result = gp_minimize(objective, dimensions=param_space, n_calls=200, random_state=42)
                best_parameters = result.x
                best_score = - result.fun
                stats_file.write(f'{gl.outcome_descriptors[outcome]},{model},{best_parameters},{best_score},\n')

    # Plot results
    plot_bayesian_optimization_results(stats_file_path, result_path)


# Take the output file of bayesian_parameter_optimization and plot it
def plot_bayesian_optimization_results(result_file_path, result_directory):
    os.makedirs(result_directory, exist_ok=True)
    f1_score_results = create_result_structure()[0]

    # Read the file line by line
    with open(result_file_path, 'r') as file:
        for line in file:
            # Strip the line of any leading/trailing whitespace and split by comma
            parts = line.strip().split(',')

            # Extract the relevant parts
            outcome_descriptor = parts[0]
            model_descriptor = parts[1]
            f1_score_result = float(parts[-2])  # Convert the last value to float

            f1_score_results[gl.outcome_descriptors.index(outcome_descriptor)][
                gl.classifiers.index(model_descriptor)].append(f1_score_result)

    for outcome in range(len(gl.outcome_descriptors)):
        ax = plt.subplot(111)
        plot_name = 'preprocessing_optimization_results_for_' + gl.outcome_descriptors[outcome]

        # Plot a scatter plot of the data including a regression line
        for model in range(len(gl.classifiers)):
            plt.scatter(x=list(range(len(f1_score_results[outcome][model]))),
                        y=f1_score_results[outcome][model],
                        color=gl.classifier_colors[model], label=gl.classifiers[model])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(
            'Bayesian Optimization Results\nof Preprocessing Hyperparameters for ' + gl.outcome_descriptors[outcome])
        plt.xlabel('Model')
        plt.ylabel('Achieved F1-score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_directory, plot_name))
        plt.close()


# Perform a bayesian optimization on the hyperparameter of the used classifier models
def bayesian_parameter_optimization_models(train_data_map, test_data_map, result_path):
    stats_directory_path = os.path.join(result_path, 'bayesian_parameter_optimization')
    os.makedirs(stats_directory_path, exist_ok=True)
    stats_file_path = os.path.join(stats_directory_path, 'model_parameter_optimization.txt')

    with open(stats_file_path, 'w') as stats_file:

        # Optimize for each model
        for classification_descriptor in gl.classifiers:
            # FOR DEBUGGING, DELETE LATER -->
            if classification_descriptor == 'NaiveBayes': continue
            # if classification_descriptor == 'LogisticRegression': continue
            # if classification_descriptor == 'DecisionTree': continue
            # if classification_descriptor == 'SVM': continue
            # if classification_descriptor == 'RandomForest': continue
            # if classification_descriptor == 'XGBoost': continue

            # Define the hyperparameter space depending on the current model
            if classification_descriptor == 'NaiveBayes':
                param_space = {
                    'var_smoothing': np.logspace(0, -9, num=10),
                }
                classifier = GaussianNB()

            elif classification_descriptor == 'LogisticRegression':
                param_space = {
                    # 'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    # 'tol': [1e-4, 1e-3, 1e-2, 1e-1],
                    'C': [0.01, 0.1, 1, 10, 100],
                    'class_weight': [None, 'balanced'],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    # 'l1_ratio': Real(0, 1, prior='uniform')  # Only used by 'elasticnet' penalty
                }

                # Custom function to filter valid hyperparameter sets
                # def is_valid(params):
                #     if params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
                #         return False
                #     if params['penalty'] == 'none' and params['solver'] == 'liblinear':
                #         return False
                #     if params['penalty'] in ['l1', 'elasticnet'] and params['solver'] not in ['liblinear', 'saga']:
                #         return False
                #     if params['solver'] == 'liblinear' and params['penalty'] not in ['none', 'l1', 'l2']:
                #         return False
                #     return True

                classifier = LogisticRegression()

            elif classification_descriptor == 'DecisionTree':
                param_space = {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    # 'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1],
                    'max_features': [None, 'sqrt', 'log2'],
                    # 'max_leaf_nodes': [None, 10, 20, 30],
                    # 'min_impurity_decrease': [0.0, 0.01, 0.1],
                    'class_weight': [None, 'balanced'],
                    # 'ccp_alpha': [0.0, 0.01, 0.1],
                }
                classifier = DecisionTreeClassifier()

            elif classification_descriptor == 'SVM':
                param_space = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4, 5],
                    'coef0': [0.0, 0.1, 0.5, 1.0],
                    'shrinking': [True, False],
                    'probability': [True, False],
                    'tol': [1e-4, 1e-3, 1e-2, 1e-1],
                    'class_weight': [None, 'balanced'],
                }
                classifier = SVC()

            elif classification_descriptor == 'RandomForest':
                param_space = {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    # 'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1],
                    'max_features': [None, 'sqrt', 'log2'],
                    # 'max_leaf_nodes': [None, 10, 20, 30],
                    # 'min_impurity_decrease': [0.0, 0.01, 0.1],
                    # 'bootstrap': [True, False],
                    # 'oob_score': [True, False],
                    'class_weight': [None, 'balanced'],
                    # 'ccp_alpha': [0.0, 0.01, 0.1],
                    # 'max_samples': [None, 0.5, 0.75, 1.0],
                }
                classifier = RandomForestClassifier()

            elif classification_descriptor == 'XGBoost':
                param_space = {
                    'n_estimators': [50, 100, 300],
                    'max_depth': [3, 6, 9, 12],
                    # 'max_leaves': [0, 31, 63, 127],
                    # 'max_bin': [256, 512, 1024],
                    # 'grow_policy': ['depthwise', 'lossguide'],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    # 'tree_method': ['auto', 'exact', 'approx', 'hist'],
                    'gamma': [0, 0.1, 0.2, 0.5, 1.0],
                    'min_child_weight': [1, 5, 10, 20],
                    # 'max_delta_step': [0, 0.1, 0.5, 1.0],
                    'subsample': [0.5, 0.7, 0.8, 0.9, 1.0],
                }
                classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

            for outcome in range(0, gl.number_outcomes):

                # DEBUGGING !!!!
                if outcome != gl.outcome_descriptors.index('AKD1'):
                    continue

                # Find matching configuration in precomputed data
                current_configurations = next((value for key, value in gl.preprocess_parameters.items()
                                               if gl.outcome_descriptors[outcome] in key
                                               and classification_descriptor in key), None)
                # Define preprocessing parameters based on former optimization
                current_standardize = current_configurations[0]
                current_impute = current_configurations[1]
                current_z_score_threshold = current_configurations[2]
                current_oversample_rate = current_configurations[3]

                # Preprocess data accordingly
                tmp_train_data_map, tmp_test_data_map = pre.preprocess_data(train_data_map.copy(), test_data_map.copy(),
                                                                            outcome,
                                                                            standardize=current_standardize,
                                                                            impute=current_impute,
                                                                            z_score_threshold=current_z_score_threshold,
                                                                            oversample_rate=current_oversample_rate)
                x_train, x_test, y_train, y_test = clf.split_maps(tmp_train_data_map, tmp_test_data_map)
                x_train = np.reshape(x_train, (len(x_train[0]), len(x_train)))
                x_test = np.reshape(x_test, (len(x_test[0]), len(x_test)))
                y_train = np.array(y_train).flatten()
                y_test = np.array(y_test).flatten()

                # Define the scoring metric
                f1_scorer = make_scorer(f1_score)

                # Custom optimizer to ensure valid parameter combinations
                # class CustomBayesSearchCV(BayesSearchCV):
                #     def _fit(self, X, y, groups=None, **fit_params):
                #         # Filter the search space based on the validity function
                #         valid_search_spaces = []
                #         for params in self.search_spaces:
                #             if is_valid(params):
                #                 valid_search_spaces.append(params)
                #         self.search_spaces = valid_search_spaces
                #
                #         super()._fit(X, y, groups, **fit_params)

                # Initialize bayesian optimization
                opt = BayesSearchCV(
                    estimator=classifier,
                    search_spaces=param_space,
                    cv=3,
                    n_iter=100,
                    random_state=42,
                    scoring=f1_scorer,
                    verbose=1,
                )

                # Perform the optimization with the preprocessed data
                opt.fit(x_train, y_train)

                # Best parameters and best score
                print(gl.outcome_descriptors[outcome] + " best parameters found: ", opt.best_params_)
                # Evaluate the best model on the test set
                best_model = opt.best_estimator_
                test_score = best_model.score(x_test, y_test)
                print("Test set accuracy score: ", test_score)
                # Evaluate the best model on the test set
                y_pred = best_model.predict(x_test)
                test_f1_score = f1_score(y_test, y_pred)
                print("Test set F1-score: ", test_f1_score)

                stats_file.write(f'{gl.outcome_descriptors[outcome]},{classification_descriptor},{opt.best_params_},'
                                 f'{test_score},{test_f1_score},\n')

        # Plot results
        plot_bayesian_optimization_results(stats_file_path, stats_directory_path)
