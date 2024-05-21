import classification as clf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import preprocess_data as pre
import global_variables as gl


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
def plot_parameter_evaluation(accuracy_results, f1_score_results, x_axis, result_path, title, xlabel, ylabel, plot_name):
    for outcome in range(len(gl.outcome_descriptors)):
        ax = plt.subplot(111)
        colors = ['red', 'orange', 'green', 'purple', 'black']
        # Plot a scatter plot of the data including a regression line
        for model in range(len(gl.classifiers)):
            plt.scatter(x=x_axis[outcome], # fix this for imp and z !!!
                        y=f1_score_results[outcome][model],
                        color=colors[model], label=gl.classifiers[model])
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


def find_best_oversampling(data_map, result_path):
    save_result_path = os.path.join(result_path, 'oversampling_evaluation')
    os.makedirs(save_result_path, exist_ok=True)
    accuracy_results, f1_score_results = create_result_structure()
    class_distributions = []
    for outcome_value in range(gl.number_outcomes):
        class_distributions.append([])
        for oversampling_rate in [0, 0.4]:
            for model in gl.classifiers:
                parameter_descriptor = [gl.standardize, gl.impute, gl.filter_outliers_z_score, oversampling_rate]
                accuracy, f1_scores = clf.classify_k_fold(data_map, outcome_value, save_result_path,
                                                          parameter_descriptor, model, False, True)
                accuracy_results[outcome_value][gl.classifiers.index(model)].append(accuracy)
                f1_score_results[outcome_value][gl.classifiers.index(model)].append(f1_scores)
            # Calculate the original and oversampled class distributions in the data
            positive_class_count = data_map[gl.original_outcome_strings[outcome_value]].count(1)
            total_class_count = len(data_map[gl.original_outcome_strings[outcome_value]])
            class_ratio = positive_class_count / total_class_count
            multiplier = int(np.floor(oversampling_rate / class_ratio)) + 1
            class_ratio = positive_class_count * multiplier / total_class_count
            class_distributions[outcome_value].append(class_ratio)
    plot_parameter_evaluation(accuracy_results, f1_score_results, class_distributions,
                              save_result_path, 'Oversampling test study ',
                              'class distribution', 'F1-score', '_oversampling_plot')


# Test every imputation strategy
def find_best_imputation(data_map, result_path):
    save_result_path = os.path.join(result_path, 'imputation_evaluation')
    os.makedirs(save_result_path, exist_ok=True)
    accuracy_results, f1_score_results = create_result_structure()
    for parameter in gl.imputation_parameters:
        for model in gl.classifiers:
            for outcome_value in range(gl.number_outcomes):
                parameter_descriptor = [gl.standardize, parameter, gl.filter_outliers_z_score]
                accuracy, f1_scores = clf.classify_k_fold(data_map, outcome_value, result_path,
                                                          parameter_descriptor, model, False, True)
                accuracy_results[outcome_value][gl.classifiers.index(model)].append(accuracy)
                f1_score_results[outcome_value][gl.classifiers.index(model)].append(f1_scores)
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
                accuracy, f1_scores = clf.classify_k_fold(data_map, outcome_value, result_path,
                                                          parameter_descriptor, model, False, True)
                accuracy_results[outcome_value][gl.classifiers.index(model)].append(accuracy)
                f1_score_results[outcome_value][gl.classifiers.index(model)].append(f1_scores)
    plot_parameter_evaluation(accuracy_results, f1_score_results,
                              list(range(gl.min_test_threshold, gl.max_test_threshold)),
                              save_result_path, 'Z-score outlier filter test study ',
                              'Z-score threshold', 'F1-score', '_z_score_plot')
