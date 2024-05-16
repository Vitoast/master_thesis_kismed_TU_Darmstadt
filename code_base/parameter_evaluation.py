import classification as clf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import preprocess_data as pre

max_test_threshold = 16
min_test_threshold = 4
number_outcomes = 5
# Values to print names of adverse outcomes
outcome_descriptors = ["AKI3", "AKD1", "LCOS", "AF", "Any"]


# Train classifiers with differently outlier-filtered data and evaluate the performance
def find_best_z_score_filter(train_data_map, test_data_map, classifiers, result_path):
    os.makedirs(result_path, exist_ok=True)
    accuracy_results, f1_score_results = [], []
    filtered_test_map = test_data_map.copy()
    pre.preprocess_data(filtered_test_map, standardize=True, impute=True,
                        z_score_threshold=0)
    for i in range(number_outcomes):
        accuracy_results.append([])
        f1_score_results.append([])
        for j in range(len(classifiers)):
            accuracy_results[i].append([])
            f1_score_results[i].append([])
    # Try thresholds for z score filtering from 0 to max_test_threshold
    for model in classifiers:
        for test_z in range(min_test_threshold, max_test_threshold):
            parameter_descriptor = [True, True, test_z]
            filtered_train_map = train_data_map.copy()
            pre.preprocess_data(filtered_train_map, standardize=True, impute=True,
                                z_score_threshold=test_z)
            accuracy, f1_scores = clf.classify(filtered_train_map, filtered_test_map, result_path, parameter_descriptor,
                                               model, False, True)
            for outcome_value in range(number_outcomes):
                accuracy_results[outcome_value][classifiers.index(model)].append(accuracy[outcome_value])
                f1_score_results[outcome_value][classifiers.index(model)].append(f1_scores[outcome_value])
    for outcome in outcome_descriptors:
        # plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        colors = ['red', 'orange', 'green', 'purple', 'black']
        # Plot a scatter plot of the data including a regression line
        for model in range(len(classifiers)):
            plt.scatter(x=list(range(min_test_threshold, max_test_threshold)),
                        y=f1_score_results[outcome_descriptors.index(outcome)][model],
                        color=colors[model], label=classifiers[model])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Z-score outlier filter test study ' + outcome)
        plt.xlabel('Z-score threshold')
        plt.ylabel('F1-score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(result_path + outcome + '_plot')
        plt.show()
        plt.close()
