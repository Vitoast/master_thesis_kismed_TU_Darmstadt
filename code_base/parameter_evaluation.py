import classification as clf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import preprocess_data as pre
import global_variables as gl


# Train classifiers with differently outlier-filtered data and evaluate the performance
def find_best_z_score_filter(data_map, result_path):
    os.makedirs(result_path, exist_ok=True)
    accuracy_results, f1_score_results = [], []
    for i in range(gl.number_outcomes):
        accuracy_results.append([])
        f1_score_results.append([])
        for j in range(len(gl.classifiers)):
            accuracy_results[i].append([])
            f1_score_results[i].append([])
    # Try thresholds for z score filtering from 0 to max_test_threshold
    for model in gl.classifiers:
        for test_z in range(gl.min_test_threshold, gl.max_test_threshold):
            parameter_descriptor = [True, True, test_z]
            for outcome_value in range(gl.number_outcomes):
                accuracy, f1_scores = clf.classify_k_fold(data_map, outcome_value, result_path,
                                                          parameter_descriptor, model, False, True)
                accuracy_results[outcome_value][gl.classifiers.index(model)].append(accuracy)
                f1_score_results[outcome_value][gl.classifiers.index(model)].append(f1_scores)
    for outcome in gl.outcome_descriptors:
        # plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)
        colors = ['red', 'orange', 'green', 'purple', 'black']
        # Plot a scatter plot of the data including a regression line
        for model in range(len(gl.classifiers)):
            plt.scatter(x=list(range(gl.min_test_threshold, gl.max_test_threshold)),
                        y=f1_score_results[gl.outcome_descriptors.index(outcome)][model],
                        color=colors[model], label=gl.classifiers[model])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Z-score outlier filter test study ' + outcome)
        plt.xlabel('Z-score threshold')
        plt.ylabel('F1-score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, outcome + '_z_score_plot'))
        plt.show()
        plt.close()
