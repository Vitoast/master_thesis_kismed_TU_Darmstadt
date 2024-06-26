import read_excel
import explorational_data_analysis as exp
import correlation_analysis as cor
import preprocess_data as pre
import classification as clf
import feature_evaluation as fe
import parameter_evaluation as pe
import os
import global_variables as gl


def main():
    # Path to Excel files with training and test data
    train_excel_file = 'data\\preprocessed_train_data.xlsx'
    test_excel_file = 'data\\preprocessed_test_data.xlsx'
    complete_excel_file = 'data\\preprocessed_complete_data.xlsx'
    source_dir_path = os.getcwd()
    train_source_path = os.path.join(source_dir_path, train_excel_file)
    test_source_path = os.path.join(source_dir_path, test_excel_file)
    complete_source_path = os.path.join(source_dir_path, complete_excel_file)
    result_path = os.path.join(source_dir_path, "results")

    # Read data from Excel file
    train_data_map = read_excel.read_excel_data(train_source_path)
    test_data_map = read_excel.read_excel_data(test_source_path)
    complete_data_map = read_excel.read_excel_data(complete_source_path)

    # Initialize the descriptors of the adverse outcomes
    for feature_name, feature_data in list(complete_data_map.items())[1:6]:
        gl.original_outcome_strings.append(feature_name)

    # Tune parameters
    # 1. Find best z-score outlier filter value
    parameter_evaluation_result_path = os.path.join(result_path, "parameter_evaluation_results")
    # pe.find_best_z_score_filter(complete_data_map, parameter_evaluation_result_path)
    # pe.find_best_imputation(complete_data_map, result_path)
    # pe.find_best_oversampling(complete_data_map, parameter_evaluation_result_path)
    # pe.bayesian_parameter_optimization_preprocessing(train_data_map, test_data_map, parameter_evaluation_result_path)
    pe.bayesian_parameter_optimization_models(train_data_map, test_data_map, parameter_evaluation_result_path)

    # Explore data and save results
    exploration_result_path = os.path.join(result_path, 'exploration_results')
    # exp.check_data_sets(train_data_map, test_data_map)
    # if standardize or impute:
    #     exp.explore_data(train_data_map, os.path.join(result_path, "standardized_exploration_results"))
    # else:
    #     exp.explore_data(train_data_map, os.path.join(result_path, "exploration_results"))
    umap_exploration_result_path = os.path.join(exploration_result_path, "umap_exploration_results")
    # exp.plot_umap(complete_data_map, umap_exploration_result_path)

    # Compute correlation metrics
    # cor.compute_marker_to_outcome_correlation(train_data_map, os.path.join(result_path, "correlation_results"))
    # cor.compute_marker_correlation_matrix(train_data_map, os.path.join(result_path, "correlation_results"))
    # cor.show_pairwise_marker_correlation(train_data_map, os.path.join(result_path, "correlation_results"))

    # Train classifier and predict
    classification_result_path = os.path.join(result_path, "classification_results")
    parameter_descriptor = [gl.standardize, gl.impute, gl.filter_outliers_z_score, gl.oversample]
    # Turn this on to retrieve model configuration
    print_model_details = False

    # if gl.validation_method == 'hold_out':
    #     for outcome in range(gl.number_outcomes):
    #         for model in gl.classifiers:
    #             print(gl.outcome_descriptors[outcome], model)
    #             print(clf.classify(train_data_map.copy(), test_data_map.copy(), outcome, classification_result_path,
    #                                parameter_descriptor, model, print_model_details, True))

    # if gl.validation_method == 'k_fold':
    #     for outcome in range(gl.number_outcomes):
    #         for model in gl.classifiers:
    #             print(gl.outcome_descriptors[outcome], model)
    #             print(clf.classify_k_fold(complete_data_map.copy(), outcome, classification_result_path,
    #                                 parameter_descriptor, model, print_model_details, True))

    # Do an ablation study to eliminate features from data set
    feature_evaluation_result_path = os.path.join(result_path, "feature_evaluation_results")
    # fe.read_csv_results(feature_evaluation_result_path, complete_data_map)
    # fe.plot_former_feature_ablation(feature_evaluation_result_path)
    # fe.perform_feature_ablation_study_vif(complete_data_map, feature_evaluation_result_path)
    # fe.continue_performance_ablation_after_vif(feature_evaluation_result_path, "", complete_data_map)
    # fe.perform_feature_ablation_study_performance(complete_data_map, feature_evaluation_result_path)

    # Do the ablation study for each subset of interest of the data set
    # for data_set in gl.possible_feature_combinations:
    #     current_result_path = os.path.join(feature_evaluation_result_path, data_set)
    #     gl.feature_blocks_to_use = data_set
    #     fe.perform_feature_ablation_study_vif(complete_data_map, current_result_path)
    #     fe.continue_performance_ablation_after_vif(current_result_path, "", complete_data_map)


if __name__ == "__main__":
    main()
