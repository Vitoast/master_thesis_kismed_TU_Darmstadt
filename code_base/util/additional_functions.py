import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import csv


# Example plot of two Gaussian classes and their decision boundary
def plot_two_gaussians(result_path):
    os.makedirs(result_path, exist_ok=True)
    mean1, std1 = 3, 1
    mean2, std2 = 6, 2
    x = np.linspace(0, 10, 1000)
    y1 = norm.pdf(x, mean1, std1)
    y2 = norm.pdf(x, mean2, std2)
    plt.clf()
    plt.plot(x, y1, label='Gaussian data class 1', color='red')
    plt.plot(x, y2, label='Gaussian data class 2', color='blue')
    decision_boundary = (mean1 + mean2) / 2 - 0.05
    plt.axvline(decision_boundary, color='grey', linestyle='--', label='Decision Boundary')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Two Gaussian Distributions with Decision Boundary')
    plt.savefig(os.path.join(result_path, "gaussian_db_example.pdf"), format='pdf')


# Example plot to explain linear regression, includes two scattered classes and boundary
def plot_linear_separation(result_path):
    os.makedirs(result_path, exist_ok=True)
    np.random.seed(0)
    class1 = np.random.randn(50, 2) + np.array([2, 2])
    class2 = np.random.randn(50, 2) + np.array([-2, -2])
    plt.clf()
    plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
    plt.scatter(class2[:, 0], class2[:, 1], color='blue', label='Class 2')
    x_values = np.linspace(-5, 5, 100)
    y_values = - x_values  # Change this to your specific decision boundary equation if needed
    plt.plot(x_values, y_values, color='grey', linestyle='--', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linearly Separable Classes with Possible Decision Boundary')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_path, "linear_reg_example.pdf"), format='pdf')


# Create a file containing latex tables for read in CSV files
def generate_latex_table(csv_files, output_file):
    with open(output_file, 'w') as output:
        for table_num, csv_file in enumerate(csv_files, start=1):

            table_str = []

            # Define the header of the LaTeX table
            table_str.append(r"\begin{table}[H]")
            table_str.append(r"\adjustbox{max width=\textwidth}{")
            table_str.append(r"\centering")
            table_str.append(r"\begin{tabular}{|l|c|c|c|c|c|}")
            table_str.append(r"\hline")

            # Add the table header
            table_str.append(
                r"\textbf{Feature} & \textbf{Feature Count} & \textbf{Accuracy}"
                r" & \textbf{Accuracy Variance} & \textbf{$F1$-Score} & \textbf{$F1$-Score Variance} \\ \hline\hline")

            # Open the CSV file and read the contents
            with open(csv_file, newline='') as f:
                reader = csv.reader(f)

                # Loop over the rows in the CSV and add them to the LaTeX table
                for i, row in enumerate(reader):
                    if i >= 20:  # Limit to 20 rows
                        break
                    feature = row[0].replace('_', r'\_').replace('^2', '$^2$')
                    accuracy = row[1]
                    accuracy_variance = row[2]
                    f1_score = row[3]
                    f1_score_variance = row[4]
                    table_str.append(f"{feature} & {i + 1} & {accuracy} & {accuracy_variance} "
                                     f"& {f1_score} & {f1_score_variance} \\\\ \hline")

            # Close the table environment
            table_str.append(r"\end{tabular}")
            table_str.append(r"}")
            caption_str = r"\caption{Table of features added in each accumulation iteration based on "
            label_str = r"\label{app_table_feature_acc_"

            if 'PRE_POST_BEFORE_DURING' in csv_file:
                caption_str += "complete set "
                label_str += "complete_"
            elif 'PRE_POST_' in csv_file:
                caption_str += "cardiovascular marker set "
                label_str += 'pre_post_'
            else:
                caption_str += "demographical and clinical data set "
                label_str += 'bef_dur_'

            if 'NaiveBayes' in csv_file:
                caption_str += "with NaiveBayes "
                label_str += "nb_"
            else:
                caption_str += "with SVM "
                label_str += 'svm_'

            if 'AKD' in csv_file:
                caption_str += "for AKD "
                label_str += "akd"
            elif 'AKI1' in csv_file:
                caption_str += "for AKI1 "
                label_str += 'aki1'
            elif 'LCOS' in csv_file:
                caption_str += "for LCOS "
                label_str += 'lcos'
            else:
                caption_str += "for AF "
                label_str += 'af'

            caption_str += "including performance measures.}"
            label_str += r"}"
            table_str.append(caption_str)
            table_str.append(label_str)
            table_str.append(r"\end{table}")

            # Create output
            output.write("\n".join(table_str))
            output.write("\n\n")
