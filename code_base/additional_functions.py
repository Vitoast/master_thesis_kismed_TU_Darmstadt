import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os


# Example plot of two Gaussian classes and their decision boundary
def plot_two_gaussians(result_path):
    os.makedirs(result_path, exist_ok=True)
    mean1, std1 = 3, 1
    mean2, std2 = 6, 2
    x = np.linspace(0, 10, 1000)
    y1 = norm.pdf(x, mean1, std1)
    y2 = norm.pdf(x, mean2, std2)
    plt.plot(x, y1, label='Gaussian data class 1', color='red')
    plt.plot(x, y2, label='Gaussian data class 2', color='blue')
    decision_boundary = (mean1 + mean2) / 2 - 0.05
    plt.axvline(decision_boundary, color='grey', linestyle='--', label='Decision Boundary')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Two Gaussian Distributions with Decision Boundary')
    plt.savefig(os.path.join(result_path, "gaussian_db_example.png"))


# Example plot to explain linear regression, includes two scattered classes and boundary
def plot_linear_separation(result_path):
    os.makedirs(result_path, exist_ok=True)
    np.random.seed(0)
    class1 = np.random.randn(50, 2) + np.array([2, 2])
    class2 = np.random.randn(50, 2) + np.array([-2, -2])
    plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
    plt.scatter(class2[:, 0], class2[:, 1], color='blue', label='Class 2')
    x_values = np.linspace(-5, 5, 100)
    y_values = - x_values  # Change this to your specific decision boundary equation if needed
    plt.plot(x_values, y_values, color='grey', linestyle='--', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linearly Separable Classes with Possible Decision Boundary')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_path, "linear_reg_example.png"))

