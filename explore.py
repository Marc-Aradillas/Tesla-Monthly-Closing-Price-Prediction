import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy import stats


# ------------------------------- VAR PAIRS FUNCTION --------------------------------------
# defined function for plotting all variable pairs.
def plot_variable_pairs(df):
    """
    This function plots all of the pairwise relationships along with the regression line for each pair.

    Args:
      df: The dataframe containing the data.

    Returns:
      None.
    """
    sns.set(style="ticks")
    
    # Created a pairplot with regression lines
    sns.pairplot(df, kind="reg", diag_kind="kde", corner=True)
    plt.show()

# ------------------------------- CAT|CONT VARS FUNCTION --------------------------------------
def plot_categorical_and_continuous_vars(df, continuous_var, categorical_var, n):
    """
    This function outputs three different plots for visualizing a categorical variable and a continuous variable.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        continuous_var (str): The name of the column that holds the continuous feature.
        categorical_var (str): The name of the column that holds the categorical feature.

    Returns:
        None.
    """

    # Created subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Box plot of the continuous variable for each category of the categorical variable
    sns.boxplot(x=categorical_var, y=continuous_var, data=df, ax=axes[0])
    axes[0].set_title('Box plot of {} for each category of {}'.format(continuous_var, categorical_var))

    # Violin plot of the continuous variable for each category of the categorical variable
    sns.scatterplot(x=categorical_var, y=continuous_var, data=df, ax=axes[1])
    axes[1].set_title('Scatter plot of {} for each category of {}'.format(continuous_var, categorical_var))

    # Histogram of the continuous variable for each category of the categorical variable
    for cat in df[categorical_var].unique():
        sns.histplot(df[df[categorical_var] == cat][continuous_var], ax=axes[2], label=cat, kde=True, bins=n)
    axes[2].set_title('Histogram of {} for each category of {}'.format(continuous_var, categorical_var))
    axes[2].legend(title=categorical_var)

    plt.tight_layout()
    plt.show()
'''
Example:

plot_categorical_and_continuous_vars(your_dataframe, "continuous_column", "categorical_column")
'''

# __________________________________________ STATS test functions _________________________________________

def evaluate_correlation(x, y, a=0.05, method="Pearson"):
    """
    Calculate and evaluate the correlation between two variables.

    Parameters:
    - x: First variable.
    - y: Second variable.
    - significance_level: The significance level for hypothesis testing.
    - method: The correlation method to use ("Pearson" or "Spearman").

    Returns:
    - correlation_coefficient: The correlation coefficient.
    - p_value: The p-value for the correlation test.
    - conclusion: A string indicating whether to reject or fail to reject the null hypothesis.
    """

    if method == "Pearson":
        
        correlation_coefficient, p_value = stats.pearsonr(x, y)
        
    elif method == "Spearman":
        
        correlation_coefficient, p_value = stats.spearmanr(x, y)
        
    else:
        
        raise ValueError("Invalid correlation method. Use 'Pearson' or 'Spearman'.")

    
    if p_value < a:
        
        conclusion = (f"Reject the null hypothesis.\n\nThere is a significant linear correlation between {x.name} and {y.name}.")
        
    else:
        
        conclusion = (f"Fail to reject the null hypothesis.\n\nThere is no significant linear correlation between {x.name} and {y.name}.")

    return correlation_coefficient, p_value, conclusion


# you don't hvae to round your coefficient, my preference.
# Replace x and y positional arguements with your actual data in the function
# correlation_coefficient, p_value, conclusion = explore.evaluate_correlation(train.tax_amount, train.area, method="Pearson")
# print(f'{conclusion}\n\nCorrelation Coefficient: {correlation_coefficient:.4f}\n\np-value: {p_value}')