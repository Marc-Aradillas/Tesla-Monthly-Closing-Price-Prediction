import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy import stats


def dist_of_data(data, column_name, x_label=None, y_label=None, title=None, x_ticks=None):
    """
    Create a histogram of data distribution.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column to visualize.
        x_label (str, optional): Label for the x-axis.
        y_label (str, optional): Label for the y-axis.
        title (str, optional): Title for the plot.
        x_ticks (list, optional): Custom tick values and labels for the x-axis.
    """
    # Define logarithmically spaced bin edges
    bins = np.logspace(np.log10(data[column_name].min()), np.log10(data[column_name].max()), num=20)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=column_name, bins=bins, color='blue', element='step', edgecolor='black').set(title='Distribution of TSLA Stock (3 Years)')
    
    if x_ticks:
        # Customize x-axis tick labels
        tick_values, tick_labels = zip(*x_ticks)
        plt.xticks(tick_values, tick_labels)  # Set custom tick values and labels
    
    if x_label:
        plt.xlabel('Tesla Close Price')
    
    if y_label:
        plt.ylabel(y_label)
    
    if title:
        plt.title(title)
    
    plt.xscale('linear')  # Use a linear scale for the x-axis
    plt.xlim(0, data[column_name].max())  # Adjust x-axis limit as 
    plt.tight_layout()
    plt.show()


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




#===================================================================final notebook explore and stats functions==================================

import seaborn as sns
import matplotlib.pyplot as plt

def explore_question_1(tsla_train):
    """
    Explore Question 1: Does AMD stock volume have a correlation with its daily closing price?
    """
    # Visualize the relationship between volume and close price
    sns.lmplot(data=tsla_train, x='tsla_volume', y='tsla_close', scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}).set(title='Volume drives Closing Price (TSLA)')
    plt.tight_layout()
    plt.show()

    # Evaluate correlation
    correlation_coefficient, p_value, conclusion = evaluate_correlation(tsla_train.tsla_volume, tsla_train.tsla_close, method="Pearson")
    print(f'\n{conclusion}\n\np-value: {p_value}')

def explore_question_2(tsla_train):
    """
    Explore Question 2: Is there a significant correlation between the month and closing price?
    """
    # Visualize closing price by month
    plt.figure()
    sns.barplot(data=tsla_train, x='tsla_close', y='month', color='blue', errorbar=None).set(title='Stock Closing Price Data by Months (TSLA)')
    plt.xticks(rotation=45)
    plt.show()

    # Provide analysis

def explore_question_3(tsla_train):
    """
    Explore Question 3: Does AMD daily high stock price have a correlation with open stock price?
    """
    # Visualize the relationship between high and open price
    sns.lmplot(data=tsla_train, x='tsla_high', y='tsla_open', scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}).set(title='High and Open Correlation (TSLA)')
    plt.tight_layout()
    plt.show()

    # Evaluate correlation
    correlation_coefficient, p_value, conclusion = evaluate_correlation(tsla_train.tsla_high, tsla_train.tsla_open, method="Pearson")
    print(f'\n{conclusion}\n\np-value: {p_value}')

def explore_question_4(tsla_train):
    """
    Explore Question 4: Is there a significant correlation between the day of the week and closing price?
    """
    # Visualize closing price by day of the week
    plt.figure()
    sns.barplot(data=tsla_train, x='tsla_close', y='day_of_week', color='blue', errorbar=None).set(title='Stock Closing Price Data by Days of the Week (TSLA)')
    plt.xticks(rotation=45)
    plt.show()

    # Provide analysis