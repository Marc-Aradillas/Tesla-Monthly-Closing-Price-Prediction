<!--Created Anchor links to navigate read me better-->

- [Project Description](#project-description)
- [Project Goal](#project-goal)
- [Initial Thoughts](#initial-thoughts)
- [Plan](#the-plan)
- [Data Dictionary](#data-dictionary)
- [Steps to Reproduce](#steps-to-reproduce) 
- [Conclusions](#conclusions)
	- [Takeaway and Key Findings](#takeaways-and-key-findings)
	- [Reccomendations](#recommendations)
	- [Next Steps](#next-steps)

----------------------------------

# **Project: Tesla Monthly Closing Price Prediction**

<p align="center">
  <img src="https://github.com/Marc-Aradillas/volatile_company_stock_project/assets/106922826/891e38d5-28cd-4e1d-b0d7-c43aea0e4035" width="750" alt="Image">
</p>

### Predict the monthly closing price for Tesla company stock based on 3 years of recorded data from Alpha Vantage API.  
##### ***note: Tesla stock is typically volatile, more than three times as volatile as the S&P 500 in recent months.***

### Project Description

In the dynamic world of finance and investment, understanding stock price behavior is paramount. Investors and financial analysts constantly seek to predict future stock prices to make well-informed decisions. This project focuses on predicting the monthly closing prices of company stocks, with a particular emphasis on those stocks that exhibit higher volatility, as they present unique opportunities and risks.

Leveraging three years of historical stock price data from the Alpha Vantage API, my goal is to identify the key drivers of monthly closing prices for Tesla company stocks and develop robust machine learning models for accurate predictions. The dataset includes crucial features such as open price, high price, low price, closing price, and trading volume. By creating a reliable predictive model, my project aims to empower investors and analysts to navigate the complex landscape of stock trading with more confidence and precision.

### Project Goal

1. Identify Drivers of Monthly Closing Prices: The primary objective is to explore the dataset and identify the factors that significantly influence monthly closing prices. Understanding these drivers is essential for building an effective predictive model.

2. Develop Machine Learning Models: Utilize the identified drivers to develop machine learning models capable of predicting monthly closing prices accurately. The project will explore multiple regression-based models, including RandomForestRegressor and LinearRegression, to determine the best-performing approach.

### Initial Thoughts

My initial hypothesis is that drivers of Tesla's closing stock prices will be trading volume and high prices.

## The Plan

* Acquire historical stock price data from the Alpha Vantage API.
* Prepare data
* Explore data in search of drivers of stock closing price
  * Answer the following initial questions
	* Does TSLA stock volume have a correlation with it's daily closing price? 
  	* Is there a significant relationship between the month in which TSLA stock was traded and its closing price?
 	 * Does TSLA daily high stock price have a correlation with open stock price?  
 	 * Is there a significant correlation between the month in which TSLA stock was traded and its closing price?
* Develop a Model to predict Tesla stock closing price
  * Use drivers identified in explore to help build predictive models of different types
  * Feature engineer data if able, no preprocess to include all values.
  * Evaluate models on train and validate data
  * Select the best model based on $RMSE$ and $R^2$
  * Evaluate the best model on test data
* Draw conclusions

### Data Dictionary

| **Feature**        | **Data Type** | **Definition**                                       |
|--------------------|---------------|-----------------------------------------------------|
| `tsla_open`        | Float         | The opening stock price of TSLA on a given date.    |
| `tsla_high`        | Float         | The highest stock price of TSLA during the day.    |
| `tsla_low`         | Float         | The lowest stock price of TSLA during the day.     |
| `tsla_close`       | Float         | The closing stock price of TSLA on a given date.    |
| `tsla_volume`      | Integer       | The volume of TSLA shares traded on that date.     |
| `month`            | String        | The month when the stock data was recorded.         |
| `day_of_week`      | String        | The day of the week when the stock data was recorded.|
| `year`             | Integer       | The year when the stock data was recorded.         |
| `next_month_close` | Float         | **(Target Variable)** The closing stock price of TSLA in the following month.|


## Steps to Reproduce

1. Clone this project repository to your local machine.

2. Install project dependencies by running pip install -r requirements.txt in your project directory.

3. Obtain an API key from the Alpha Vantage website.

4. Create a config.py file in your project directory with your API key using the following format:

> ALPHA_VANTAGE_API_KEY = "YOUR_API_KEY"
 
5. Ensure that config.py is added to your .gitignore file to protect your API key.

6. Run the acquire.py script to fetch stock data from the Alpha Vantage API:

> python acquire.py

7. Execute the prepare.py script for data preprocessing and splitting:

> python prepare.py

8. Explore the dataset and answer initial questions using the explore.py script:

> python explore.py

9. Develop machine learning models by running the model.py script:

> python model.py

10. Evaluate the models, select the best-performing one, and draw conclusions based on the results of the model.py script.


# Conclusion

## Takeaways and Key Findings

- Company stocks with low volatility or are considered stable would not benefit from regression modeling.
- Company stocks with high volatility need to be analyzed over a shorter amount of time versus the span of twenty plus years.
- It's easier to analyze data when there is ups and downs on the stock you are trying to predict closing price.
- Tesla's stock is not easy to predict accurately and it goes the same for other volatile stocks explored during the exploration phase outside of the data science project.


## Model Improvement
- The model does well with default setting and hyperparameter tuning may or may not aid in regression modeling efforts.

## Recommendations and Next Steps

- I would recommend maybe gaining sentiment data, user data, and other forms of unstructured data that can be used with deep learning methodoligies may help in predicting certain pricing elements for high volatile company stocks. Additionally I would also detail time frames to adjust for open and closed time frames of the stock market for optimal opportunity in price predictions.

- Given more time, the following actions could be considered:
  - Gather more data to improve model performance.
  - Revisit the data exploration phase to gain a more comprehensive dataset.
    - Time Series Analysis would have been my alternate route in predicting closing prices.
      - Utilizing Time series models like:
        - ARIMA (AutoRegressive Integrated Moving Average)
          - or more advanced techniques like LSTM (Long Short-Term Memory) and FBProphet are designed specifically for time-dependent data like stock prices.


>  In this individual project, I leveraged data science techniques to create a robust and versatile financial analysis and prediction tool. The primary goal was to develop a comprehensive solution for analyzing historical stock market data, making predictions, and evaluating the performance of different machine learning models.
