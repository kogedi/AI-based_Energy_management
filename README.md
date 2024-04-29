# Smart Energy Management 
A repository for a Smart Energy Management solution to optimize purchase of energy on the one-day-ahead market based 
on a price forecast and learnings of historical data of charging events using machine learning methods and optimization tools.

## Environment
See requirements.txt 


## Optimization task
0. setup variables
1. objective
2. start x0
3. bounds
4. constraints
5. MINIMIZE OPTIMIZATION

$$
J = \min_{E_s(t)} \text {cost of energy for the household} 
$$

6. resulting vector
7. plot results
8. safe results

## Figures

1. Plots of the covariance matrix
2. Plots of optimal energy purchase based on price forecast

## Results
json and csv for scheduled energy purchase

## Utils
1. `data_preparation.py` for drop of useless variables 
2. `extract_prices.py` to make use of the historical data and one day ahead forecast 
3. `battery_distribution.py` to estimate the battery capacity to be charged at the next day given historical data
4. `max_power_distribution.py` to estimate the maximal charging power for a given vehicle based on historical data
5. `plug_distribution.py` to estimate when the vehicle is plugged a necessary condition to optimize charging
6. `data_exploration.py` for first plots, covariance matrix and filters
