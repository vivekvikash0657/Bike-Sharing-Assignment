# Project Name
> Linear Regression - Bike Sharing 


## Table of Contents
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor



## General Information
Problem Statement
This assignment is a programming assignment wherein you have to build a multiple linear regression model for the prediction of demand for shared bikes. You will need to submit a Jupyter notebook for the same. 

 

Problem Statement
A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.


A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 


In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.


They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

Which variables are significant in predicting the demand for shared bikes.
How well those variables describe the bike demands
Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors. 


Business Goal:
You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market. 


Data Preparation:

You can observe in the dataset that some of the variables like 'weathersit' and 'season' have values as 1, 2, 3, 4 which have specific labels associated with them (as can be seen in the data dictionary). These numeric values associated with the labels may indicate that there is some order to them - which is actually not the case (Check the data dictionary and think why). So, it is advisable to convert such feature values into categorical string values before proceeding with model building. Please refer the data dictionary to get a better understanding of all the independent variables.
 
You might notice the column 'yr' with two values 0 and 1 indicating the years 2018 and 2019 respectively. At the first instinct, you might think it is a good idea to drop this column as it only has two values so it might not be a value-add to the model. But in reality, since these bike-sharing systems are slowly gaining popularity, the demand for these bikes is increasing every year proving that the column 'yr' might be a good variable for prediction. So think twice before dropping it. 
 

Model Building

In the dataset provided, you will notice that there are three columns named 'casual', 'registered', and 'cnt'. The variable 'casual' indicates the number casual users who have made a rental. The variable 'registered' on the other hand shows the total number of registered users who have made a booking on a given day. Finally, the 'cnt' variable indicates the total number of bike rentals, including both casual and registered. The model should be built taking this 'cnt' as the target variable.


Model Evaluation:
When you're done with model building and residual analysis and have made predictions on the test set, just make sure you use the following two lines of code to calculate the R-squared score on the test set.

 

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
 

where y_test is the test data set for the target variable, and y_pred is the variable containing the predicted values of the target variable on the test set.
Please don't forget to perform this step as the R-squared score on the test set holds some marks. The variable names inside the 'r2_score' function can be different based on the variable names you have chosen.


## Conclusions
Multiple Linear Regression Model Summary

Dependent Variable: count

Coefficients:
const            0.0753
yr               0.2331
workingday       0.0563
temp             0.5499
windspeed       -0.1552
season_2         0.0874
season_4         0.1318
mnth_9           0.0972
weekday_6        0.0677
weathersit_2    -0.0813
weathersit_3    -0.2880

R square value = 0.7955844377237249

Adjusted R square Value = 0.7857567664604425

Interpretation:

The equation of best fitted surface based on the best fit model is:

cnt = 0.0753 + (yr × 0.2331) + (workingday × 0.0563) + (temp × 0.5499) − (windspeed × 0.1552) + (season2 × 0.0874) + (season4 ×0.1318) + (mnth9 × 0.0972) + (weekday6 ×0.0677) − (weathersit2 × 0.0813) − (weathersit3 × 0.2880)

Final Report on Bike Booking Prediction

Overview

The final multiple linear regression model aims to predict bike bookings based on a set of predictor variables. After thorough analysis, I have identified the impact of each variable on bike bookings.

Top Predictor Variables:
Year (yr):
Coefficient: 0.2331
Interpretation: A unit increase in the year variable increases the bike hire numbers by 0.2331 units.

working day (workingday):
Coefficient: 0.0563
Interpretation: A unit increase in the year variable increases the bike hire numbers by 0.0563 units.

Temperature (temp):
Coefficient: 0.5499
Interpretation: A unit increase in the temperature variable increases the bike hire numbers by 0.5499 units.

Windspeed (windspeed):
Coefficient: -0.1552
Interpretation: A unit increase in windspeed decreases the bike hire numbers by -0.1552 units.

season_2 (season_2):
Coefficient: 0.0874
Interpretation: A unit increase in season_2 increases the bike hire numbers by 0.0874 units.

season_4 (season_4):
Coefficient: 0.1318
Interpretation: A unit increase in season_4 increases the bike hire numbers by 0.1318 units.

mnth_9 (mnth_9):
Coefficient: 0.0972
Interpretation: A unit increase in mnth_9 increases the bike hire numbers by 0.0972 units.

weekday_6 (weekday_6):
Coefficient: 0.0677
Interpretation: A unit increase in weekday_6 increases the bike hire numbers by 0.0677 units.

weathersit_2 (weathersit_2):
Coefficient: - 0.0813
Interpretation: A unit increase in weathersit_2 decreases the bike hire numbers by 0.0813 units.

season_4 (season_4):
Coefficient: - 0.2880
Interpretation: A unit increase in season_4 decreases the bike hire numbers by 0.2880 units.

## Technologies Used
- library - version 1.0
- library - version 2.0
- library - version 3.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
Give credit here.
- This project was inspired by...
- References if any...
- This project was based on [this tutorial](https://www.example.com).


## Contact
Created by [@Vivek Vikash] - feel free to contact me!


Use of this dataset in publications must be cited to the following publication:

[1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

@article{
	year={2013},
	issn={2192-6352},
	journal={Progress in Artificial Intelligence},
	doi={10.1007/s13748-013-0040-3},
	title={Event labeling combining ensemble detectors and background knowledge},
	url={http://dx.doi.org/10.1007/s13748-013-0040-3},
	publisher={Springer Berlin Heidelberg},
	keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
	author={Fanaee-T, Hadi and Gama, Joao},
	pages={1-15}
}

=========================================
Contact
=========================================
	
For further information about this dataset please contact Hadi Fanaee-T (hadi.fanaee@fe.up.pt)
