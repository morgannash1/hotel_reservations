# Predicting Hotel Reservation Cancellations

Author: [Morgan Nash](mailto:morganmichellenash@gmail.com)

December 2025

See the full analysis in the [Jupyter Notebook](notebook.ipynb) 

# Project Objective:
The goal of this project is to create a predictive classification model that identifies when a hotel reservation is high-risk before it's cancelled, allowing your team to proactively intervene. We try several classification models and select a top performing model with high Precision to minimize the cost of offering unnecessary incentives to committed guests.

# Business Understanding:

Maintaining occupancy is a huge challenge for the modern hotel industry. Intense competition, outdated technology systems, as well as the ease of online booking (and cancelling), are just a few of the issues that hotels are required to battle. It goes without saying that every cancelled booking means revenue is lost. 

* Aggressive pricing from competitors, along with the rise of short-term rental platforms like AirBnb, make guest retention increasingly difficult.
  
* On top of that, many of the hotel reservation systems are outdated and lack the capabilities that are needed to predict customer behavior, limiting the reliability of reservation projections.
  
* Another reason for occupancy struggles is the ease of online booking and common "free cancellation" policies. This allows for customers to make multiple reservations simultaneously, which greatly increases the chance of last-minute cancellations.

With all of this, it's not shocking that your hotel is having issues with cancelled reservations, and we understand the immediate need for a predictive intervention system. 

We are hoping to help with that by creating a classification model that predicts when a hotel reservation is "high risk" so your team can intervene and reach out with incentives to secure the booking.

# Data Understanding

This project uses the Hotel Reservations Dataset, accessed via [Kaggle](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset/data), and contains 36,275 records detailing customer reservations and characteristics from 2017 to 2018. This dataset is a good choice for this project for the following reasons:

* It contains real data, rather than synthethic. This data represents genuine booking activity from a single hotel, although the hotel location is undisclosed. This means our classification model will learn patterns based on real data which allows us to derive practical insights from the real customer behavior.

* It has a binary target: booking_status ('Canceled' or 'Not_Canceled')

* It contains a mix of variables that can be tied to a customer's likelihood to cancel including:

    - Customer Demographics: number of adults and children, whether customer was a repeat guest or not
    - Customer Behavior: number of weekend and weeknights booked, type of meal plan, lead time (days between booking and arrival), special requests, past booking history, whether or not a parking space was required
    - Financial & Operational Information: Average room price, room type


This Data Dictionary, taken from the dataset's Kaggle page, describes each of the 19 columns:

**Booking_ID:** unique identifier of each booking

**no_of_adults:** Number of adults

**no_of_children:** Number of Children

**no_of_weekend_nights:** Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel

**no_of_week_nights:** Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel

**type_of_meal_plan:** Type of meal plan booked by the customer:

**required_car_parking_space:** Does the customer require a car parking space? (0 - No, 1- Yes)

**room_type_reserved:** Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.

**lead_time:** Number of days between the date of booking and the arrival date

**arrival_year:** Year of arrival date

**arrival_month:** Month of arrival date

**arrival_date:** Date of the month

**market_segment_type:** Market segment designation

**repeated_guest:** Is the customer a repeated guest? (0 - No, 1- Yes)

**no_of_previous_cancellations:** Number of previous bookings that were canceled by the customer prior to the current booking

**no_of_previous_bookings_not_canceled:** Number of previous bookings not canceled by the customer prior to the current booking

**avg_price_per_room:** Average price per day of the reservation; prices of the rooms are dynamic. (in euros)

**no_of_special_requests:** Total number of special requests made by the customer (e.g. high floor, view from the room, etc)

**booking_status:** Flag indicating if the booking was canceled or not.


### Data Limitations:
The following list contains this dataset's primary limitations, and focuses on factors that could impact the model's generalizability:

* Limited Geographic Scope and Timeline: The data is restricted to a 1 hotel only and is slightly outdated with the latest record from 2018 (before the Covid-19 Pandemic). The hotel location is also undisclosed. The dataset creator, Ahsan Raza, commented in the Discussion section: "This example data has been captured from single location/country which, due to discretionary reasons, cannot be disclosed." This adds a limit to the model's ability to generalize well to other locations after 2018.

* Missing External and Factors: The dataset lacks external and economic influences. Information about competitor pricing, weather forecasts, or large events taking place nearby are just to name a few outside factors that frequently drive cancellation decisions, but they are invisible to the current model.

* Lack of Detailed Guest and Pricing Data: Certain customer demographic details (like age and income), as well as records of the actual price paid by the customers (only the average is included) are absent. This limits the model's ability to truly understand a customer's price sensitivity.

* Feature Ambiguity: Interpretation is hampered by the ciphered room type values (ie. room type, meal plan type), which cannot be leveraged fully without the encoding key, which is not included. 




## Data Preparation:



Numerical Columns Initial Observations:
The required_car_parking_space and repeated_guest columns should be treated as categorical (0, or 1).
There are outliers in the following columns:
lead_time
no_of_children (75th percentile is still 0.0, max is 10)
no_of_week_nights (75th percentile is 3, max is 17)
no_of_previous_cancellations (75th percentile is 0, max is 13)
no_of_previous_bookings_not_cancelled (75th percentile is 0, max is 58)
I am curious if there are any records where no_of_weekend_nights and no_of_week_nights for a certain record are both 0.

Additional Data Preprocessing:
In this section, we:

identify the features and target for the models
train, test, split the data to prevent data leakage
create a dummy model to produce baseline metrics so we ensure our models performance improves
define OneHotEncoder to use for categorical columns
define StandardScaler to use for numerical columns


Data Cleaning and Feature Engineering:
In this section, we:

check for duplicate rows
drop Booking_ID column
create a column 'is_cancelled' that maps the values in 'booking_status' to 1 or 0: 'Cancelled' to 1 and 'Not_Cancelled' to 0
address some of the low-frequency categories found in certain categorical columns' value counts by consolidating
make sure binary/discrete integer columns are changed to data type 'object' (required_car_parking_space, repeated_guest, and arrival_year(because there are only 2 years)) so they'll be treated as categorical
plot the distribution of numeric columns
manually add a maximum cap to certain numerical columns to handle their outliers
check for (and drop) records where the total number of nights is zero


## Model Evaluation Metrics:¶
During modeling, we set "Canceled" reservations to 1, and reservationes that were "Not Canceled" to 0. Below, we explain the difference in the costs between False Positives and False Negatives:

**False Positive:** Our model predicts that a reservation is "high risk" but the guests were planning on showing up.
**Cost:** You end up spending money, time, and effort on trying to save a booking that was never at risk of canceling.
High Precision minimizes these False Positives. Of all predicted positives (cancellations), how many were correct? It is important because money spent on a customer who was already committed is a direct, unnecessary cost.

**False Negative:** Our model predicts the reservation was low-risk, but the guest actually canceled the booking.
**Cost:** You fail to intervene and lose the revenue associated with that booking.
High Recall minimizes False Negatives. Of all actual positives (cancellations), how many did we find? This is relevant for intervening to save bookings.

These costs are both very important to consider. Businesses often accept a slightly lower Recall if it means the model's Precision is very high, and they are highly confident in every intervention they make. To align with our goal of proactive cancellation prevention, we decide to prioritize a model with **High Precision** but still take other evaluation metrics into consideration.

# Modeling:

We decided to try the following models before selecting a top performer (High Precision) to fine tune its parameters:

**Logistic Regression:** simple, interpretable model that establishes a performance baseline and helps to clearly identify which features have a direct, positive or negative influence on the risk of cancellation
**Random Forest:** bagging ensemble model that combines the predictions from hundreds of independent decision trees and is better at finding hidden, non-linear patterns in the data that a simple model might miss
**Gradient Boosting:** boosting ensemble model that builds trees one after the other, with each new tree focused on fixing the errors made by the previous ones. This was included to challenge the Random Forest model.

The following sections, split by model type, each create a pipeline that uses the preprocessors defined above along with the specific model. Each pipeline is fit on the training data and predicts test data. A Classification Report, Confusion Matrix, ROC AUC score, and ROC Curve Plot are displayed for each pipeline. Finally, we evaluate Feature Importance from each model to gain insight on what features lead a reservation to be high-risk. (A reminder that feature importance is a relative score, not a percentage.) After we evaluate these models, we tune the top performing model.

# Evaluation

## Limitations

## Recommendations

## Next Steps

## For More Information

See the full analysis in the [Jupyter Notebook](notebook.ipynb) 
