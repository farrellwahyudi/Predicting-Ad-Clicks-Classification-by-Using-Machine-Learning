# Predicting Ad Clicks: Classification by Using Machine Learning
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/header.png" alt="Predict Ad Clicks" style="width:600px;height:400px;">

## Background
An Indonesian company is interested in evaluating the performance of advertisement on their website. This evaluation is essential for the company as it helps gauge the advertisement’s reach and its ability to engage customers.

By analyzing historical advertisement data and uncovering insights and trends, this approach can aid the company in defining their marketing objectives. In this specific case, the primary emphasis lies in developing a machine learning classification model that can accurately identify the target customer demographic.

## Problem
1. The company in question currently displays ads for all of its users. This “shotgun” strategy yields them an ad click rate of 50%.
2. The company spends a nonoptimal amount of resources by displaying ads for all of its users.

## Objectives
1. Create well-fit machine learning models that can reliably predict which users are likely to click on an ad.
2. Determine the best machine learning model to be implemented in this use case based on; evaluation metrics (Recall and Accuracy), model simplicity, and prediction time.
3. Identify the factors that most influence users’ likelihood to click on an ad.
4. Provide recommendations for potential strategies regarding targeted ads to marketing and management teams based on findings from analyzes and modeling.
5. Calculate the potential impact of model implementation on profit and click rate.

## About the Dataset
The dataset was obtained from [Rakamin Academy](https://www.rakamin.com/).

**Description:**

- <code>Unnamed: 0</code> = ID of Customers
- <code>Daily Time Spent on Site</code> = Time spent by the user on a site in minutes
- <code>Age</code> = Customer’s age in terms of years
- <code>Area Income</code> = Average income of geographical area of consumer
- <code>Daily Internet Usage</code> = Average minutes in a day consumer is on the internet
- <code>Male</code> = Gender of the customer
- <code>Timestamp</code> = Time at which user clicked on an Ad or the closed window
- <code>Clicked on Ad</code> = Whether or not the customer clicked on an Ad (Target Variable)
- <code>city</code> = City of the consumer
- <code>province</code> = Province of the consumer
- <code>category</code> = Category of the advertisement

**Overview:**

1. Dataset contains 1000 rows, 10 features and 1 <code>Unnamed: 0</code> column which is the ID column.
2. Dataset consists of 3 data types; float64, int64 and object.
3. <code>Timestamp</code> feature could be changed into datetime data type.
4. Dataset contains null values in various columns.

## Data Analysis
### Univariate
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/uni_nums.png" alt="Univariate Numerical" style="width:600px;height:400px;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>KDE Plot of Numerical Features</b>
<br><br><br>
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/uni_nums2.png" alt="Univariate Numerical" style="width:700px;height:300px;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Boxplot of Numerical Features</b>

**Analysis:**

- <code>Area Income</code> is the only feature with a slight skew (left-skewed).
- <code>Daily Internet Usage</code> is nearly uniformly distributed.
- While <code>Age</code> and <code>Daily Time Spent on Site</code> is nearly normally distributed.

### Bivariate
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/bivariate_nums.png" alt="Bivariate Numerical" style="width:600px;height:400px;">
&nbsp;&nbsp;&nbsp;&nbsp;<b>KDE Plot of Numerical Features Between Users That Clicked and Didn’t Click On Ad
</b>
<br><br><br>
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/bivariate_nums2.png" alt="Bivariate Numerical" style="width:700px;height:400px;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Boxplot of Numerical Features Between Users That Clicked and Didn’t Click On Ad</b>
<br>

### Scatterplot

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/age_vs_daily_internet_usage.png" alt="Scatterplot" style="width:500px;height:500px;">
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/age_vs_daily_time_spent_on_site.png" alt="Scatterplot" style="width:500px;height:500px;">
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/daily_internet_usage_vs_daily_spent_time_on_site.png" alt="Scatterplot" style="width:500px;height:500px;">

**Analysis:**

- The more time is spent on site by the customer the less likely they will click on an ad.
- The average age of customers that clicked on an ad is 40, while the average for those that didn’t is 31.
- The average area income of customers that clicked on an ad is considerably lower than those that didn’t.
- Similar to time spent, the more the daily internet usage is, the less likely the customer will click on an ad.
- As can be seen the last scatterplot above, there is a quite clear separation between two clusters of data. One cluster is less active and the other more so. Less active customers have a higher tendency to click on an ad compared to more active customers.

### Multivariate (Numerical)
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/corr_nums.png" alt="Numerical Correlation" style="width:500px;height:500px;">

Since the target variable is binary (Clicked or didn’t click), we can’t use the standard Pearson correlation to see the correlation between the numerical features and the target. Hence, the **Point Biserial correlation** was used to analyze the correlation between the target and the numerical features.

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/corr_nums_target.png" alt="Numerical Correlation with Target" style="width:500px;height:400px;">

### Multivariate (Categorical)
In order to see the correlation between categorical features, I again couldn’t employ the standard Pearson correlation. There are numerous methods out there, but in this study I used **Cramer’s V** to understand the association and by extension the correlation between categorical features.

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/corr_cats.png" alt="Categorical Correlation" style="width:500px;height:500px;">

**Analysis:**

- The perfect correlation coefficient of 1 indicates that <code>city</code> and <code>province</code> are perfectly associated. This makes sense, since if you knew the city you would also know the province. This means that using both features for machine learning modeling is redundant.
- All the numerical features (especially <code>Daily Internet Usage</code> and <code>Daily Time Spent on Site</code>) have high correlation with the target variable.

## Data Preprocessing
### Handling Missing Values
Missing or null values were present in various columns in the data. These null values were imputed using statistically central values such as mean or median accordingly.

### Feature Extraction
The <code>Timestamp</code> feature was of “object” data type. This feature was transformed into “date-time” data type in order to have its contents extracted. There are 3 extra features that were extracted from the Timestamp feature, these are:

1. Month
2. Week
3. Day (Mon-Sun)

### Handling Outliers
Outliers were present in the <code>Area Income</code> feature. To help the data in becoming more “model friendly” to linear and distance based models, these outliers were removed from the data (using IQR method).

**Before:**

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/outliers_before.png" alt="Outliers Before" style="width:500px;height:150px;">

**After:**

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/outliers_after.png" alt="Outliers Before" style="width:500px;height:150px;">

### Feature Selection
From the multivariate analysis it can be seen that <code>city</code> and <code>province</code> are redundant. However, both features have high cardinality and don’t correlate too well with the target variable. As a result, both of these features were excluded in the modeling as to prevent the curse of dimensionality. <code>Timestamp</code> was also excluded since its contents have been extracted and it was no longer needed. <code>Unnamed: 0</code> or ID was obviously excluded since it is an identifier column and is unique for every user.

### Feature Encoding
The categorical features were encoded so that machine learning models could read and understand its values. The <code>category</code> feature was One-Hot encoded whilst the rest (<code>Gender</code> and <code>Clicked on Ad</code>) were label encoded.

### Dataset Splitting
The data was split into training and testing sets. This is common practice as to make sure that the machine learning models weren’t simply memorizing the answers from the data it was given. The data was split in a 75:25 ratio, 75% training data and 25% testing data.

## Modeling
In the modeling phase, an experiment was conducted. Machine learning models were trained and tested with non-normalized/non-standardized version of the data. The results of which were compared to models trained and tested with normalized/standardized version of the data. Hyperparameter tuning was done in both scenarios, as to get the best out of each model. The model and standardization method with the best results will be selected. Six models were experimented with, these are:

- <code>Logistic Regression</code>
- <code>Decision Tree</code>
- <code>Random Forest</code>
- <code>K-Nearest Neighbors</code>
- <code>Gradient Boosting</code>
- <code>XGBoost</code>

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/clicked_on_ad.png" alt="Target Variable" style="width:300px;height:50px;">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Target Variable Class Balance</b>
<br><br>
Because the target has perfect class balance the primary metric that will be used is <code>Accuracy</code>. <code>Recall</code> will be the secondary metric as to minimize false negatives.

### Without Normalization/Standardization
**Results:**

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/before_normalization.png" alt="Before Normalization/Standardization" style="width:800px;height:200px;">

**Observation:**
- <code>Decision Tree</code> had the lowest fit time of all the models but the second lowest accuracy overall.
- <code>Gradient Boosting</code> had the highest accuracy and recall scores but <code>XGBoost</code> is not far behind.
- Due to the non-normalized data, distance based algorithms like <code>K-Nearest Neighbours</code> and linear algorithms like <code>Logistic Regression</code> suffered heavily.
  - <code>Logistic Regression</code> could not converge properly using newton-cg and as a result had the highest fit time of all the models, even though it probably is the simplest model of them all.
  - <code>K-Nearest Neighbours</code> suffered in accuracy and recall scores, with both being by far the lowest of all the models tested.

### Using Normalization/Standardization
**Results:**

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/before_normalization.png" alt="After Normalization/Standardization" style="width:800px;height:200px;">

**Observation:**
- <code>K-Nearest Neighbours</code> had the highest fit time and elapsed time.
- <code>Gradient Boosting</code> had the highest accuracy and the highest recall (tied with <code>Random Forest</code>), but <code>Logistic Regression</code> in close second, had the highest cross-validated accuracy of all the models tested.
- <code>Random Forest</code> and <code>XGBoost</code> also had nearly identical scores in close third and fourth, although <code>XGBoost</code> had the better fit and elapsed times.
- With normalized data, the previously poor performing distance based and linear models have shone through.
  - <code>Logistic Regression</code>'s fit and elapsed times had been reduced significantly making it the model with the lowest times. It's scores have also massively improved making it a close second-place model.
  - <code>K-Nearest Neighbours</code> also saw massive improvement in its scores, with the model no longer sitting in last place in terms of scores.
 
By taking consideration of not only the above metrics but also the simplicity, explainability and fit and elapsed times, the model that will be chosen is the <code>Logistic Regression</code> model with normalization/standardization. This is not only because of the very high scores (especially cross-validated scores) but also the simplicity, explainability and relatively quick fit and elapsed times.

### Evaluation of Selected Model (Logistic Regreession)
**Learning Curve:**

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/learning_curve_best_model.png" alt="Learning Curve" style="width:700px;height:400px;">
As can be seen from the above learning curve, the tuned <code>Logistic Regression</code> model with normalization/standardization is well fitted with no overfitting/underfitting.
<br><br>

**Confusion Matrix:**

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/confusion_matrix_best_model.png" alt="Confusion Matrix" style="width:600px;height:400px;">

From the test set confusion matrix above, from 120 people that clicked on an ad the algorithm correctly classified 116 of them and incorrectly classified 4 of them. Similarly, out of 128 people that did not click on an ad the algorithm correctly classified 124 of them and incorrectly classified only 4.

Based on the confusion matrix and also the learning curve, it can be seen that <code>Logistic Regression</code> is a more than capable model to be implemented on this dataset.

**Feature Importance:**

Since <code>Logistic Regression</code> is such a simple and explainable model, to get the feature importance we can simply look at the <code>coefficients</code> of each feature in the model.

The <code>coefficients</code> represent the change in the log odds for a one-unit change in the feature variable. Larger absolute values indicate a stronger relationship between the feature and the target variable, while the sign of the <code>coefficients</code> (negative or positive) indicates the direction of the relationship between the two.

<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/feature_importance.png" alt="Feature Importance" style="width:600px;height:300px;">
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/feature_importance2.png" alt="Feature Importance" style="width:600px;height:300px;">

**Analysis:**

Based on the feature importance charts above, it can clearly be seen that the two features with most effect on the model are <code>Daily Time Spent on Site</code> and <code>Daily Internet Usage</code>.
- The lower the <code>Daily Time Spent on Site</code> the bigger the odds that the customer will click on an ad and vice versa.
- Similarly, the lower the <code>Daily Internet Usage</code> the higher the chances that the customer will click on an ad and vice versa.
- Other Important features include; <code>Area Income</code> and <code>Age</code>.

## Business Recommendations
Based on the insights that have been gathered in the EDA as well as the feature importance from the model, the following business recommendations are formulated.

- **Content Personalization and Targeting:**
Since the lower the Daily Time Spent on Site is the more likely the user is to click on an ad, it’s essential to focus on content personalization and user engagement. Tailor content to keep users engaged but not overloaded. This can be achieved through strategies like recommending relevant content and using user data to customize the user experience.
- **Age-Targeted Advertising:**
Older individuals are more likely to engage with ads. Therefore we can consider creating ad campaigns that are specifically designed to target and appeal to older demographics. This may include promoting products or services relevant to their age group.
- **Income-Level Targeting:**
Users in areas with lower income levels are more likely to click on ads. Therefore we can create ad campaigns that are budget-friendly and appealing to users with lower income. Additionally, consider tailoring the ad messaging to highlight cost-effective solutions.
- **Optimize Ad Placement for Active Internet Users:**
Heavy internet users are less responsive to ads. To improve ad performance, consider optimizing ad placement for users with lower internet usage or finding ways to make ads stand out to this group, such as through eye-catching visuals or unique offers.

## Potential Impact Simulation
<img src="https://github.com/farrellwahyudi/Predicting-Ad-Clicks-Classification-by-Using-Machine-Learning/blob/main/Images/clicked_on_ad.png" alt="Target Variable" style="width:250px;height:50px;">
Using the original dataset’s Clicked on Ad numbers as can be seen above, the business simulation of before and after model implementation are as follows:
<br><br>

**Assumption:**

Cost per Advertisement: Rp.1000

Revenue per Ad clicked: Rp.4000
****
**Before model implementation:**

- **No. Users Advertised**:<br>
Every User = 1000
- **Click Rate**: <br>
500/1000 = 50%
- **Total Cost**: <br>
No. Users Advertised x Cost per Ad = 1000 x 1000 = Rp.1,000,000
- **Total Revenue**: <br>
Click Rate x No. Users Advertised x Revenue per Ad Clicked = 0.5 x 1000 x 4000 = Rp.2,000,000
- **Total Profit**:<br>
Total Revenue - Total Cost = **Rp.1,000,000**

**After model implementation:**

- **No. Users Advertised**:<br>
(Precision x 500) + ((1-Specificity) x 500) = (96.67% x 500) + (0.03125 x 500) = 483 + 16 = 499
- **Click Rate**:<br>
(Precision x 500)/No. Users Advertised = 483/499 = 96.8%
- **Total Cost**:<br>
No. Users Advertised x Cost per Ad = 499 x 1000 = Rp.499,000
- **Total Revenue**:<br>
Click Rate x No. Users Advertised x Revenue per Ad Clicked = 0.968 x 499 x 4000 = Rp.1,932,000
- **Total Profit**:<br>
Total Revenue - Total Cost = 1,932,000 - 499,000 = **Rp.1,433,000**

**Conclusion:**

By comparing the profits and click rates of before and after model implementation, we can see that with model implementation click rate is up from **50%** to **96.8%**, and similarly profit is up from **Rp.1,000,000** to **Rp.1,433,000** (a **43.3%** increase).

<img src="https://cdn-icons-gif.flaticon.com/6416/6416353.gif" alt="Predict Ad Clicks" style="width:400px;height:400px;" loop=infinite>
