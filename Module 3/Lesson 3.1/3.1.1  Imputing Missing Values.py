# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Imputing Missing Values
# MAGIC 
# MAGIC **Objective**: *Demonstrate the imputing with the median of feature columns.*
# MAGIC 
# MAGIC In this demo, we will complete a series of exercises to handle missing values by imputing a reasonable value

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC ### Aggregate our user-level table
# MAGIC 
# MAGIC Remember that one of our project objectives is to predict a customer's `lifestyle` based on their recorded metrics. Therefore, we are interested in a table of user-level data. To prepare the dataset to do this, we'll aggregate our **`adsda.ht_user_metrics`** table at the user level.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_lifestyle
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-lifestyle" 
# MAGIC AS (
# MAGIC   SELECT avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          avg(steps) AS steps,
# MAGIC          first(lifestyle) AS lifestyle
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics_lifestyle

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Convert Spark DataFrame to Pandas
# MAGIC 
# MAGIC We will use this Pandas DataFrame in this demo.

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> 🧐 Our dataset is synthetic. which from a learning perspective, makes it easier on you as a student. However, we have a task at hand that requires missing data from the outstart.  
# MAGIC 
# MAGIC **The following cell interjects missing data into our dataset.  **
# MAGIC 
# MAGIC Outside of this notebook, there are not a lot of use cases for putting missing data *into* your dataset. You'll almost always have missing data for you already!

# COMMAND ----------

import numpy as np
ht_lifestyle_pd_df.loc[ht_lifestyle_pd_df.sample(frac=0.18).index, 'avg_resting_heartrate'] = np.nan

# COMMAND ----------

# MAGIC %md
# MAGIC View the data

# COMMAND ----------

ht_lifestyle_pd_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Looks good. Now, let's get a look at the number of missing values. There are several methods to achieve this.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 1
# MAGIC Chain together the `.isnull()` method with `.sum()`. Recall that `False` in Python is equivalent to 0. NaN values are similar to `False`, so when we use the `.sum()` method, we're asking Pandas to sum all of the non-zero values. 

# COMMAND ----------

ht_lifestyle_pd_df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 2
# MAGIC Glimpse the dataset and calculate non-null values offline. This requires that we know the count of rows in our dataset. We then take that count and subtract the non-null values to arrive at the total number of null values.

# COMMAND ----------

print(f"The number of rows in this dataset is {ht_lifestyle_pd_df.shape[0]}")
print()
ht_lifestyle_pd_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 3
# MAGIC Filter down the dataframe.
# MAGIC 
# MAGIC We can show which rows are missing. 
# MAGIC 
# MAGIC This is helpful for examining if there are patterns in the missing data!

# COMMAND ----------

ht_lifestyle_pd_df.loc[:, ht_lifestyle_pd_df.isnull().any()]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imputing Missing Values
# MAGIC Now that we know which values are missing, we can fill in the missing values with an appropriate value. First, we'll need to understand what our DataFrame's features' summary statistics look like. There are a few methods for this as well.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 1
# MAGIC Pandas has some built in methods.

# COMMAND ----------

ht_lifestyle_pd_df.describe(include='all')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 2
# MAGIC Spark DataFrame has useful methods to help do this pretty easily.

# COMMAND ----------

display(spark.sql("SELECT * FROM adsda.ht_user_metrics_lifestyle").summary())

# COMMAND ----------

# MAGIC %md
# MAGIC Regardless of method we choose to use, we see that in our column with the missing values, `avg_resting_heartrate`, which has a median and mean that differ from one another, which indicates a skew. This is in contrast to `BMI`, where we have a fairly equal mean and median. In the case where we have a skew, a median value can be more appropriate value to impute because it's more representative of the majority of values. 

# COMMAND ----------

# MAGIC %md
# MAGIC Before finding any summary stat values that we're going to perform a train test split so that our model only learns from the training set

# COMMAND ----------

from sklearn.model_selection import train_test_split
X = ht_lifestyle_pd_df.drop('lifestyle', axis=1)
y = ht_lifestyle_pd_df['lifestyle'] # target variable 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=90210)

# COMMAND ----------

# Let's examine the median of the avg_resting_heartrate column
X_train['avg_resting_heartrate'].median()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's save that to a variable

# COMMAND ----------


avg_rest_heartrate_median = X_train['avg_resting_heartrate'].median()


# COMMAND ----------

avg_rest_heartrate_median

# COMMAND ----------

# MAGIC %md
# MAGIC In order to fill in the missing values, we can use one of several possible methods

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 1
# MAGIC Comes with a copy warning.

# COMMAND ----------

X_train['avg_resting_heartrate'] = X_train['avg_resting_heartrate'].fillna(avg_rest_heartrate_median)
X_test['avg_active_heartrate'] = X_test['avg_resting_heartrate'].fillna(avg_rest_heartrate_median)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 2
# MAGIC Using `.loc`. This can be helpful if method #1 doesn't work. Typically it will, but from time to time, Pandas has issues with null vs. nan

# COMMAND ----------

# Save missing value locations to a variable for indexing
missing_values_train = X_train[X_train['avg_resting_heartrate'].isnull() == True].index
missing_values_test = X_test[X_test['avg_resting_heartrate'].isnull() == True].index
# Then using loc and the index, find all of the rows in the avg_resting_heartrate column and fill
X_train.loc[missing_values_train, 'avg_resting_heartrate'] = avg_rest_heartrate_median
X_test.loc[missing_values_test, 'avg_resting_heartrate'] = avg_rest_heartrate_median

# COMMAND ----------

# MAGIC %md
# MAGIC ### Double Checking
# MAGIC Regardless of which method we choose, we should examine for missing data.

# COMMAND ----------

X_train.isnull().sum()

# COMMAND ----------

X_test.isnull().sum()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Nicely done!

# COMMAND ----------

# MAGIC %md
# MAGIC Bonus:   
# MAGIC Imputation using other summary statistics. 
# MAGIC 
# MAGIC Imputing values using other summary statistics is very similar from a coding perspective to imputing using the median. We would need to switch out the `median()` method from earlier. Example:

# COMMAND ----------

# Save the summary statistic - the mean here - to a variable 
avg_resting_heartrate_mean = ht_lifestyle_pd_df['avg_resting_heartrate'].mean() 
# Impute the missing values with fillna()
ht_lifestyle_pd_df['avg_resting_heartrate'] = ht_lifestyle_pd_df['avg_resting_heartrate'].fillna(avg_resting_heartrate_mean)

# COMMAND ----------

# MAGIC %md
# MAGIC Practice:
# MAGIC 
# MAGIC This would also work with strings! Try it out on the `lifestyle` column.

# COMMAND ----------

# TODO
# Save the summary statistic - the mode here - to a variable 
lifestyle_most_frequent = ht_lifestyle_pd_df['lifestyle'].mode() 
# Impute the missing values with fillna()
ht_lifestyle_pd_df['lifestyle'] = ht_lifestyle_pd_df['lifestyle'].fillna(lifestyle_most_frequent)


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>