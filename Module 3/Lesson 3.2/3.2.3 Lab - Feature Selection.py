# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Selection Lab
# MAGIC 
# MAGIC **Objective**: *Apply feature selection to a dataset to derive more meaningful features and improve predictions.*
# MAGIC 
# MAGIC In this lab, you will apply what you've learned in this lesson. When complete, please use the answers to the exercises to answer questions in the following quiz within Coursera.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 1
# MAGIC 
# MAGIC In this exercise, you will create a user-level table with the following columns:
# MAGIC 
# MAGIC 1. `avg_resting_heartrate` – the average resting heartrate
# MAGIC 1. `avg_active_heartrate` - the average active heartrate
# MAGIC 1. `avg_bmi` – the average BMI
# MAGIC 1. `avg_vo2` - the average oxygen volume
# MAGIC 1. `avg_workout_minutes` - the average of total workout minutes
# MAGIC 1. `avg_steps` - the average of total steps
# MAGIC 1. `lifestyle` - the lifestyle that best describes the observation
# MAGIC 
# MAGIC Run the cell below to create the table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_pca
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-pca" AS (
# MAGIC   SELECT min(resting_heartrate) AS min_resting_heartrate,
# MAGIC          avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          max(resting_heartrate) AS max_resting_heartrate,
# MAGIC          min(active_heartrate) AS min_active_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          max(active_heartrate) AS max_active_heartrate,
# MAGIC          avg(bmi) AS avg_bmi,
# MAGIC          min(vo2) AS min_vo2,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          max(vo2) AS max_vo2,
# MAGIC          min(workout_minutes) AS min_workout_minutes,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          max(workout_minutes) AS max_workout_minutes,
# MAGIC          min(steps) AS min_steps,
# MAGIC          avg(steps) AS avg_steps,
# MAGIC          max(steps) AS max_steps,
# MAGIC          avg(steps) * avg(active_heartrate) AS as_x_aah,
# MAGIC          first(lifestyle) AS lifestyle
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC Run the cell below to convert to a Pandas DataFrame and introduce missing values.

# COMMAND ----------

import numpy as np
import pandas as pd
np.random.seed(0)
df = spark.table("adsda.ht_user_metrics_pca").toPandas()
df.loc[df.sample(frac=0.18).index, 'min_active_heartrate'] = np.nan
df.loc[df.sample(frac=0.05).index, 'min_steps'] = np.nan
df.shape

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 2
# MAGIC 
# MAGIC In this exercise, you'll one-hot encode the `lifestyle` column.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# TODO
df = pd.get_dummies(df, prefix='dummy', columns=['lifestyle'])

# COMMAND ----------

# MAGIC %md
# MAGIC Run this cell to ensure that all columns are numeric.

# COMMAND ----------

df = df.apply(pd.to_numeric)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3
# MAGIC 
# MAGIC In this exercise, you'll split the data into a training set and an inference set.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split

train_df, inference_df = train_test_split(df, train_size=0.85, test_size=0.15, random_state=42)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Coursera Quiz:** How many rows have missing values in the `min_steps` column in the training set?<br>
# MAGIC **Answer=127**
# MAGIC 
# MAGIC Write your code in the empty cell below to answer the question.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer back to the previous lesson for guidance on how to complete this task.

# COMMAND ----------

# Answer
train_df.isnull().sum()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 4
# MAGIC 
# MAGIC In this exercise, you will fill in these missing values. Using the identified columns from the previous exercise, fill in the missing values with the mean of their respective column.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Recall that we want to find the mean of training set and use that to impute values on the training set *and* the test set.

# COMMAND ----------

# Answer
min_active_heartrate_mean = train_df['min_active_heartrate'].mean()
min_steps_mean = train_df['min_steps'].mean()

train_df['min_active_heartrate'] = train_df['min_active_heartrate'].fillna(min_active_heartrate_mean)
train_df['min_steps'] = train_df['min_steps'].fillna(min_steps_mean)

inference_df['min_active_heartrate'] = inference_df['min_active_heartrate'].fillna(min_active_heartrate_mean)
inference_df['min_steps'] = inference_df['min_steps'].fillna(min_steps_mean)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** What is the mean of the `min_steps_mean` feature rounded to the nearest hundredth place? <br>
# MAGIC **Answer=9090.12 **

# COMMAND ----------

# ANSWER
print(round(min_steps_mean,2))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 5
# MAGIC 
# MAGIC Create the `X_train`, `X_test`, `y_train`, `y_test` from the train_df. Recall that we are trying to predict the `avg_bmi`.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# Answer
from sklearn.model_selection import train_test_split

X = train_df.drop('avg_bmi', axis=1) # except avg_bmi
y = train_df['avg_bmi'] # Dependent Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: How many rows are in the training set? <br>
# MAGIC **Answer= 2295**

# COMMAND ----------

X_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 6
# MAGIC 
# MAGIC In this exercise, you will fit a LASSO model. Fill in the blanks to fit a model with a `0.01` alpha, then run the cells to check coefficients.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# Answer
from sklearn.linear_model import Lasso

lr = Lasso(alpha=.01)
lr.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Print out the R^2 score <br>
# MAGIC **0.90** <br>
# MAGIC 
# MAGIC This is another term for the predictioin accuracy

# COMMAND ----------

print(lr.score(X_test, y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: Which feature had the largest coefficient? <br>
# MAGIC **Answer= dummy_athlete, max_resting_heartrate**

# COMMAND ----------

pd.DataFrame(list(zip(lr.coef_, X.columns)), columns=['coef', 'feature_name']).sort_values('coef', ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 7
# MAGIC 
# MAGIC In this exercise, you will take the feature with the highest coeficients and refit a model.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# ANSWER
X = train_df[['dummy_Athlete']]
y = train_df['avg_bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)
lr = Lasso(alpha=.01)
lr.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Compute the the R-squared score.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# Answer
lr.score(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Congrats! That concludes our lesson on feature selection!
# MAGIC 
# MAGIC Be sure to submit your quiz answers to Coursera, and join us in the next lesson to learn about tree based models!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>