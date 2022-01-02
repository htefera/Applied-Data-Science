# Databricks notebook source
# MAGIC %md
# MAGIC d-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering Lab
# MAGIC 
# MAGIC **Objective**: *Apply feature engineering to a dataset to derive more meaningful features and improve predictions.*
# MAGIC 
# MAGIC In this lab, you will apply what you've learned in this lesson. When complete, please use the answers to the exercises to answer questions in the following quiz within Coursera.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 1
# MAGIC 
# MAGIC In this exercise, you will create a user-level table with the following columns:
# MAGIC 
# MAGIC 1. `avg_resting_heartrate` – the average resting heartrate
# MAGIC 1. `avg_active_heartrate` - the average active heartrate
# MAGIC 1. `avg_bmi` – the average BMI
# MAGIC 1. `avg_vo2` - the average oxygen volume
# MAGIC 1. `sum_workout_minutes` - the sum of total workout minutes
# MAGIC 1. `sum_steps` - the sum of total steps
# MAGIC 
# MAGIC Fill in the blanks in the below cell to create the `adsda.ht_user_metrics_lab` table.
# MAGIC **There is some problem between the problem description and the answer**

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_lifestyle
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-lifestyle" AS (
# MAGIC   SELECT avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS avg_bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          avg(steps) AS avg_steps,
# MAGIC          first(lifestyle) AS lifestyle
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** Why did we run a `group by`?

# COMMAND ----------

import numpy as np
np.random.seed(0)
df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()
df.loc[df.sample(frac=0.18).index, 'avg_bmi'] = np.nan
df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC ## Exercise 2
# MAGIC 
# MAGIC In this exercise, you will split your data into an training set and an inference set.
# MAGIC 
# MAGIC Fill in the blanks below to split the data.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> In practice, you should use as much data as possible for your training set. An inference set will usually become available after the training process, rather than being split apart from your training set prior to the training of the model.

# COMMAND ----------

# ANSWER
from sklearn.model_selection import train_test_split

train_df, inference_df = train_test_split(df, train_size=0.85, test_size=0.15, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** How many rows have missing values in the `avg_bmi` column in the training set?

# COMMAND ----------

# ANSWER
train_df.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3
# MAGIC 
# MAGIC In this exercise, you'll fill the `avg_bmi` with the median.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# ANSWER
import pandas as pd

avg_bmi_median = train_df['avg_bmi'].median()

train_df['avg_bmi'] = train_df['avg_bmi'].fillna(avg_bmi_median)
inference_df['avg_bmi'] = inference_df['avg_bmi'].fillna(avg_bmi_median)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** What is the value of `avg_bmi_median` rounded to the nearest hundredth place?

# COMMAND ----------

# ANSWER
print(round(avg_bmi_median, 2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 4
# MAGIC 
# MAGIC Scale the `avg_bmi`, using the `train_df` to fit and transform the data.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# ANSWER
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
scaled_training_data = sc.fit_transform(train_df[['avg_bmi']])
train_df['avg_bmi_scaled'] = scaled_training_data

scaled_inference_data = sc.transform(inference_df[['avg_bmi']])
inference_df['avg_bmi_scaled'] = scaled_inference_data

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: Using the `.min()` method on the original `avg_bmi` and `avg_bmi_scaled` columns, find the difference, rounded to the nearest tenth

# COMMAND ----------

print(inference_df['avg_bmi'].min() - inference_df['avg_bmi_scaled'].min())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 5
# MAGIC 
# MAGIC In this exercise, you will create one-hot encoded columns on the `lifestyle` column.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# ANSWER
train_df = pd.get_dummies(train_df, prefix='ohe', columns=['lifestyle'])
inference_df = pd.get_dummies(inference_df, prefix='ohe', columns=['lifestyle'])

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: How many rows in our training set (`train_df`) have a value of 1 for the column `ohe_Weight Trainer`?

# COMMAND ----------

# ANSWER
train_df[train_df['ohe_Weight Trainer'] > 0].shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 6
# MAGIC 
# MAGIC Over the next series of exercises, you will fit a Logistic Regression model, utilizing several steps above and a few new ones. 
# MAGIC 
# MAGIC Our target here is the lifestyle column. The cell below will reset our dataframe and also transform the target lifestyle column so that we have a binary classification task.

# COMMAND ----------

df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()
df.loc[df.sample(frac=0.18).index, 'avg_workout_minutes'] = np.nan
df['lifestyle'] = df['lifestyle'].map({'Sedentary':0, 'Weight Trainer':1, 'Athlete':1, 'Cardio Enthusiast':1})

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: How many observations of class `sedentary` do we have in the totality of our dataset?
# MAGIC 
# MAGIC Write the code in the below cell to answer the question.

# COMMAND ----------

# ANSWER
df["lifestyle"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 7
# MAGIC 
# MAGIC In this exercise, you will train-test split the data, using `lifestyle` as the target. Set the test size to be `10%` and the random state to `3`.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# ANSWER
from sklearn.model_selection import train_test_split
X = df.drop('lifestyle', axis=1)
y = df['lifestyle']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC ## Exercise 8
# MAGIC 
# MAGIC In this exercise, you will clean up any missing values by imputing with the mean. 
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Recall that we always want to learn values from the training set!

# COMMAND ----------

# ANSWER
avg_wo_minutes_mean = X_train['avg_workout_minutes'].mean()
X_train['avg_workout_minutes'] = X_train['avg_workout_minutes'].fillna(avg_wo_minutes_mean)
X_test['avg_workout_minutes'] = X_test['avg_workout_minutes'].fillna(avg_wo_minutes_mean)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 8
# MAGIC 
# MAGIC In this exericse, you will scale *all* of the columns.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# ANSWER

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 9
# MAGIC 
# MAGIC In this exercise, you will fit a Logistic Regression model on our target: `lifestyle`.
# MAGIC 
# MAGIC Fill in the blanks below to complete the task.

# COMMAND ----------

# ANSWER
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: 
# MAGIC What might account for our score?

# COMMAND ----------

# MAGIC %md
# MAGIC Congrats! That concludes our lesson on feature engineering!
# MAGIC 
# MAGIC Be sure to submit your quiz answers to Coursera, and join us in the next lesson to learn about feature selection.

# COMMAND ----------

# MAGIC %md
# MAGIC -sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>