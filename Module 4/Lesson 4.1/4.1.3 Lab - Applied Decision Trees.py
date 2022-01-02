# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Applied Decision Trees Lab
# MAGIC 
# MAGIC **Objective**: *Apply decision trees to a regression problem in an effort to represent more complex relationships without overfitting.*
# MAGIC 
# MAGIC In this lab you will apply what you've learned about decision trees and overfitting. When complete, please use your answers to the exercises to answer questions in the following quiz within Coursera.

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md-sandbox
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
# MAGIC 
# MAGIC Fill in the blanks in the below code block to aggregate the metrics by `device_id`.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer back to previous demos and labs for guidance on creating a table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Answer
# MAGIC CREATE OR REPLACE TABLE adsda.ht_daily_metrics_agg
# MAGIC USING DELTA LOCATION "/adsda/ht-daily-metrics-agg" AS (
# MAGIC   SELECT device_id,
# MAGIC          avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          avg(steps) AS steps
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** How many distinct values of `device_id` are in the aggregated table?<br>
# MAGIC **Answer=3000**
# MAGIC 
# MAGIC Write your code in the below empty cell to answer the question.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Answer
# MAGIC Select distinct count(device_id) from adsda.ht_daily_metrics_agg

# COMMAND ----------

# MAGIC %md
# MAGIC Fill in the following cell to create a Pandas DataFrame from the Spark table.

# COMMAND ----------

# Answer
ht_metrics_pd_df =spark.table("adsda.ht_daily_metrics_agg").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Separate the DataFrame into the features (X) and the target (y). In this case, we want to predict a user's average workout minutes, using only the numeric features as predictors.

# COMMAND ----------

# Answer
X = ht_metrics_pd_df[['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'avg_vo2', 'steps']]
y = ht_metrics_pd_df['avg_workout_minutes']

# COMMAND ----------

# MAGIC %md
# MAGIC Perform a train-test split. Set a random state seed to 42 so that you will obtain consistent results.

# COMMAND ----------

# Answer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** How many rows are in the training and test sets? <br>
# MAGIC **Answer=2250 and 750**

# COMMAND ----------

X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 2 
# MAGIC 
# MAGIC In this exercise, you will fit and train a base decision tree regression model.
# MAGIC 
# MAGIC For this first model, use the default hyperparameters.
# MAGIC 
# MAGIC Fill in the blanks of the below code block.

# COMMAND ----------

# Answer

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)

dt.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** What is the R-squared on the training set and on the test set? <br>
# MAGIC **Answer= 1.0 and 0.894**

# COMMAND ----------

# Answer
y_train_predicted = dt.predict(X_train)
y_test_predicted = dt.predict(X_test)

print("R2 on training set: ", round(dt.score(X_train, y_train),3))
print("R2 on test set: ", round(dt.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC Congratulations, you built a decision tree that does and excellent job predicting a user's average workout minutes, on both training and test sets! No overfitting or high variance here. There must be a particular feature or features that are highly correlated with the target.
# MAGIC 
# MAGIC Now let's see how a decision tree regression model will perform with a new target: a user's average `vo2`.

# COMMAND ----------

# Answer
X = ht_metrics_pd_df[['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'steps', 'avg_workout_minutes']]
y = ht_metrics_pd_df['avg_vo2']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 

dt = DecisionTreeRegressor(random_state=42)

dt.fit(X_train,y_train)

y_train_predicted = dt.predict(X_train)
y_test_predicted = dt.predict(X_test)

print("R2 on training set: ", round(dt.score(X_train,y_train),3))
print("R2 on test set: ", round(dt.score(X_test,y_test),3))

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like we now have a model that is overfitting to the training set, as the test set accuracy is much lower. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3 
# MAGIC 
# MAGIC In this exercise, you will try some hyperparameter tuning to prune the tree and prevent it from overfitting, thus decreasing the variance and improving performance on the test set.
# MAGIC 
# MAGIC Build three decision tree models with three different `max_depth` options: 3, 5, and 9. Then compare which one performs the best. Remember to set the `random_state` to 42.

# COMMAND ----------

# Answer
dt_depth_3 = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_depth_3.fit(X_train, y_train)

y_train_predicted = dt_depth_3.predict(X_train)
y_test_predicted = dt_depth_3.predict(X_test)

print("R2 on training set: ", round(dt_depth_3.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_depth_3.score(X_test, y_test), 3))

# COMMAND ----------

# Answer
dt_depth_5 = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_depth_5.fit(X_train, y_train)

y_train_predicted = dt_depth_5.predict(X_train)
y_test_predicted = dt_depth_5.predict(X_test)

print("R2 on training set: ", round(dt_depth_5.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_depth_5.score(X_test, y_test), 3))

# COMMAND ----------

# TODO
dt_depth_9 = DecisionTreeRegressor(max_depth=9, random_state=42)
dt_depth_9.fit(X_train, y_train)

y_train_predicted = dt_depth_9.predict(X_train)
y_test_predicted = dt_depth_9.predict(X_test)

print("R2 on training set: ", round(dt_depth_9.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_depth_9.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** Which value for `max_depth` gives the best R-squared score on the test set, and what is the score? <br>
# MAGIC **Answwer== dt_depth_5=0.937**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 4
# MAGIC 
# MAGIC In this exercise, you will practice visualizing the decision tree.
# MAGIC 
# MAGIC Since decision trees are a highly transparent model, we can actually visualize the decisions and splits it made at each step. 
# MAGIC 
# MAGIC Run the following cell to create a visualization of the decision tree with `max_depth` of 3.

# COMMAND ----------

from sklearn import tree
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows = 1, ncols = 1,figsize = (8, 4), dpi=400)

tree.plot_tree(
  dt_depth_3,
  feature_names = ['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'steps', 'avg_workout_minutes'],
  filled = True
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** Which feature did the decision tree choose for most of the splits?<br>
# MAGIC **Answer==average resting heartate**

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** How many samples are in the terminal node at the far left? <br>
# MAGIC **Answer==160**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Coursera Quiz:** What value would the decision tree predict for `avg_vo2` if the `avg_resting_heartrate` is 60 and bmi is 22? <br>
# MAGIC **Answer== 35.769**
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** If a data point meets the criteria in the top row of each node, i.e. a "yes", follow the tree down to the left; if not, follow the tree down to the right.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>