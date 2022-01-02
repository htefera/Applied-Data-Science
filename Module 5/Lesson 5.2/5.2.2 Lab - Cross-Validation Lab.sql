-- Databricks notebook source
-- MAGIC 
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Cross-Validation Lab
-- MAGIC 
-- MAGIC **Objective**: *Assess your ability to apply cross-validated hyperparameter tuning to a model.*
-- MAGIC 
-- MAGIC In this lab, you will apply what you've learned in this lesson. When complete, please use the answers to the exercises to answer questions in the following quiz within Coursera.

-- COMMAND ----------

-- MAGIC %run "../../Includes/Classroom-Setup"

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC ## Exercise 1
-- MAGIC 
-- MAGIC In this exercise, you will create an enhanced user-level table to try to better predict whether or not each user takes at least *8,000* steps in a day. For this exercise, assume we only have access to heart rate information.
-- MAGIC 
-- MAGIC Fill in the blanks in the below cell to create the `adsda.ht_user_metrics_cv_lab` table.
-- MAGIC 
-- MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Note that this lab is focused on predicting whether users take 8,000 steps per day rather than 10,000 steps per day.

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC -- TODO
-- MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_cv_lab
-- MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-cv-lab" AS (
-- MAGIC   SELECT min(resting_heartrate) AS min_resting_heartrate,
-- MAGIC          avg(resting_heartrate) AS avg_resting_heartrate,
-- MAGIC          max(resting_heartrate) AS max_resting_heartrate,
-- MAGIC          max(resting_heartrate) - min(resting_heartrate) AS resting_heartrate_change,
-- MAGIC          min(active_heartrate) AS min_active_heartrate,
-- MAGIC          avg(active_heartrate) AS avg_active_heartrate,
-- MAGIC          max(active_heartrate) AS max_active_heartrate,
-- MAGIC          max(active_heartrate) - min(active_heartrate) AS active_heartrate_change,
-- MAGIC          CASE WHEN avg(steps)>8000 THEN 1 ELSE 0 END AS steps_8000
-- MAGIC   FROM adsda.ht_daily_metrics
-- MAGIC   GROUP BY device_id)

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC **Coursera Quiz:** How many users in `adsda.ht_user_metrics_cv_lab` take, on average, 8,000 steps per day?<br>
-- MAGIC **Answer == 1924**
-- MAGIC 
-- MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer back to the previous lab for guidance on how to answer this question.

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC -- Answer
-- MAGIC SELECT steps_8000, count(*) FROM adsda.ht_user_metrics_cv_lab GROUP BY steps_8000

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC ## Exercise 2
-- MAGIC 
-- MAGIC In this exercise, you will split your data into a cross-validation set (`cross_val_df`) and test set (`test_df`).
-- MAGIC 
-- MAGIC Fill in the blanks below to split your data.
-- MAGIC 
-- MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer to the previous demo for guidance.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Answer
-- MAGIC from sklearn.model_selection import train_test_split
-- MAGIC 
-- MAGIC ht_user_metrics_pd_df = spark.table("adsda.ht_user_metrics_cv_lab").toPandas()
-- MAGIC 
-- MAGIC cross_val_df, test_df = train_test_split(ht_user_metrics_pd_df, train_size=0.80, test_size=0.20, random_state=42)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Coursera Quiz:** How many rows are in the `cross_val_df` DataFrame?<br>
-- MAGIC **Answer==2400**
-- MAGIC 
-- MAGIC Fill in the blanks below to answer the question.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # TODO
-- MAGIC cross_val_df.shape

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Exercise 3
-- MAGIC 
-- MAGIC In this exercise, you will prepare your random forest classifier.
-- MAGIC 
-- MAGIC Fill in the blanks below to complete the task.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Answer
-- MAGIC from sklearn.ensemble import RandomForestClassifier
-- MAGIC 
-- MAGIC rfc = RandomForestClassifier(random_state=42)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Exercise 4
-- MAGIC 
-- MAGIC In this exercise, you will create a hyperparameter grid to use during the grid search process.
-- MAGIC 
-- MAGIC Use the following hyperparameter values:
-- MAGIC 
-- MAGIC 1. `max_depth`: 5, 8, 20
-- MAGIC 1. `n_estimators`: 25, 50, 100
-- MAGIC 1. `min_samples_split`: 2, 4
-- MAGIC 1. `max_features`: 3, 4
-- MAGIC 1. `max_samples`: 0.6, 0.8
-- MAGIC 
-- MAGIC Fill in the blanks below to create the grid.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Answer
-- MAGIC parameter_grid = {
-- MAGIC   "max_depth": [5, 8, 20],
-- MAGIC   "n_estimators": [25, 50, 100],
-- MAGIC   "min_samples_split": [ 2, 4],
-- MAGIC   "max_features": [3, 4],
-- MAGIC   "max_samples": [ 0.6, 0.8]
-- MAGIC }

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC **Coursera Quiz**: How many total unique combinations of hyperparameters are there in `parameter_grid`?<br>
-- MAGIC **Answer==72**
-- MAGIC 
-- MAGIC Use the below empty cell to determine the answer to the above question.
-- MAGIC 
-- MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer to the previous lesson's lab for guidance.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Answer
-- MAGIC len(parameter_grid["max_depth"]) * len(parameter_grid["n_estimators"]) * len(parameter_grid["min_samples_split"]) * len(parameter_grid["max_features"]) * len(parameter_grid["max_samples"])

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Exercise 5
-- MAGIC 
-- MAGIC In this exercise, you will create the cross-validated grid-search object that you will use to optimize your hyperparameter values while using cross-validation.
-- MAGIC 
-- MAGIC Fill in the blanks below to create the object.
-- MAGIC 
-- MAGIC :NOTE: Please use 4-fold cross-validation.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # ANSWER
-- MAGIC from sklearn.model_selection import GridSearchCV
-- MAGIC 
-- MAGIC grid_search = GridSearchCV(estimator=rfc, cv=4, param_grid=parameter_grid)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Exercise 6
-- MAGIC 
-- MAGIC In this exercise, you will fit the grid search process.
-- MAGIC 
-- MAGIC Fill in the blanks below to perform the grid search process.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # ANSWER
-- MAGIC grid_search.fit(cross_val_df.drop("steps_8000", axis=1), cross_val_df["steps_8000"])

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC **Coursera Quiz**: How many unique models are being trained by the cross-validated grid search process?<br>
-- MAGIC **Answer==289**
-- MAGIC 
-- MAGIC * 4
-- MAGIC * 289
-- MAGIC * 21
-- MAGIC * 288
-- MAGIC 
-- MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Consider the number of unique feature combinations, the number of cross-validation folds and the final retraining of the model on the entire cross-validation set.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC ## Exercise 7
-- MAGIC 
-- MAGIC In this exercise, you will return a Pandas DataFrame of the `grid_search` results.
-- MAGIC 
-- MAGIC Fill in the blanks below to return the DataFrame.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Answer
-- MAGIC import pandas as pd
-- MAGIC 
-- MAGIC pd.DataFrame(grid_search.cv_results_)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC ## Exercise 8
-- MAGIC 
-- MAGIC In this exercise, you will identify the optimal hyperparameter values.
-- MAGIC 
-- MAGIC Fill in the blanks below to indentify the optimal hyperparameter values.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Answer
-- MAGIC grid_search.best_params_

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC **Coursera Quiz:** What is the optimal hyperparameter value for `max_depth` according to the cross-validated grid search process?<br>
-- MAGIC **Answer==5**

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC ## Exercise 9
-- MAGIC 
-- MAGIC In this exercise, you will identify the test accuracy achieved by the final, refit model.
-- MAGIC 
-- MAGIC Fill in the blanks below to identify the test accuracy.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # ANSWER
-- MAGIC from sklearn.metrics import accuracy_score
-- MAGIC 
-- MAGIC accuracy_score(
-- MAGIC   test_df["steps_8000"], 
-- MAGIC   grid_search.predict(test_df.drop("steps_8000", axis=1))
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Coursera Quiz:** What is the test set accuracy?<br>
-- MAGIC **Answer==0.895**

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Congratulations! That concludes our lesson on cross-validated hyperparameter optimization and our course!
-- MAGIC 
-- MAGIC Be sure to submit your quiz answers to Coursera to fully complete the course!

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>