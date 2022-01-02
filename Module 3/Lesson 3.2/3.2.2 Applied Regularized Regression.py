# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Applied Regularized Regression
# MAGIC 
# MAGIC **Objective**: *Demonstrate the use of LASSO regression for feature selection*
# MAGIC 
# MAGIC In this demo, we will complete a series of exercises to show how to use a LASSO model for feature selection.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC ### Aggregate our user-level table
# MAGIC 
# MAGIC Remember that one of our project objectives is to predict a customer's `max_BMI` based on their recorded metrics. Therefore, we are interested in a user-level clustering. To prepare the dataset to do this, we'll aggregate our **`adsda.ht_user_metrics`** table at the user level.

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
# MAGIC          min(bmi) AS min_bmi,
# MAGIC          avg(bmi) AS avg_bmi,
# MAGIC          max(bmi) AS max_bmi,
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
# MAGIC          max(bmi) - min(bmi) AS bmi_change
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics_pca

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Convert Spark DataFrame to Pandas
# MAGIC 
# MAGIC We will use this Pandas DataFrame in this demo.

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_pca").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC View the data

# COMMAND ----------

ht_lifestyle_pd_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fitting a Linear Regression Model and Examining Coefficients and P Values
# MAGIC 
# MAGIC This process has a few steps so we'll string everything together and explain step by step.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step #1 - Create feature matrix and target
# MAGIC Now we need to create our X and y from our features and target. Recall that our target is the thing we are trying to predict, BMI, given some features about an observation.

# COMMAND ----------

X = ht_lifestyle_pd_df.drop(['max_bmi'], axis=1)
y = ht_lifestyle_pd_df['max_bmi']

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step #2 - Instantiate and Fit our model
# MAGIC Import LASSO from sklearn.

# COMMAND ----------

from sklearn.linear_model import Lasso

# COMMAND ----------

# MAGIC %md
# MAGIC Instantiate and fit

# COMMAND ----------

lasso_reg = Lasso(alpha=10)
lasso_reg.fit(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step #3 - Examine results
# MAGIC Once our model is fit, we have helper methods and attributes available.

# COMMAND ----------

lasso_reg.score(X, y) # score means the accuracy of the model

# COMMAND ----------

import pandas as pd
pd.DataFrame(list(zip(lasso_reg.coef_, X.columns)), columns=['coefficient', 'feature_name']).sort_values('coefficient', ascending=False)

# COMMAND ----------

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
pd.DataFrame(list(zip(lr.coef_, X.columns)), columns=['coefficient', 'feature_name']).sort_values('coefficient', ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nicely done!

# COMMAND ----------

lr.score(X,y)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
