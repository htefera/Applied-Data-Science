# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # One-Hot Encoding
# MAGIC 
# MAGIC **Objective**: *Demonstrate using one-hot encoding to represent categorical features numerically*
# MAGIC 
# MAGIC In this demo, we will complete a series of exercises to show how to take a categorical feature and transform it into a numerical one for consumption by a machine learning model.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC ### Aggregate our user-level table
# MAGIC 
# MAGIC Remember that one of our project objectives is to predict a customer's `lifestyle` based on their recorded metrics. Therefore, we are interested in a user-level clustering. To prepare the dataset to do this, we'll aggregate our **`adsda.ht_user_metrics`** table at the user level.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_lifestyle
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-lifestyle" AS (
# MAGIC   SELECT avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          avg(steps) AS steps,
# MAGIC          first(lifestyle) AS lifestyle
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

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

# MAGIC %md
# MAGIC View the data

# COMMAND ----------

ht_lifestyle_pd_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Looks good. Now, let's get a look at the datatypes of our features. There are several methods to achieve this. This is similar to our schema in SQL.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 1
# MAGIC Use a built-in attribute on the DataFrame

# COMMAND ----------

ht_lifestyle_pd_df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Method 2
# MAGIC Use a built-in method

# COMMAND ----------

ht_lifestyle_pd_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## One Hot Encoding Values
# MAGIC 
# MAGIC This process has a few steps so we'll string everything together and explain step by step.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1.
# MAGIC Pandas has a built-in method to one-hot encode called `get_dummies()`. It can optionally be passed with a few parameters, which we'll explain in 1(a) and 1(b)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1(a) One Column at a time.
# MAGIC We'll run this as a standalone command first to see what the output looks like. In practice, this next cell is superfluous 

# COMMAND ----------

import pandas as pd
pd.get_dummies(ht_lifestyle_pd_df['lifestyle'])

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the object we receive back is a new DataFrame. We'll save this out to a variable

# COMMAND ----------

lifestyle_dummies_df = pd.get_dummies(ht_lifestyle_pd_df['lifestyle'])

# COMMAND ----------

# MAGIC %md
# MAGIC Then we'll join this back onto our original dataframe

# COMMAND ----------

ht_lifestyle_pd_df = ht_lifestyle_pd_df.join(lifestyle_dummies_df)

# COMMAND ----------

ht_lifestyle_pd_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we'll drop the original `lifestyle` column because it is now uncessary.

# COMMAND ----------

ht_lifestyle_pd_df.drop('lifestyle', axis=1, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1(b) All at once.
# MAGIC If we wanted to do this multiple columns at once, or if we wanted to skip the extra code, we could use `.get_dummies()` in the following way.

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

  ht_lifestyle_pd_df = pd.get_dummies(ht_lifestyle_pd_df, prefix='ohe', columns=['lifestyle'])

# COMMAND ----------

ht_lifestyle_pd_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nicely done!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>