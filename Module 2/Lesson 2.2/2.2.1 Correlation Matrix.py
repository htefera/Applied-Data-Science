# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Correlation Matrix
# MAGIC 
# MAGIC **Objective**: *Demonstrate the development of a correlation matrix.*
# MAGIC 
# MAGIC In this demo, we will complete a series of exercises to determine how features are correlated.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC Just as with the previous lesson, we'll recreate our **`adsda.ht_user_metrics`** table. As a reminder, this table is at the user-level.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics" AS (
# MAGIC   SELECT avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS avg_bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          avg(steps) AS steps
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id)

# COMMAND ----------

# MAGIC %md
# MAGIC And again, we can visualize the result as a reminder of our table's structure.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation
# MAGIC 
# MAGIC ### Two-feature Correlation
# MAGIC As we mentioned before, some features are correlated with one another.
# MAGIC 
# MAGIC If we want to see how features are correlated with one another, we can use the `corr` SQL function.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT corr(avg_resting_heartrate, avg_active_heartrate) FROM adsda.ht_user_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC This is useful, but there's a major limitation: we can only calculate the Pearson correlation coefficient for two features at once.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Multi-feature Correlation
# MAGIC 
# MAGIC When we want to view how all of our feature variables are correlated with one another, we can create something called a correlation matrix.
# MAGIC 
# MAGIC There's a method in the Pandas DataFrame called `corr` that makes this very easy.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This functionality also exists in Spark DataFrames. Spark DataFrames are outside the scope of this class, but they can be really helpful when your data is too large to fit on the driver node.

# COMMAND ----------

df = spark.table("adsda.ht_user_metrics").toPandas()
df.corr()

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the above correlation matrix, we can gain a lot of information:
# MAGIC 
# MAGIC 1. You'll notice that each feature variable is perfectly correlated with itself.
# MAGIC 1. Active heartrate and resting heartrate are highly correlated.
# MAGIC 1. Steps has a strong negative correlation with heartrates.
# MAGIC 
# MAGIC **Question:** What's the strongest correlation (positive or negative) that you can find?

# COMMAND ----------

# MAGIC %md
# MAGIC Through the rest of this lesson, we'll look at how to better understand these feature relationships using principal components analysis.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>