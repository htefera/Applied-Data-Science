# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualizing Data
# MAGIC 
# MAGIC **Objective**: *Demonstrate how to explore your data in Databricks using its built-in visualization capabilities.*
# MAGIC 
# MAGIC In this demo, we will complete a series of exercises to practice exploring our data and utilize Databricks visualization.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exploring Data
# MAGIC 
# MAGIC ### Import and Display
# MAGIC 
# MAGIC First, it's beneficial import and display your data to ensure you have access to it.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_users

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_daily_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Record-level
# MAGIC 
# MAGIC Next, remember that we always want to determine the record-level of our data.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- user level(devi_id column)
# MAGIC SELECT * FROM adsda.ht_users LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- user day level-device -id and day column 
# MAGIC SELECT * FROM adsda.ht_daily_metrics LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC It seems like our **adsda.ht_daily_metrics** table is at the user-day level, but our objective is just at the user level.
# MAGIC 
# MAGIC Let's aggregate this data across days to be at the user level.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics" AS (
# MAGIC   SELECT device_id,
# MAGIC          avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS avg_bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          max(lifestyle) AS lifestyle,
# MAGIC          avg(steps) AS steps
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id)

# COMMAND ----------

# MAGIC %sql
# MAGIC --check if the data is at a record level 
# MAGIC SELECT * FROM adsda.ht_user_metrics LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shape
# MAGIC 
# MAGIC In the lecture, we talked about the benefits of understanding how many rows and columns are in your data.
# MAGIC 
# MAGIC There are a few ways to do this.
# MAGIC 
# MAGIC #### SQL
# MAGIC 
# MAGIC Using SQL, we can do this in a few steps.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Python
# MAGIC 
# MAGIC And we can do it with one step in Python using two different methods.
# MAGIC 
# MAGIC *Small Data*

# COMMAND ----------

spark.sql("SELECT * FROM adsda.ht_user_metrics").toPandas().shape

# COMMAND ----------

# MAGIC %md
# MAGIC *Big Data*

# COMMAND ----------

ht_user_metrics_df = spark.sql("SELECT * FROM adsda.ht_user_metrics")
(ht_user_metrics_df.count(), len(ht_user_metrics_df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Columns
# MAGIC 
# MAGIC So once we know that we have 8 columns, we might want to look at what those columns contain and verify that we have columns that we can use to meet our project's objective.
# MAGIC 
# MAGIC One way to do this is by just viewing the table:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC But it can also be helpful to view the schema of the data to verify the data types of the columns.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary Statistics
# MAGIC 
# MAGIC Remember we talked about computing summary statistics on columns, as well.
# MAGIC 
# MAGIC We could do this by manually computing everything we're interested in using SQL, but the PySpark DataFrame has useful methods to help do this pretty easily.

# COMMAND ----------

display(spark.sql("SELECT * FROM adsda.ht_user_metrics").summary())

# COMMAND ----------

# MAGIC %md
# MAGIC **Question**: Do you think that computing summary statistics gives us a complete view of our data?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization
# MAGIC 
# MAGIC ### Anscombe's Quartet
# MAGIC 
# MAGIC [Ancsombe's Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) is a collection of four unique data sets, each having columns X and Y.
# MAGIC 
# MAGIC Across each data set, the following summary statistics are identical:
# MAGIC * Mean of X
# MAGIC * Mean of Y
# MAGIC * Standard deviation of X
# MAGIC * Standard deviation of Y
# MAGIC * Correlation of X and Y
# MAGIC * Linear regression line
# MAGIC * Coefficient of determination (R2) of the linear regression line
# MAGIC 
# MAGIC After computing the summary statistics, it would appear that each of these datasets are identical.
# MAGIC 
# MAGIC **But they are very different**, and the only way we can see that is by looking at each individual point.
# MAGIC 
# MAGIC ![Ancombe's Quartet](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/800px-Anscombe%27s_quartet_3.svg.png)
# MAGIC 
# MAGIC Each of these datasets requires a different approach to data cleansing and algorithm selection, and we wouldn't have known that without visualizing the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks Visualization
# MAGIC 
# MAGIC In order to visualize our own data, we can utilize Databricks' built-in visualization capabilities.
# MAGIC 
# MAGIC #### Numeric variable distribution

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC #### Numeric variable distribution by category

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM adsda.ht_user_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC #### Numeric variables scatterplot

# COMMAND ----------

# MAGIC %sql
# MAGIC -- see the correlation between average BMI vs average active heart rate
# MAGIC SELECT * FROM adsda.ht_user_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Limits of Data Visualization
# MAGIC 
# MAGIC While data visualization does provide us with benefits in exploring our data, it also has its limitations and things to be aware of:
# MAGIC 
# MAGIC 1. Data visualization can bias our understanding of our data.
# MAGIC 2. It's difficult visualize data in more than two dimensions.
# MAGIC 3. ....
# MAGIC 
# MAGIC In the rest of this lesson, we'll look at how we can use clustering to help us learn from data with a more than two dimensions.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>