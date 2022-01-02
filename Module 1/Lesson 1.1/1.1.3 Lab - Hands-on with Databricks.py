# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Hands-on with Databricks
# MAGIC 
# MAGIC **Objective**: *Familiarize yourself with the Databricks platform, the use of notebooks, and basic SQL operations in Databricks.*
# MAGIC 
# MAGIC In this lab, you will complete a series of exercises to familiarize yourself with the content covered in Lesson 0.1.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 1
# MAGIC 
# MAGIC In order to execute code with Databricks, you need to have your notebook attached to an active cluster. 
# MAGIC 
# MAGIC Ensure that:
# MAGIC 1. You have created a cluster following the walkthrough of the video in this lesson.
# MAGIC 2. Your cluster's Databricks Runtime Version is 7.2 ML.
# MAGIC 3. Your cluster is active and running.
# MAGIC 4. This notebook is attached to your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 2
# MAGIC 
# MAGIC The fundamental piece of a Databricks notebook is the command cell. We use command cells to write and run our code. 
# MAGIC 
# MAGIC Complete the following:
# MAGIC 1. Insert a command cell beneath this one.
# MAGIC 2. Write `1 + 1` in the command cell.
# MAGIC 3. Run the command cell.
# MAGIC 4. Verify that the output of the executed code is `2`.

# COMMAND ----------

# ANSWER
1+1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3
# MAGIC 
# MAGIC Command cells can also be used to add comments using a lightweight markup language named *markdown*. (That's how these command cells are written).
# MAGIC 
# MAGIC Complete the following:
# MAGIC 
# MAGIC 1. Double-click on this command cell.
# MAGIC 2. Notice the *magic command* at the top of the command cell that enables the use of markdown.
# MAGIC 3. Insert a command cell beneath this one and add the magic command to the first line.
# MAGIC 4. Write `THE MAGIC COMMAND FOR MARKDOWN IS _____` with the magic command filling the blank.

# COMMAND ----------

# MAGIC %md
# MAGIC # ANSWER
# MAGIC THE MAGIC COMMAND FOR MARKDOWN IS %md.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 4
# MAGIC 
# MAGIC Throughout this course, we will be using a setup file in each of our notebooks that connects Databricks to our data.
# MAGIC 
# MAGIC Complete the following:
# MAGIC 
# MAGIC 1. Run the below command cell to execute the setup file.
# MAGIC 2. Insert a SQL command cell beneath the command cell containg the setup file.
# MAGIC 3. Query all of the data in the table **`adsda.ht_users`** using the query `SELECT * FROM adsda.ht_users`.
# MAGIC 4. Examine the displayed table to learn about its columns and rows.

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM adsda.ht_users

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 5
# MAGIC 
# MAGIC Throughout this course, we will need to manipulate data and save it as new tables using Delta, just as we did in the video during the lesson.
# MAGIC 
# MAGIC Complete the following:
# MAGIC 
# MAGIC 1. Insert a new SQL command cell beneath this one.
# MAGIC 2. Write a SQL query to compute the average value of `resting_heartrate`, `active_heartrate`, `bmi`, `vo2`, and `steps` from the **adsda.ht_users** table. Group the aggregation by the `device_id` column.
# MAGIC 3. Use the SQL query to create a new Delta table named **adsda.ht_device_metrics** and store the data in the following location: `"/adsda/ht-device-metrics"`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ANSWER
# MAGIC CREATE OR REPLACE TABLE adsda.ht_device_metrics
# MAGIC USING DELTA LOCATION "/adsda/ht-device-metrics"
# MAGIC AS (
# MAGIC   SELECT
# MAGIC     device_id,
# MAGIC     avg(resting_heartrate) AS resting_heartrate,
# MAGIC     avg(active_heartrate) AS active_heartrate,
# MAGIC     avg(bmi) AS bmi,
# MAGIC     avg(vo2) AS vo2,
# MAGIC     avg(steps) AS steps
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id)

# COMMAND ----------

# MAGIC %md
# MAGIC Query the delta table

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from ht_device_metrics 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 6
# MAGIC 
# MAGIC Throughout this course, we will need to convert Delta tables into Pandas DataFrames and build models using `sklearn`.
# MAGIC 
# MAGIC Complete the following:
# MAGIC 
# MAGIC 1. Create a Pandas DataFrame `pd_df` from the **adsda.ht_device_metrics** table which was created in the previous exercise. Include all columns and rows.
# MAGIC 2. Split `pd_df` into training and test sets, where 80 percent of the records are in the training set. Be sure to set a random state (seed) for reproducibility.
# MAGIC 3. Train a linear regression on the training data.
# MAGIC 4. Evaluate the root mean squared error (RMSE) on the test set.
# MAGIC 
# MAGIC Fill in the blanks in the below code block to complete the tasks.

# COMMAND ----------

# MAGIC %python
# MAGIC # TODO
# MAGIC from sklearn.linear_model import LinearRegression
# MAGIC from sklearn.metrics import mean_squared_error
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC 
# MAGIC # Create a Pandas DataFrame pd_df from the adsda.ht_device_metrics table
# MAGIC pd_df = spark.sql("SELECT * FROM adsda.ht_device_metrics").toPandas()
# MAGIC 
# MAGIC # Split pd_df into training and test sets
# MAGIC X = pd_df.drop(["device_id", "steps"], axis = 1)
# MAGIC y = pd_df[["steps"]] 
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# MAGIC 
# MAGIC # Train a linear regression on the training data
# MAGIC model = LinearRegression()
# MAGIC model.fit(X_train,y_train)
# MAGIC 
# MAGIC # Evaluate the root mean squared error on the test set
# MAGIC test_predictions = model.predict(X_test)
# MAGIC print(f"Root mean squared error: {mean_squared_error(y_test, test_predictions, squared = False)}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Great job! You've completed the first lesson of the Applied Data Science for Data Analysts course.
# MAGIC 
# MAGIC Please proceed to the next lesson to begin Module 1: Applied Unsupervised Learning.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>