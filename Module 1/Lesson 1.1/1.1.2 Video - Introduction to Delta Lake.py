# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Introduction to Delta Lake
# MAGIC 
# MAGIC This notebook complements the lesson video "Introduction to Delta Lake". Please use the notebook to follow along with the video.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC 
# MAGIC Please run the below cell to attach the course data to Databricks.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Video Content
# MAGIC 
# MAGIC Use the below cells to follow long with the video.

# COMMAND ----------

# MAGIC %md
# MAGIC The below command cell will display the first 1000 rows of data in the **`ht_users`** table in the **`adsda`** database.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT distinct lifestyle  FROM adsda.ht_users

# COMMAND ----------

# MAGIC %md
# MAGIC The below command cell will create a table containing only the athletes from **`dsfd.ht_users`**. It will name the new table **`ht_users_athletes`** and put it in the **`dsfda`** database. It will store the data for the table at `/dsfda/ht-users-athletes`.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_users_athletes
# MAGIC USING DELTA LOCATION '/adsda/ht-users-athletes'
# MAGIC AS (
# MAGIC   SELECT *
# MAGIC   FROM adsda.ht_users
# MAGIC   WHERE lifestyle = 'Athlete'
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC We can also run SQL queries on tables and return the results as a Spark DataFrame.

# COMMAND ----------

ht_users_non_athletes_df = spark.sql("SELECT * FROM adsda.ht_users WHERE lifestyle != 'Athlete'")

# COMMAND ----------

# MAGIC %md
# MAGIC In order to view the data in DataFrames, we need to run the `display()` function.

# COMMAND ----------

display(ht_users_non_athletes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Sometimes we might want to create tables from DataFrames. First, we need to write out the DataFrame in the Delta format.

# COMMAND ----------

ht_users_non_athletes_df.write.format("delta").mode("overwrite").save("/adsda/ht-users-non-athletes")

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we registed the table with location associated with DataFrame we just saved.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE adsda.ht_users_non_athletes
# MAGIC USING DELTA LOCATION '/adsda/ht-users-non-athletes'

# COMMAND ----------

# MAGIC %md
# MAGIC The below command cell confirms that we saved our DataFrame as a Delta table.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_users_non_athletes

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>