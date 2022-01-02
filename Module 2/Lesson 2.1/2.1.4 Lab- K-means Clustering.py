# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # K-means Clustering Lab
# MAGIC 
# MAGIC **Objective**: *Apply K-means clustering to a dataset to learn more about how the records are related to one another.*
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
# MAGIC 1. `sum_workout_minutes` - the sum of total workout minutes
# MAGIC 1. `sum_steps` - the sum of total steps
# MAGIC 
# MAGIC Fill in the blanks in the below cell to create the `adsda.ht_user_metrics_lab` table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_lab
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-lab" AS (
# MAGIC   SELECT avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS avg_bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          sum(workout_minutes) AS sum_workout_minutes,
# MAGIC          sum(steps) AS sum_steps
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** How many rows and columns are in `adsda.ht_user_metrics_lab`? (3000 rows and 6 columns)

# COMMAND ----------

df = spark.table("adsda.ht_user_metrics_lab").toPandas()
df.shape

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 2
# MAGIC 
# MAGIC In this exercise, you will split your data into an training set and an inference set.
# MAGIC 
# MAGIC Fill in the blanks below to split the data.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> In practice, you should use as much data as possible for your training set. An inference set will usually become available after the training process, rather than being split apart from your training set prior to the training of the model.

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split

train_df, inference_df = train_test_split(df, train_size=0.85, test_size=0.15, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** How many rows are in the training set and the inference set? 

# COMMAND ----------

print(f"{train_df.shape[0]} and {inference_df.shape[0]}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3
# MAGIC 
# MAGIC In this this exercise, you will identify the optimal number of clusters for K-means using the training set.
# MAGIC 
# MAGIC When completing this exercise:
# MAGIC 
# MAGIC * Assess values of *K* from 2-16
# MAGIC * Set the maximum number of iterations to 500
# MAGIC * Use a random state of 1234
# MAGIC 
# MAGIC Fill in the blanks below to compute the distortion for each value of *K*.

# COMMAND ----------

# TODO
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

distortions = []
values_of_k = range(2, 16)

for k in values_of_k:
  k_means = KMeans(n_clusters=k, max_iter=500, random_state=1234)
  k_means.fit(scale(train_df))
  distortion = k_means.score(scale(train_df))
  distortions.append(-distortion)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** Which of the values of *K* has the lowest level of distortion?

# COMMAND ----------

list(zip(distortions, values_of_k)) 

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** According to the elbow method, which of the values of *K* is the optimal number of clusters? between 4 to 6

# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(values_of_k, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 4
# MAGIC 
# MAGIC In this exercise, you will retrain the model with the optimal number of clusters.
# MAGIC 
# MAGIC Fill in the blanks below to retrain the model.

# COMMAND ----------

# TODO
k_means = KMeans(n_clusters=4, max_iter=500, random_state=1234)
k_means.fit(scale(train_df))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: Which of the following datasets should be used when retraining the model with the optimal number of clusters? Select one.
# MAGIC 
# MAGIC * The training set
# MAGIC * The inference set
# MAGIC * The full set
# MAGIC * Data that is not yet available

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 5
# MAGIC 
# MAGIC In this exercise, you will examine the centroids for the clusters.
# MAGIC 
# MAGIC Fill in the blanks below to examine the centroids.

# COMMAND ----------

# TODO
k_means.cluster_centers_

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: In how many dimensions is each centroid?
# MAGIC 
# MAGIC * 2
# MAGIC * 4
# MAGIC * 6
# MAGIC * 16

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 6
# MAGIC 
# MAGIC In this exercise, you will perform inference by placing new rows from the inference set into the existing clusters.
# MAGIC 
# MAGIC Fill in the blanks below to assign rows from the inference set into new clusters.

# COMMAND ----------

# TODO
inference_df_clusters = k_means.predict(scale(inference_df))
clusters_df = inference_df.copy()
clusters_df["cluster"] = inference_df_clusters

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: Which of the clusters has the most rows from the inference set assigned to it?

# COMMAND ----------

clusters_df["cluster"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: Which of the clusters has the rows that take the greatest number of average total steps?

# COMMAND ----------

clusters_df.groupby(["cluster"])[["sum_steps"]].mean()

# COMMAND ----------

# MAGIC %md
# MAGIC Congrats! That concludes our lesson on K-means clustering!
# MAGIC 
# MAGIC Be sure to submit your quiz answers to Coursera, and join us in the next lesson to learn about principal components analysis.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>