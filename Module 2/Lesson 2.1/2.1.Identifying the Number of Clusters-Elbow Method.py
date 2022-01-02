# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Identifying the Number of Clusters
# MAGIC 
# MAGIC **Objective**: *Demonstrate the use of the elbow method to determine the optimal number of clusters for K-means clustering.*
# MAGIC 
# MAGIC In this demo, we will determine the optimal number of clusters for a clustering problem.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC Again, we are preparing our feature table like we did in the previous demo. We are getting our user-day level table to be at the user-level by aggregating across days.

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
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC And we want this table as a Pandas DataFrame.

# COMMAND ----------

df = spark.table("adsda.ht_user_metrics").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimizing the Number of Clusters
# MAGIC 
# MAGIC ### Computing the Distortions
# MAGIC 
# MAGIC Next, we will identify the optimal number of clusters for our DataFrame `df`. 
# MAGIC 
# MAGIC Recall that we want to train a K-means clustering for a series of values of *K*, and we want to compare the *distortion* or *Mean Squared Error* of each of the resulting clusters.
# MAGIC As the number of clusters increase the *distortion* decreases, but low *distortion* values does not mean a good cluster
# MAGIC We can begin by importing the necessary libraries.

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Next, we will set up a few Python objects that we'll use:
# MAGIC 
# MAGIC * An empty list into which we'll put the distortion values
# MAGIC * A generator of values from 2 to 10 – these are the values of K we will test

# COMMAND ----------

distortions = []
values_of_k = range(2, 10)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC And finally, we'll loop through our `values_of_k` to create a clustering for each value of *K* and compute the distortion for each of those clusterings.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Because the `score` method returns the negative, we are negating that value to return the positive value

# COMMAND ----------

for k in values_of_k:
  k_means = KMeans(n_clusters=k, random_state=42)
  k_means.fit(scale(df))
  distortion = k_means.score(scale(df))
  distortions.append(-distortion)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determining the Optimal Value of K
# MAGIC 
# MAGIC As you can see below, we've calculated the distortion for each value of K from 2 to 10.

# COMMAND ----------

distortions

# COMMAND ----------

# MAGIC %md
# MAGIC We can plot this data using Python's plotting tools to visually determine our "elbow" point.

# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(values_of_k, distortions, 'bx-') 
plt.xlabel('Number of clusters') 
plt.ylabel('Distortion') 
plt.show()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC Based on the above plot, our elbow seems to fall around values 4 or 5. **This means that our optimal number of clusters is 4 or 5**.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Sometimes this elbow point will clearly be a specific value, and other times you'll need to pick between a couple of values of K.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retraining the K-means Model
# MAGIC 
# MAGIC Once we've identified the optimal number of clusters, we want to be sure to recreate our K-means model with that value of *K*. This is because we've overwritten our previous model during the distortion computing process.

# COMMAND ----------

k_means = KMeans(n_clusters=4, random_state=42)
k_means.fit(scale(df))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>