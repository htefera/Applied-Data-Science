# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Applied K-means Clustering
# MAGIC 
# MAGIC **Objective**: *Demonstrate how to perform K-means clustering using Python and sklearn.*
# MAGIC 
# MAGIC In this demo, we will complete a series of exercises to practice performing K-means clustering analyses.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC ### Aggregate our user-level table
# MAGIC 
# MAGIC Remember we are interested in a user-level clustering based on our project objective. As a result, we'll recreate our **`adsda.ht_user_metrics`** table from the previous demo.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> We are removing the **`lifestyle`** and **`device_id`** columns from this analysis because K-means clustering requires all feature variables to be numeric.

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
# MAGIC And we can visualize the result.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics LIMIT 10

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Split data
# MAGIC 
# MAGIC Next, we are going to split our table into a training set and inference set.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> In unsupervised learning, we do not perform label-based evaluation like we do in supervised learning. We are going to use the inference set as an example for assigning rows that were not a part of the training process to clusters.

# COMMAND ----------

from sklearn.model_selection import train_test_split

ht_user_metrics_pd_df = spark.table("adsda.ht_user_metrics").toPandas()

train_df, inference_df = train_test_split(ht_user_metrics_pd_df, train_size=0.9, test_size=0.1, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the resulting DataFrames have the same number of columns, but they have a different number of rows.

# COMMAND ----------

train_df.shape

# COMMAND ----------

inference_df.shape

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## K-means
# MAGIC 
# MAGIC ### Training
# MAGIC Now, we can apply the K-means algorithm to our **`train_df`** DataFrame.
# MAGIC 
# MAGIC Remember that in order to do this, we need to manually specify *K* ahead of the training process to the `num_clusters` parameter.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> We need to scale our feature variables because the K-means algorithm treats all features as if they're on the same scale. We'll go into more detail on this with more advanced tools in the next module.

# COMMAND ----------

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

k_means = KMeans(n_clusters=4, random_state=42)
k_means.fit(scale(train_df))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Other Parameters
# MAGIC 
# MAGIC There are plenty of other parameters to the K-means process, including:
# MAGIC 
# MAGIC * `init` - how the initial centroids are determined
# MAGIC * `max_iter` - how many iterations of the algorithm (i.e. how many times the centroids are reset)
# MAGIC * `random_state` - Determines random number generation for centroid initialization. Use an int to make the randomness deterministic

# COMMAND ----------

# MAGIC %md
# MAGIC #### Getting Centroids
# MAGIC 
# MAGIC Once the model has been fit, the centroid locations can be extracted using the `cluster_centers_` attribute. The last value in the result set shows the centroid of the cluster formed

# COMMAND ----------

k_means.cluster_centers_

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Note that each of these array elements corresponds to a point, and the nested elements are the locations of each centroid for the various features used.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Inference
# MAGIC 
# MAGIC Once we've trained our K-means model, we can use the final cluster centroids to place new, unseen rows into clusters, as well.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> We are scaling our inference set, too. There are more advanced tools that can do this in an unbiased way, and we'll go over them in the next module.

# COMMAND ----------

inference_df_clusters = k_means.predict(scale(inference_df))

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that the output is a numpy array of the same length as **`inference_df`** DataFrame. This is because we returned a cluster for each one of our rows.

# COMMAND ----------

type(inference_df_clusters)

# COMMAND ----------

len(inference_df_clusters)

# COMMAND ----------

# MAGIC %md
# MAGIC Since we have a numpy array of the same length, we can bind the array with **`inference_df`** into a new DataFrame.

# COMMAND ----------

clusters_df = inference_df.copy()
clusters_df["cluster"] = inference_df_clusters

# COMMAND ----------

# MAGIC %md
# MAGIC So we can easily view the cluster of each row.

# COMMAND ----------

display(clusters_df)

# COMMAND ----------

display(clusters_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Through the rest of this lesson, we'll look at optimizing the use of the K-means algorithm.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>