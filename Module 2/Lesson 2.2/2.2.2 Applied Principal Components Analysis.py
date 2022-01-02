# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Applied Principal Components Analysis
# MAGIC 
# MAGIC **Objective**: *Demonstrate the use of Principal Components Analysis on a dataset.*
# MAGIC 
# MAGIC In this demo, we will complete a series of exercises to perform PCA on a dataset and interpret the results.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC In this demo, we're going to prepare our data to a user-level table again, but we're going to add some more aggregations and some interacted features in the process. This will help us see the impact of reducing our feature space using PCA.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *FROM adsda.ht_daily_metrics LIMIT 10

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

# MAGIC %md
# MAGIC If we display this table, we'll see there are a lot more features in our table than we had in previous demos.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics_pca LIMIT 10

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## PCA
# MAGIC 
# MAGIC ### Training Process
# MAGIC When we perform PCA in Python, we again want our data in a Pandas DataFrame.
# MAGIC 
# MAGIC Then, we can take advantage of `sklearn`'s `PCA` class to easily perform the reduction.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> Rememeber that we need to scale our features for PCA.

# COMMAND ----------

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

df = spark.table("adsda.ht_user_metrics_pca").toPandas()
pca = PCA(random_state=42)
pca.fit(scale(df))

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that we didn't set very many hyperparameters – options that we can use to control the training process.
# MAGIC 
# MAGIC As a result, a lot of the defaults are set, including the number of components:

# COMMAND ----------

pca.n_components_

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This is equal to the number of features in our DataFrame, but remember: we likely only need to use a few.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variance Explained
# MAGIC 
# MAGIC Remember that PCA attempts to explain as much of the variance in the input features with as few of components as possible.
# MAGIC 
# MAGIC To see how effective this was, we can look at the percent of variance explained by each of our components.

# COMMAND ----------

pca.explained_variance_ratio_

# COMMAND ----------

# MAGIC %md
# MAGIC While getting these values in an array can be helpful programmatically, it's helpful to visualize them directly.

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

plt.bar(range(1, 21), pca.explained_variance_ratio_) 
plt.xlabel('Component') 
plt.xticks(range(1, 21))
plt.ylabel('Percent of variance explained')
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, step=0.1))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, the initial few components capture the majority of the variance in our 20 original features.
# MAGIC 
# MAGIC To see just how much, it can be helpful to plot this as the cumulative sum of variance explained.

# COMMAND ----------

plt.plot(range(1, 21), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Component') 
plt.xticks(range(1, 21))
plt.ylabel('Percent of cumulative variance explained')
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, step=0.1))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can now easily see that the first three components account for over 80 percent of the variation in our data.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC The above graphs are helpful but can be hard to visualize. Let's determine this programmatically.
# MAGIC * How many components does it take to explain 90 percent of the variation in the original feature set? 5 components

# COMMAND ----------

for component in list(zip(range(1, 25), np.cumsum(pca.explained_variance_ratio_))):
  if component[1] >= 0.9:
    print(component)
    break

# COMMAND ----------

# MAGIC %md 
# MAGIC Through the rest of this lesson, we'll look at how we can use the results of our PCA process on real-world data science projects.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>