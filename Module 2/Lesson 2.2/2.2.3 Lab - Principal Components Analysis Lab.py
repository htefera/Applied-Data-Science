# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Principal Components Analysis Lab
# MAGIC 
# MAGIC **Objective**: *Apply PCA to a dataset to learn more about how the features are related to one another.*
# MAGIC 
# MAGIC In this lab, you will apply what you've learned in this lesson. When complete, please use the answers to the exercises to answer questions in the following quiz within Coursera.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Exercise 1
# MAGIC 
# MAGIC In this exercise, you will create a user-level table with the following additional columns:
# MAGIC 
# MAGIC 1. `steps_change` – the difference between the maximum steps and the minimum steps
# MAGIC 1. `workout_minutes_change` - the difference between the maximum workout minutes and the minimum workout minutes
# MAGIC 1. `var_workout_minutes` – the variance of the workout minutes
# MAGIC 1. `var_steps` - the population variance of the steps
# MAGIC 
# MAGIC Fill in the blanks in the below cell to create the `adsda.ht_user_metrics_pca_lab` table.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_pca_lab
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-pca-lab" AS (
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
# MAGIC          max(bmi) - min(bmi) AS bmi_change,
# MAGIC          max(steps) - min(steps) AS steps_change,
# MAGIC          max(workout_minutes) - min(workout_minutes) AS workout_minutes_change,
# MAGIC          var_pop(workout_minutes) AS var_workout_minutes,
# MAGIC          var_pop(steps) AS var_steps
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Coursera Quiz:** How many rows and columns are in `adsda.ht_user_metrics_pca_lab`?
# MAGIC 
# MAGIC Fill in the blanks to get the answer to the question.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer back to the previous lesson's lab for help.

# COMMAND ----------

# TODO
df = spark.table("adsda.ht_user_metrics_pca_lab").toPandas()
df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 2
# MAGIC 
# MAGIC In this exercise, you will perform PCA.
# MAGIC 
# MAGIC Fill in the blanks below to perform the PCA analysis.

# COMMAND ----------

# TODO
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca = PCA(random_state=42)
pca.fit(scale(df))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Coursera Quiz:** How many components were computed?
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Refer back to the Applied PCA demo.

# COMMAND ----------

# TODO
pca.n_components_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3
# MAGIC 
# MAGIC In this this exercise, you will visualize and identify the variance explained by the first component.
# MAGIC 
# MAGIC Fill in the blanks below to complete these tasks.

# COMMAND ----------

# TODO
import matplotlib.pyplot as plt
import numpy as np

plt.bar(range(1, 25), pca.explained_variance_ratio_) 
plt.xlabel('Component') 
plt.xticks(range(1, 25))
plt.ylabel('Percent of variance explained')
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, step=0.1))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** How much of the total variation in the feature set is explained by the first component?

# COMMAND ----------

# TODO
print(pca.explained_variance_ratio_[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 4
# MAGIC 
# MAGIC In this exercise, you will determine how many components it takes to account for 90 percent of the variation in the feature set.
# MAGIC 
# MAGIC Fill in the blanks below to visualize the cumulative sum of variance explained.

# COMMAND ----------

# TODO
plt.plot(range(1, 25), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Component') 
plt.xticks(range(1, 25))
plt.ylabel('Percent of cumulative variance explained')
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, step=0.1))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The above graphs are helpful but can be hard to visualize. Let's determine this programmatically.
# MAGIC 
# MAGIC **Coursera Quiz**: How many components does it take to explain 90 percent of the variation in the original feature set? 

# COMMAND ----------

for component in list(zip(range(1, 25), np.cumsum(pca.explained_variance_ratio_))):
  if component[1] >= 0.9:
    print(component)
    break

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 5
# MAGIC 
# MAGIC In this exercise, you will examine the factor loadings of the PCA model.
# MAGIC 
# MAGIC Fill in the blanks below to return the factor loadings.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Check out the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to determine which attribute returns the components.

# COMMAND ----------

# TODO
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loadings

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: How many rows and columns are there in this loadings matrix?

# COMMAND ----------

loadings.shape

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 6
# MAGIC 
# MAGIC In this exercise, you will use the loadings from the previous exercise to determine which is the most correlated with the first component.
# MAGIC 
# MAGIC Fill in the blanks below to create a more useful loadings DataFrame using the `loadings` matrix defined in the previous exercise.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> This is the same data, but it now has helpful column and row names to make more sense of what we're looking at.

# COMMAND ----------

# TODO
import pandas as pd

component_columns = ["PC" + str(x) for x in range(1, 25)]
loadings_df = pd.DataFrame(loadings, columns=component_columns, index=df.columns)
loadings_df

# COMMAND ----------

loadings_df.shape

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC **Coursera Quiz**: Which of the features is most correlated (in any direction) with the first component `PC1`?
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> The below code uses the Pandas DataFrame API. You can always turn this back into a SQL table if you're more comfortable with SQL.

# COMMAND ----------

abs(loadings_df["PC1"]).sort_values(ascending=False)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Exercise 7
# MAGIC 
# MAGIC In this exercise, you will prepare a feature set using the first few components from our PCA process.
# MAGIC 
# MAGIC Fill in the blanks below to prepare the new feature set using only the first three components.
# MAGIC 
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Check out the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to determine which method applies dimensionality reduction to an existing feature set.

# COMMAND ----------

# TODO
component_df = pd.DataFrame(pca.transform(scale(df)), columns=component_columns)
component_3_df = component_df.loc[:, ["PC1", "PC2", "PC3"]]
component_3_df

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz**: Which of the following is a drawback of using PCA to reduce the feature space for supervised learning problems?
# MAGIC 
# MAGIC * PCA only needs a few components to represent the original features
# MAGIC * The curse of dimensionality
# MAGIC * PCA only works with a few columns at a time
# MAGIC * The resulting features are less interpretable

# COMMAND ----------

# MAGIC %md
# MAGIC Congrats! That concludes our lesson on PCA!
# MAGIC 
# MAGIC Be sure to submit your quiz answers to Coursera, and join us in the next module to learn about feature engineering and selection.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>