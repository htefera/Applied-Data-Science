# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # A Review of Assigning Classes
# MAGIC 
# MAGIC **Objective**: *Demonstrate how to assign classes based on predicted probabilities.*
# MAGIC 
# MAGIC In this video, we will review how to assign classes based on predicted probabilities. We will look at using different threshold values to do this.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC We will once again be using the `adsda.ht_user_metrics_lifestyle` table that we previously created. This time, however, we will attempt to predict which lifestyle class they fall into: 

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

ht_lifestyle_pd_df['lifestyle'].unique().tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC We want to know whether each user is either "sedentary" or a "cardio enthusiast", or "athlete" or "weight trainer" (grouping these four categories into two). 
# MAGIC 
# MAGIC Therefore we will convert the four categories to two numeric classes, 0 and 1. We will accomplish this using a list comprehension and Pandas.

# COMMAND ----------

ht_lifestyle_pd_df['lifestyle_num'] = [0 if (x=='Sedentary' or x=='Cardio Enthusiast') else 1 for x in ht_lifestyle_pd_df['lifestyle']]

# COMMAND ----------

# MAGIC %md
# MAGIC We can check how many of each category there are:

# COMMAND ----------

ht_lifestyle_pd_df['lifestyle_num'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a classification model
# MAGIC 
# MAGIC Now that we have a two class target, we will build a model to predict which class a user is in.

# COMMAND ----------

X = ht_lifestyle_pd_df[['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'avg_vo2', 'avg_workout_minutes', 'steps']]

y = ht_lifestyle_pd_df['lifestyle_num']

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Fit a random forest classifier

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examine the output
# MAGIC 
# MAGIC In this example, we are not interested in evaluating how the model performs (accuracy) - we are only looking at the predicted classes. Therefore, we will skip reviewing the classification metrics and go straight to the predicted classes.

# COMMAND ----------

import pandas as pd

preds_df = pd.DataFrame({'lifestyle': y_test, 
              'lifestyle_predicted': rf.predict(X_test),
              'predicted_proba_0': rf.predict_proba(X_test)[:, 0]
             })

# COMMAND ----------

preds_df.sample(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adjusting the threshold for predicted class
# MAGIC 
# MAGIC The model predicts that a particular sample is class 0 if its predicted probability for that class is greater than 0.5, which is the default threshold. 
# MAGIC 
# MAGIC In this case, we have decided that we want to err on the side of caution and be very careful not to mistakenly assign someone to class 0 (sedentary or cardio enthusiast), even if that means we incorrectly assign some to class 1. We will do this by adjusting the probability threshold to 0.7 for class 0. This means that we only assign someone to class 0 if the model says the probability of belonging to that class is greater than 70%.

# COMMAND ----------

preds_df['lifestyle_predicted_adjusted'] = [0 if x > 0.7 else 1 for x in preds_df['predicted_proba_0']]

# COMMAND ----------

preds_df.sample(20)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>