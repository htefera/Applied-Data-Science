# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # String Indexing Categorical Features
# MAGIC 
# MAGIC **Objective**: *Demonstrate how to numerically index categorical features for decision trees.*
# MAGIC 
# MAGIC In this video, we will demonstrate the one-hot encoding of categorical features and the indexing of categorical features. Then, we will compare the results from each set of features by building and evaluating a decision tree.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC ### Aggregate the user-level table
# MAGIC 
# MAGIC Remember that one of our project objectives is to predict a customer's BMI based on their recorded metrics. Therefore, we are interested in a user-level aggregation. To prepare the dataset to do this, we'll aggregate our **`adsda.ht_daily_metrics`** table at the user level, by grouping on device_id.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_lifestyle
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-lifestyle" AS (
# MAGIC   SELECT first(device_id) AS device_id,
# MAGIC          avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          avg(workout_minutes) AS avg_workout_minutes,
# MAGIC          avg(steps) AS steps,
# MAGIC          first(lifestyle) AS lifestyle
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Convert spark dataframe to Pandas
# MAGIC 
# MAGIC We will use this Pandas DataFrame with Scikit-Learn in this demo.

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examine the "lifestyle" column

# COMMAND ----------

ht_lifestyle_pd_df['lifestyle'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Method 1: Label Encoding
# MAGIC 
# MAGIC We will ue LabelEncoder to encode the lifestyle column.

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

ht_lifestyle_pd_df['lifestyle_cat'] = le.fit_transform(ht_lifestyle_pd_df['lifestyle'])

ht_lifestyle_pd_df.head()

# COMMAND ----------

ht_lifestyle_pd_df['lifestyle_cat'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Split the data into features (X) and target (y) and train-test split the data

# COMMAND ----------

X = ht_lifestyle_pd_df[['avg_active_heartrate', 'avg_vo2', 'avg_workout_minutes', 'avg_resting_heartrate', 'steps', 'lifestyle_cat']]


y = ht_lifestyle_pd_df['bmi']

# COMMAND ----------

X.shape, y.shape

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Train the decision tree

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Evaluate the decision tree
# MAGIC ##### Measure the accuracy of the decision tree regresser 

# COMMAND ----------

print("R2 on training set: ", round(dt.score(X_train, y_train),3))
print("R2 on test set: ", round(dt.score(X_test, y_test), 3))

# COMMAND ----------

From the above result the score on training set is  **1.0** and the score on the test set is **.929** <br>
From the bias variance tradeoff concept we say the model is overfitted



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Method 2: One Hot Encoding
# MAGIC 
# MAGIC We will use the Pandas method 'get_dummies' to one-hot encode the lifestyle column. Scikit-learn has a "OneHotEncoder" class that does the same thing, but using the built-in Pandas method is a bit more straightforward in our case.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Make a new version of the table

# COMMAND ----------

ht_lifestyle_dummies_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

import pandas as pd

ht_lifestyle_dummies_df = pd.get_dummies(ht_lifestyle_dummies_df, prefix=['lifestyle'], columns=['lifestyle'])

# COMMAND ----------

ht_lifestyle_dummies_df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Split the data into features (X) and target (y) and train-test split the data

# COMMAND ----------

X = ht_lifestyle_dummies_df.drop('bmi', axis=1).drop('device_id', axis=1)

y = ht_lifestyle_pd_df['bmi']

# COMMAND ----------

X.shape, y.shape

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Train the decision tree

# COMMAND ----------

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Evaluate the decision tree

# COMMAND ----------

from sklearn.metrics import mean_squared_error

y_train_predicted = dt.predict(X_train)
y_test_predicted = dt.predict(X_test)

# COMMAND ----------

print("R2 on training set: ", round(dt.score(X_train, y_train),3))
print("R2 on test set: ", round(dt.score(X_test, y_test), 3))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>