# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Applied Random Forest
# MAGIC 
# MAGIC **Objective**: *Demonstrate the use of random forest to build more predictive linear models.*
# MAGIC 
# MAGIC In this demo, we will use sklearn to create a random forest model.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC We will once again try to predict a customer's steps based on their recorded metrics, using the `adsda.ht_user_metrics_lifestyle` table that we created previously.

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

X = ht_lifestyle_pd_df[['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'avg_vo2', 'avg_workout_minutes']]

y = ht_lifestyle_pd_df['steps']

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit a random forest regressor model

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the results

# COMMAND ----------

y_train_predicted = rf.predict(X_train)
y_test_predicted = rf.predict(X_test)

print("R2 on training set: ", round(rf.score(X_train, y_train),3))
print("R2 on test set: ", round(rf.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC These results are already looking much better than they did in the last lesson when we used just a single decision tree.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tune the random forest hyperparameters
# MAGIC 
# MAGIC Remember that some of the hyperparameters that are available for decision trees can also be tuned in random forest. These are just applied to each of the individual decision trees, or estimators, that comprise the random forest. 
# MAGIC 
# MAGIC There are also some hyperparameters specific to the random forest itself.
# MAGIC 
# MAGIC For example, 
# MAGIC 
# MAGIC `bootstrap:bool, default=True`
# MAGIC 
# MAGIC Defines whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.

# COMMAND ----------

rf_tuned_1 = RandomForestRegressor(n_estimators=100, max_depth=8, bootstrap=True)

rf_tuned_1.fit(X_train, y_train)

# COMMAND ----------

y_train_predicted = rf_tuned_1.predict(X_train)
y_test_predicted = rf_tuned_1.predict(X_test)

print("R2 on training set: ", round(rf_tuned_1.score(X_train, y_train),3))
print("R2 on test set: ", round(rf_tuned_1.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see what happens if we don't use the bootstrap, and use the entire dataset for each tree.

# COMMAND ----------

rf_tuned_2 = RandomForestRegressor(n_estimators=50, max_depth=8, bootstrap=False)

rf_tuned_2.fit(X_train, y_train)

# COMMAND ----------

y_train_predicted = rf_tuned_2.predict(X_train)
y_test_predicted = rf_tuned_2.predict(X_test)

print("R2 on training set: ", round(rf_tuned_2.score(X_train, y_train),3))
print("R2 on test set: ", round(rf_tuned_2.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC If you do not use the boostrap sampling techinque the random forests is just a pure decision tree.<br>
# MAGIC Also the accuracy is compromised

# COMMAND ----------

# MAGIC %md
# MAGIC Performance gets slightly worse.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>