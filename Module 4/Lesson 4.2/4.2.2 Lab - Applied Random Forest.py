# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Applied Random Forest Lab
# MAGIC 
# MAGIC **Objective**: *Apply random forests to a regression problem in an effort to improve model generalization.*
# MAGIC 
# MAGIC In this lab you will complete a series of guided exercises where you will build a random forest model to solve a regression problem. You will need to prepare the categorical variable appropriately and assess the output of the model. When complete, please use your answers to the exercises to answer questions in the following quiz within Coursera.

# COMMAND ----------

# MAGIC %run ../../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 1
# MAGIC 
# MAGIC In this exercise, you will use the user-level lifestyle table. Run the following cell to make sure you can access the `adsda.ht_user_metrics_lifestyle` table.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM adsda.ht_user_metrics_lifestyle
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC Fill in the following cell to create a Pandas DataFrame from the Spark table.

# COMMAND ----------

# TODO
ht_metrics_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 2
# MAGIC 
# MAGIC In this exercise, you will encode the categorical feature `lifestyle` column using `LabelEncoder`.
# MAGIC 
# MAGIC Fill in the blanks to complete this task.

# COMMAND ----------

# TODO
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
ht_metrics_pd_df['lifestyle_cat'] = le.fit_transform(ht_metrics_pd_df['lifestyle'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3 
# MAGIC 
# MAGIC In this exercise, you will build a random forest regression model
# MAGIC 
# MAGIC We will once again try to predict a user's average `vo2` using their other metrics.
# MAGIC 
# MAGIC Remember to set the `random_state` to 42!

# COMMAND ----------

# MAGIC %md
# MAGIC Before splitting the data and fitting the model, import the packages you will need from sklearn for the train test split and the Random Forest regressor.

# COMMAND ----------

# Answer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# COMMAND ----------

# TODO
X = ht_metrics_pd_df[['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'steps', 'avg_workout_minutes', 'lifestyle_cat']]
y = ht_metrics_pd_df['avg_vo2']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train,y_train)

y_train_predicted =rf.predict(X_train)
y_test_predicted = rf.predict(X_test)

print("R2 on training set: ", round(rf.score(X_train, y_train), 3))
print("R2 on test set: ", round(rf.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the `rf` model, what is the R-squared score on the training and test set? <br>
# MAGIC **Answer= 0.992 and .944**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 4
# MAGIC 
# MAGIC Even though the untuned random forest did very well already, explore how tuning some hyperparameters affects the output.
# MAGIC 
# MAGIC You will build three models:
# MAGIC 1. With `n_estimators`=10
# MAGIC 1. With `max_depth`=2
# MAGIC 1. With `bootstrap`=False

# COMMAND ----------

rf_tuned_1 = RandomForestRegressor(n_estimators=10, random_state=42)

rf_tuned_1.fit(X_train, y_train)

y_train_predicted = rf_tuned_1.predict(X_train)
y_test_predicted = rf_tuned_1.predict(X_test)

print("R2 on training set: ", round(rf_tuned_1.score(X_train, y_train),3))
print("R2 on test set: ", round(rf_tuned_1.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the `rf_tuned_1` model, what is the R-squared score on the training and test set? <br>
# MAGIC **Answer==.99 and .937**

# COMMAND ----------

rf_tuned_2 = RandomForestRegressor(max_depth=2, random_state=42)

rf_tuned_2.fit(X_train, y_train)

y_train_predicted = rf_tuned_2.predict(X_train)
y_test_predicted = rf_tuned_2.predict(X_test)

print("R2 on training set: ", round(rf_tuned_2.score(X_train, y_train),3))
print("R2 on test set: ", round(rf_tuned_2.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the `rf_tuned_2` model, what is the R-squared score on the training and test set? <br>
# MAGIC **Answer== 0.868 and 0.86**

# COMMAND ----------

rf_tuned_3 = RandomForestRegressor(bootstrap=False, random_state=42)

rf_tuned_3.fit(X_train, y_train)

y_train_predicted = rf_tuned_3.predict(X_train)
y_test_predicted = rf_tuned_3.predict(X_test)

print("R2 on training set: ", round(rf_tuned_3.score(X_train, y_train),3))
print("R2 on test set: ", round(rf_tuned_3.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the `rf_tuned_3` model, what is the R-squared score on the training and test set?<br>
# MAGIC **Answer==1.0 and 0.901 **

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** Which of the tuned random forest models had the best performance on the test set?<br>
# MAGIC **Answer==rf_runned_3**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>