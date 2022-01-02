# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Label Imbalance Lab
# MAGIC 
# MAGIC **Objective**: *Apply label balancing techniques to a random forest to optimize a recall metric.*
# MAGIC 
# MAGIC In this lab you will complete a series of guided exercises where you build a random forest to solve an imbalanced binary classification problem and apply both bootstrapping and record weighting to balance the data set. They should then compare the results to an unbalanced baseline that was given. When complete, please use your answers to the exercises to answer questions in the following quiz within Coursera.

# COMMAND ----------

# MAGIC %pip install imbalanced-learn

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
# MAGIC In this exercise, encode the `lifestyle` column by passing a dictionary of all values to the `lifestyle`. 
# MAGIC 
# MAGIC Use the following values:  `Sedentary` should be a 0 and all other classes should be 1. 
# MAGIC 
# MAGIC Fill in the blanks in the below code block.

# COMMAND ----------

# ANSWER
ht_metrics_pd_df['lifestyle_cat'] = ht_metrics_pd_df['lifestyle'].map({'Sedentary':0, 'Weight Trainer':1, 'Athlete':1, 'Cardio Enthusiast':1 })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 3
# MAGIC 
# MAGIC In this exercise, you will build a random forest classification model.
# MAGIC 
# MAGIC We will once again try to predict a user's `lifestyle` using their other metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC Before splitting the data and fitting the model, import the packages you will need from sklearn for the train test split and the random forest classifier.

# COMMAND ----------

ht_metrics_pd_df.columns

# COMMAND ----------

# ANSWER
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

# ANSWER
X = ht_metrics_pd_df[['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'steps', 'avg_workout_minutes', 'avg_vo2']]
y = ht_metrics_pd_df['lifestyle_cat']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)

y_train_predicted = rf_base.predict(X_train)
y_test_predicted = rf_base.predict(X_test)

print("Accuracy on training set: ", round(rf_base.score(X_train, y_train),3))
print("Accuracy on test set: ", round(rf_base.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the base `rf` model, what is the accuracy score on the training and test set?<br>
# MAGIC **Answer= 1.0 and 1.0**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 4
# MAGIC 
# MAGIC Change the `class_weight` parameter to `balanced`, and compute the accuracy and print the confusion matrix on the test set.

# COMMAND ----------

# ANSWER
from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier(class_weight='balanced')
rf.fit(X_train, y_train)

print(rf.score(X_test, y_test))
print(rf.score(X_train, y_train))
print(confusion_matrix(y_test, rf.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the class weight balanced model, what is the accuracy score on the test set?<br>
# MAGIC **Answer==1.0 **

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the class weight balanced model, what is the total delta of True Negatives from the base Random Forest model?
# MAGIC **Answer=True Negative=671**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercise 5
# MAGIC 
# MAGIC Choose a `sampling strategy` of `minority`. Then fit and score a random forest model. 

# COMMAND ----------

# Answer
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')

# COMMAND ----------

# MAGIC %md
# MAGIC Create an oversample of the features and target.

# COMMAND ----------

# ANSWER
X_over, y_over = oversample.fit_resample(X, y)

print(y_over.value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC Create a train test split with the resampled data

# COMMAND ----------

# Answer
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over)

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** How many of our `sedentary` class do we now have in our train set?<br>
# MAGIC **Answer==2001**

# COMMAND ----------

# Answer
y_train.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Fit a model with the new train test split and print the confusion matrix

# COMMAND ----------

# ANSWER
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
print(confusion_matrix(y_test, rf.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the class weight balanced model, what is the accuracy score on the test set?<br>
# MAGIC **Anser==1.0**

# COMMAND ----------

# MAGIC %md
# MAGIC **Coursera Quiz:** For the class weight balanced model, what is the total delta of True Negatives from the base Random Forest model?<br>
# MAGIC **Answer==657**

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>