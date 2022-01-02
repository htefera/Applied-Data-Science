# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Weighting Classes in Random Forest
# MAGIC 
# MAGIC **Objective**: *Demonstrate label-based record weighting as a method for balancing classes in evaluation.*
# MAGIC 
# MAGIC In this video we will demonstrate how to perform record weighting based on the class distribution in the training data set, in order to achieve equal weighting of label classes when evaluating models. 

# COMMAND ----------

# MAGIC %pip install imbalanced-learn

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC We will once again be using the `adsda.ht_user_metrics_lifestyle` table that we previously created, but now joined to the `adsda.ht_users` table, and predicting which country a user is from. 

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_lifestyle_country
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-lifestyle_country" AS (
# MAGIC   SELECT metrics.*, users.country 
# MAGIC   FROM adsda.ht_user_metrics_lifestyle AS metrics
# MAGIC   JOIN adsda.ht_users AS users
# MAGIC   ON metrics.device_id = users.device_id
# MAGIC   )

# COMMAND ----------

ht_df = spark.table("adsda.ht_user_metrics_lifestyle_country").toPandas()

# COMMAND ----------

ht_df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC We can check how many of each class there are:

# COMMAND ----------

print(ht_df['country'].value_counts())

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

ht_df['country_cat'] = le.fit_transform(ht_df['country'])

ht_df.head(5)

# COMMAND ----------

X = (ht_df.drop("country", axis=1)
     .drop("country_cat", axis=1)
     .drop("lifestyle", axis=1)
     .drop("device_id", axis=1)
    )
                           
y = ht_df['country_cat']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a random forest model using class weights
# MAGIC 
# MAGIC Recall that sklearn has a built in utility function that will calculate weights based on class frequencies. It does this by automatically weighting classes inversely proportional to how frequently they appear in the data.
# MAGIC 
# MAGIC We can use this class weight function as a parameter specified for a model, with several options:
# MAGIC 
# MAGIC  - `None` 
# MAGIC   - this is the default
# MAGIC   - the class weights will be uniform
# MAGIC  - `balanced`
# MAGIC   - the function will calculate the class weights automatically 
# MAGIC  - `balanced_subsample`
# MAGIC   - same as “balanced” except that weights are computed based on the bootstrap sample for each individual tree
# MAGIC  - as a dictionary
# MAGIC   - the keys are the classes and the values are the desired class weights

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

y_train.value_counts()

# COMMAND ----------

y_test.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC **Default (None):**

# COMMAND ----------

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight=None)

rf.fit(X_train, y_train)

print(confusion_matrix(y_test, rf.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Balanced:**

# COMMAND ----------

rf = RandomForestClassifier(class_weight="balanced")

rf.fit(X_train, y_train)

print(confusion_matrix(y_test, rf.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Balanced subsample:**

# COMMAND ----------

rf = RandomForestClassifier(class_weight="balanced_subsample")

rf.fit(X_train, y_train)

print(confusion_matrix(y_test, rf.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Dictionary of ratios:**
# MAGIC 
# MAGIC We can calculate the exact ratio we would use to evenly balance the classes, and use that in our class weight dictionary. We can use the sklearn.utils class_weight function to accomplish this.

# COMMAND ----------

from sklearn.utils import class_weight

weights = class_weight.compute_class_weight(class_weight='balanced', classes=[0, 1], y=y)

print(weights)

# COMMAND ----------

class_weights_dict = dict(enumerate(weights))

print(class_weights_dict)

# COMMAND ----------

rf = RandomForestClassifier(class_weight=class_weights_dict)

rf.fit(X_train, y_train)

print(confusion_matrix(y_test, rf.predict(X_test)))

# COMMAND ----------

rf = RandomForestClassifier(class_weight={0: 999, 1: 0.0009})

rf.fit(X_train, y_train)

print(confusion_matrix(y_test, rf.predict(X_test)))

# COMMAND ----------

# MAGIC %md
# MAGIC The performance isn't improving with any of these methods for balancing the classes, which indicates that we might need to try some hyperparameter tuning, or another type of machine learning model altogether, to obtain more accuract classifications. We can first check how a Random Forest model would do if we balanced the classes before training, instead of using class weights.

# COMMAND ----------

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')

X_over, y_over = oversample.fit_resample(X, y)

print(y_over.value_counts())

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over)

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

print(confusion_matrix(y_test, rf.predict(X_test)))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>