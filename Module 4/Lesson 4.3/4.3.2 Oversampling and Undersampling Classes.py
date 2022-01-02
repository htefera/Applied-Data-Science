# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Oversampling and Undersampling Classes
# MAGIC 
# MAGIC **Objective**: *Demonstrate how to bootstrap records based on their label values.*
# MAGIC 
# MAGIC In this video we will demonstrate how to bootstrap training set records into a new training set based on the target class distribution to ensure a more balanced distribution in the training set.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC We also need to load the below library.

# COMMAND ----------

# MAGIC %pip install imbalanced-learn

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC We will once again be using the `adsda.ht_user_metrics_lifestyle` table that we previously created, and predicting which lifestyle class users fall into. 

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

ht_lifestyle_pd_df['lifestyle'].unique().tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC In this example, we want to know whether each user is "sedentary" or "active" (any of the other three lifestyle classes). 
# MAGIC 
# MAGIC Therefore we will convert the four categories into two numeric classes, 0 and 1. We will accomplish this using a list comprehension and Pandas.

# COMMAND ----------

ht_lifestyle_pd_df['lifestyle_num'] = [0 if x=='Sedentary' else 1 for x in ht_lifestyle_pd_df['lifestyle']]

# COMMAND ----------

# MAGIC %md
# MAGIC We can check how many of each category there are:

# COMMAND ----------

ht_lifestyle_pd_df['lifestyle_num'].value_counts()

# COMMAND ----------

X = ht_lifestyle_pd_df.drop("lifestyle", axis=1).drop("lifestyle_num", axis=1)
                           
y = ht_lifestyle_pd_df['lifestyle_num']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bootstrap sample the minority class
# MAGIC 
# MAGIC 
# MAGIC We now have a dataset with a slightly imbalanced target class: class 1 is almost 10 times bigger than class 0. (This is not a very major imbalance, relatively speaking.) We will attempt to balance the dataset using the bootstrap method.
# MAGIC 
# MAGIC We will perform random oversampling, where we randomly duplicate examples in the minority class by sampling with replacement.

# COMMAND ----------

# MAGIC %md
# MAGIC We set the sampling strategy to `minority` which will make the minority class the same size as the majority class.

# COMMAND ----------

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')

# COMMAND ----------

# MAGIC %md
# MAGIC Fit and apply the oversample transformation:

# COMMAND ----------

X_over, y_over = oversample.fit_resample(X, y)

# COMMAND ----------

y_over.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Undersample the majority class
# MAGIC 
# MAGIC Next, we will perform random undersampling, where we randomly delete examples from the majority class.
# MAGIC 
# MAGIC Setting the sampling strategy to `majority` will make the majority class the same size as the minority class.

# COMMAND ----------

from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy='majority')

# COMMAND ----------

X_under, y_under = undersample.fit_resample(X, y)

# COMMAND ----------

y_under.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of simply saying that we want the majority class to be the same size as the minority, we can specify a ratio:

# COMMAND ----------

undersample_2 = RandomUnderSampler(sampling_strategy=0.75)

# COMMAND ----------

X_under2, y_under2 = undersample_2.fit_resample(X, y)

# COMMAND ----------

y_under2.value_counts()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>