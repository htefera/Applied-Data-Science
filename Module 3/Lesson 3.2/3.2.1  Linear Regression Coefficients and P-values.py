# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Linear Regression Coefficients and P-values
# MAGIC 
# MAGIC **Objective**: *Demonstrate feature importance within linear regression.*
# MAGIC 
# MAGIC In this demo, we will complete a series of exercises to show how to examine the P Values from a Linear Regression model.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC ### Aggregate our user-level table
# MAGIC 
# MAGIC Remember that one of our project objectives is to predict a customer's BMI based on their recorded metrics. Therefore, we are interested in a user-level clustering. To prepare the dataset to do this, we'll aggregate our **`adsda.ht_user_metrics`** table at the user level.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE adsda.ht_user_metrics_lifestyle
# MAGIC USING DELTA LOCATION "/adsda/ht-user-metrics-lifestyle" AS (
# MAGIC   SELECT avg(resting_heartrate) AS avg_resting_heartrate,
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

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics_lifestyle

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Convert Spark DataFrame to Pandas
# MAGIC 
# MAGIC We will use this Pandas DataFrame in this demo.

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC View the data

# COMMAND ----------

ht_lifestyle_pd_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fitting a Linear Regression Model and Examining Coefficients and P Values
# MAGIC 
# MAGIC This process has a few steps so we'll string everything together and explain step by step.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 1 - Feature Engineering
# MAGIC Pandas has a built-in method to one-hot encode called `get_dummies()`. We'll use that here to transform the `lifestyle` feature into a numeric feature.

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC We can see that the object we receive back is a new DataFrame. We'll save this out to a variable

# COMMAND ----------

lifestyle_dummies_df = pd.get_dummies(ht_lifestyle_pd_df['lifestyle'])

# COMMAND ----------

# MAGIC %md
# MAGIC Then we'll join this back onto our original dataframe

# COMMAND ----------

ht_lifestyle_pd_df = ht_lifestyle_pd_df.join(lifestyle_dummies_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we'll drop the original `lifestyle` column because it is now uncessary.

# COMMAND ----------

ht_lifestyle_pd_df.drop('lifestyle', axis=1, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step #2 - Create feature matrix and target
# MAGIC Now we need to create our X and y from our features and target. Recall that our target is the thing we are trying to predict, BMI, given some features about an observation.

# COMMAND ----------

X = ht_lifestyle_pd_df.drop('bmi', axis=1) # all columns except bmi
y = ht_lifestyle_pd_df['bmi'] # BMI is the target variable 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 3 - Fit our model
# MAGIC Import statsmodels.
# MAGIC 
# MAGIC 🎯Note that the statsmodels api refers to our target variable as the endogenous or dependent variable and our features as the exogenous or independent variable.

# COMMAND ----------

import statsmodels.api as sm

# COMMAND ----------

model = sm.OLS(endog=y, exog=X) # train the model Y is the dependent variable where as X is the independent variable 
#Ordinary Least Squares (OLS) is a form of regression, widely used in Machine Learning
bmi_ols_results = model.fit() # fit the whole model 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step 4 - Examine results
# MAGIC Once our model is fit, we have helper methods and attributes available. Of interest to us first are the coefficients. The most robust of all of these is a method called `.summary()`

# COMMAND ----------

bmi_ols_results.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC The coefficients of our model are in the middle of the output.<br>
# MAGIC **It looks like the largest coefficient is our `athlete` column, which if we recall, refers to whether or not a person is an athlete**.<br> 
# MAGIC **Intuitively, this makes sense - people who are athletes are likely to have a different BMI than people who are not. We also see some negative coefficients.**<br> 
# MAGIC **This does not imply that these features are unwanted or not helpful, it just means that the relationship between the two variables moves in opposite directions.**<br> 
# MAGIC **Intuitively, this also makes sense - the more average workout minutes someone has, the lower their BMI is. Put another way: as workout minutes go 👆🏽, BMI goes 👇🏽.**
# MAGIC 
# MAGIC Let's examine P-Values

# COMMAND ----------

bmi_ols_results.pvalues

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like all of our features have very low P Values. **P-values here are answering the question: what is the probability that a world exists where the coefficient for this is equal to zero (no effect)? Given our P-values can assume that there is a very low probability that these coefficients do not have `no effect`.**

# COMMAND ----------

# MAGIC %md
# MAGIC To interpret these in the context of `feature importance` is first with coefficients. <br>
# MAGIC **We can interpret coeffcients as a measure of the importance in our model. Here, our larger coefficients - `Athlete`, for example - can be thought of as being important.**<br>
# MAGIC **With P-values, the smaller the value, the more likely our feature has an effect on the target! We can think of this as an impactful feature.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Nicely done!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>