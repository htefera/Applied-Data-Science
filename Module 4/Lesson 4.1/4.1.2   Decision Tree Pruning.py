# Databricks notebook source
# MAGIC 
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Decision Tree Pruning
# MAGIC 
# MAGIC **Objective**: *Demonstrate the use of decision tree pruning to prevent overfitting.*
# MAGIC 
# MAGIC In this demo, we will walk through the process of tuning hyperparameters to prune decision trees with sklearn.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Prepare data
# MAGIC 
# MAGIC Remember that one of our project objectives is to predict a customer's daily average number of steps based on their other recorded metrics. Therefore, we are interested in a user-level aggregation. We will use the `adsda.ht_user_metrics_lifestyle` table that we created in the previous demo.

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

# MAGIC %sql
# MAGIC SELECT * FROM adsda.ht_user_metrics_lifestyle LIMIT 10

# COMMAND ----------

ht_lifestyle_pd_df = spark.table("adsda.ht_user_metrics_lifestyle").toPandas()

# COMMAND ----------

# X is an Independent variable(we are using five features to  predict the customer steps)
X = ht_lifestyle_pd_df[['avg_resting_heartrate', 'avg_active_heartrate', 'bmi', 'avg_vo2', 'avg_workout_minutes']]
# Y is dependent variable
y = ht_lifestyle_pd_df['steps']

# COMMAND ----------

X.shape, y.shape

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit a base decision tree
# MAGIC 
# MAGIC We will start with a baseline model with no hyperparameter tuning.

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the results

# COMMAND ----------

y_train_predicted = dt.predict(X_train)
y_test_predicted = dt.predict(X_test)

print("R2 on training set: ", round(dt.score(X_train, y_train),3))
print("R2 on test set: ", round(dt.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Overfitting and high variance!
# MAGIC 
# MAGIC The decision tree is fitting 100% perfectly on the training set, but doesn't do very well on the test set. <br>
# MAGIC **In machine learning domain this is called variance**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tune the decision tree hyperparameters
# MAGIC 
# MAGIC Remember some of the hyperparameters that can be tuned in a decision tree to prevent overfitting on the training set.

# COMMAND ----------

# MAGIC %md
# MAGIC **Maximum tree depth**
# MAGIC    - limiting how deep the tree grows (how many levels of splitting)
# MAGIC    - In many scenarios the accuracy could be optimzed by adding the depth level but this is not always true

# COMMAND ----------

dt_depth = DecisionTreeRegressor(max_depth=7)

dt_depth.fit(X_train, y_train)

y_train_predicted = dt_depth.predict(X_train)
y_test_predicted = dt_depth.predict(X_test)

print("R2 on training set: ", round(dt_depth.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_depth.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The model is no longer overfitting to the training data, and has gotten slightly better on the test set, but now it has high bias (not learning the training set very well). Let's try tuning another hyperparameter.

# COMMAND ----------

# MAGIC %md
# MAGIC  **Minimum node size**
# MAGIC    - requiring that each node have a minimum number of data points in order to split it further

# COMMAND ----------

dt_node = DecisionTreeRegressor(max_depth=6, min_samples_split=3)

dt_node.fit(X_train, y_train)

y_train_predicted = dt_node.predict(X_train)
y_test_predicted = dt_node.predict(X_test)

print("R2 on training set: ", round(dt_node.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_node.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC This is getting better, but the bias and variance are still too high and the performance on the test set is pretty far below the training performance. Let's try something else.

# COMMAND ----------

# MAGIC %md
# MAGIC **Minimum leaf size**
# MAGIC   - Requiring at least a certain number of data points in each leaf

# COMMAND ----------

dt_leaf = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=5)

dt_leaf.fit(X_train, y_train) # training the decisioin tree

y_train_predicted = dt_leaf.predict(X_train) # predicting on the train data
y_test_predicted = dt_leaf.predict(X_test) # predicting on the test data

print("R2 on training set: ", round(dt_leaf.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_leaf.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC That didn't help much. We'll try one more.

# COMMAND ----------

# MAGIC %md
# MAGIC **Maximum features**
# MAGIC   - maximum number of features to consider at each split
# MAGIC   - introduces randomness

# COMMAND ----------

dt_features = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=3, max_features=3)

dt_features.fit(X_train, y_train)

y_train_predicted = dt_features.predict(X_train)
y_test_predicted = dt_features.predict(X_test)

print("R2 on training set: ", round(dt_features.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_features.score(X_test, y_test), 3))

# COMMAND ----------

# MAGIC %md
# MAGIC **Minimum Weighted Fraction** <br>
# MAGIC - The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node <br>
# MAGIC - min_weight_fraction_leaf is the fraction of the input samples required to be at a leaf node where weights are determined by sample_weight, this is a way to deal with class imbalance
# MAGIC - In our case this addition does not bring any change as we do not have the calss imblance problem 

# COMMAND ----------

dt_features = DecisionTreeRegressor(max_depth=8, min_samples_split=2, min_samples_leaf=3, max_features=3,min_weight_fraction_leaf=0.001)

dt_features.fit(X_train, y_train)

y_train_predicted = dt_features.predict(X_train)
y_test_predicted = dt_features.predict(X_test)

print("R2 on training set: ", round(dt_features.score(X_train, y_train),3))
print("R2 on test set: ", round(dt_features.score(X_test, y_test), 3))
tree.plot_tree(dt_features)

# COMMAND ----------

# MAGIC %md
# MAGIC **Ploting Decision tree**

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree


# COMMAND ----------

import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# COMMAND ----------

# MAGIC %md 
# MAGIC **Splitting Data into Training and Test Sets** 

# COMMAND ----------

X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)


# Step 1: Import the model you want to use
# This was already imported earlier in the notebook so commenting out
#from sklearn.tree import DecisionTreeClassifier
# Step 2: Make an instance of the Model
dtree = DecisionTreeClassifier(max_depth = 2, 
                             random_state = 42)
# Step 3: Train the model on the data
dtree.fit(X_train, Y_train)
# Step 4: Predict labels of unseen (test) data
# Not doing this step in the tutorial
dtree.predict(X_test)

# COMMAND ----------

tree.plot_tree(dtree)

# COMMAND ----------

# MAGIC %md
# MAGIC the code below tries to make the decision tree more interpretable by adding in feature and class names (as well as setting filled = True).

# COMMAND ----------

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dtree,
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('imagename.png')

# COMMAND ----------

# MAGIC %md
# MAGIC **How to Visualize Decision Trees using Graphviz**<br>
# MAGIC **First: Export your model to a dot file**<br>
# MAGIC **Second:use the dot command to convert the dot file into a png file**

# COMMAND ----------

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
# Prepare the data data
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Fit the classifier with default hyper-parameters
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)


# COMMAND ----------

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names=iris.feature_names,  
                   class_names=iris.target_names,
                   filled=True)

# COMMAND ----------

# MAGIC %md
# MAGIC **Visualize Decision Tree with graphviz**

# COMMAND ----------

import graphviz
# DOT data
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=iris.feature_names,  
                                class_names=iris.target_names,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 

# COMMAND ----------



# COMMAND ----------





# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC 
# MAGIC We have seen that it is difficult to get a decision tree that doesn't have high variance, even with some tuning of the hyperparameters. In the next lesson, we'll learn about a method that combines decision trees to obtain better models with higher predictive power.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>