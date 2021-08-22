# Gradient Boosting

# Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
# It begins by initializing the ensemble with a single model, whose predictions can be pretty naive.
# (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)

# Then, we start the cycle:

# 1. First, we use the current ensemble to generate predictions for each observation in the dataset.
#    To make a prediction, we add the predictions from all models in the ensemble.

# 2. These predictions are used to calculate a loss function (like mean squared error, for instance).

# 3. Then, we use the loss function to fit a new model that will be added to the ensemble.
#    Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss.
#    (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss
#    function to determine the parameters in this new model.)

# 4. Finally, we add the new model to ensemble, and ...... repeat!

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("file")

# select subset of predictors
cols_to_use = ['rooms','distance','landsize','buildingarea','yearbuilt']
x = data[cols_to_use]

# select target
y = data.price

# separate data into training and validation sets
x_train,x_valid,y_train,y_valid = train_test_split(x,y)

# In this example, you'll work with the XGBoost library. XGBoost stands for extreme gradient boosting, which is an 
# implementation of gradient boosting with several additional features focused on performance and speed. 
# (Scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages.

from XGboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(x_train,y_train)

from sklearn.metrics import mean_absolute_error

preds = my_model.predict(x_valid)
print("mae : " + str(mean_absolute_error(y_valid,preds)))

# Parameter tuning

# XGBoost has a few parameters that can dramatically affect accuracy and training speed.
#  The first parameters you should understand are:

# 1. n_estimators

# n_estimators specifies how many times to go through the modeling cycle described above. 
# It is equal to the number of models that we include in the ensemble.

# 1) Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
# 2) Too high a value causes overfitting, which causes accurate predictions on training data, but inaccurate predictions
#    on test data (which is what we care about).

# Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.

# Here is the code to set the number of models in the ensemble:

my_model = XGBRegressor(n_estimators=500)
my_model.fit(x_train,y_train)

# 2. early_stopping_rounds

# early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. Early stopping causes
# the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for
# n_estimators. It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the
# optimal time to stop iterating.

# Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a 
# number for how many rounds of straight deterioration to allow before stopping. Setting early_stopping_rounds=5 is a 
# reasonable choice. In this case, we stop after 5 straight rounds of deteriorating validation scores.

# When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - 
# this is done by setting the eval_set parameter.

my_model = XGBRegressor(n_estimators=500)
my_model.fit(x_train,y_train,
              early_stopping_rounds = 5,
              eval_set = [(x_valid,y_valid)],
              verbose = False)

# If you later want to fit a model with all of your data, set n_estimators to whatever value you found to
# be optimal when run with early stopping.              

# 3. learning_rate

# Instead of getting predictions by simply adding up the predictions from each component model, we can multiply 
# the predictions from each model by a small number (known as the learning rate) before adding them in.

# This means each tree we add to the ensemble helps us less. So, we can set a higher value for n_estimators without
# overfitting. If we use early stopping, the appropriate number of trees will be determined automatically.

# In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it 
# will also take the model longer to train since it does more iterations through the cycle. As default, XGBoost sets 
# learning_rate=0.1.

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(x_train,y_train,
              early_stopping_rounds = 5,
              eval_set = [(x_valid,y_valid)],
              verbose = False)

# 4. n_jobs

# On larger datasets where runtime is a consideration, you can use parallelism to build your models faster.
#  It's common to set the parameter n_jobs equal to the number of cores on your machine. On smaller datasets,
#  this won't help.

# The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction. 
# But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(x_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(x_valid, y_valid)], 
             verbose=False)











