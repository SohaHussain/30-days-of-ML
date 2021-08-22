# In cross-validation, we run our modeling process on different subsets of the data to get 
# multiple measures of model quality.

import pandas as pd

# read the data
data = pd.read_csv("file")

# select subset of predictors
cols_to_use = ['Rooms','distance','landsize','buildingarea','yearbuilt']
x = data[cols_to_use]

# select target
y = data.price

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor',SimpleImputer()),
                               ('model',RandomForestRegressor(n_estimators=100,random_state=0))])


# We obtain the cross-validation scores with the cross_val_score() function from scikit-learn.
#  We set the number of folds with the cv parameter.

from sklearn.model_selection import cross_val_score

# multiply by -1 since sklearn calculates negative value
scores= -1* cross_val_score(my_pipeline,x,y,cv=5,scoring='neg_mean_absolute_error')
print("mae scores \n", scores)

# We typically want a single measure of model quality to compare alternative models. 
# So we take the average across experiments.

print("average mae score")
print(scores.mean())

