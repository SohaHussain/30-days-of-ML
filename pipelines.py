# Pipelines are a simple way to keep your data preprocessing and modeling code organized. 
# Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were 
# a single step.

# step 1 : define preprocessing steps

# Similar to how a pipeline bundles together preprocessing and modeling steps, we use the ColumnTransformer class 
# to bundle together different preprocessing steps. The code below:
# 1. imputes missing values in numerical data, and
# 2. imputes missing values and applies a one-hot encoding to categorical data.

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])

# bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[ 
    ('num',numerical_transformer,numerical_cols),
    ('cat',categorical_transformer,categorical_cols)
])

# step 2 : define the model

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,random_state=0)

# step 3 : create and evaluate the pipeline

# Finally, we use the Pipeline class to define a pipeline that bundles the preprocessing and modeling steps. 
# There are a few important things to notice:

# 1. With the pipeline, we preprocess the training data and fit the model in a single line of code.
#  (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps.
#  This becomes especially messy if we have to deal with both numerical and categorical variables!)

# 2. With the pipeline, we supply the unprocessed features in X_valid to the predict() command,
#  and the pipeline automatically preprocesses the features before generating predictions. 
# (However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)

from sklearn.metrics import mean_absolute_error

# bundle preprocessing and modelling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                            ('model',model)])

# preprocessing of training data, fit the model
my_pipeline.fit(x_train,y_train)

# preprocessing of validation data , get predictions
preds = my_pipeline.predict(x_valid)

# evaluate the model
score = mean_absolute_error(y_valid,preds)
print('MAE',score)
