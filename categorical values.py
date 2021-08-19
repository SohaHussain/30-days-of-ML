import pandas as pd
from sklearn.model_selection import train_test_split

# read data
data = pd.read_csv('file name')

# separate target variables
y=data.Price
x=data.drop('Price',axis=1)

# divide data into training and validation set
x_train_full, x_val_full, y_train, y_valid = train_test_split(x,y,train_size=0.8, tets_size=0.2, random_state=0)

# cardinality means the no. of unique values in the column
# select categorical columns with relatively low cardinality
low_cardinality_cols = [cname for cname in x_train_full.columns if x_train_full[cname].nunique() <10 and 
                         x_train_full[cname].dtype == 'object']

# select numerical columns
num_cols = [col for col in x_train_full.columns if x_train_full[col].dtype in ['int64','float64']]

# keep selected columns only
my_cols = low_cardinality_cols + num_cols
x_train = x_train_full[my_cols].copy()
x_valid = x_val_full[my_cols].copy()

# get list of categorical columns
s= (x_train.dtypes == 'object')
object_cols = list(s[s].index)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# func for comparing different approaches
def score_data(x_train,x_valid,y_train,y_valid):
    model = RandomForestRegressor(n_estimators=100,random_state=0)
    model.fit(x_train,y_train)
    pred = model.predict(x_valid)
    return mean_absolute_error(y_valid,pred)

# Approach 1 : dropping categorical columns

drop_x_train = x_train.select_dtypes(exclude=['object'])
drop_x_valid = x_valid.select_dtypes(exclude=['object'])
print(score_data(drop_x_train,drop_x_valid,y_train,y_valid))

# Approach 2 : ordinal encoding

from sklearn.preprocessing import OrdinalEncoder

# make copy to avoid changing original data
label_x_train = x_train.copy()
label_x_valid = x_valid.copy()

# apply ordinal encoder to each column with categorical data
oe = OrdinalEncoder()
label_x_train[object_cols] = oe.fit_transform(x_train[object_cols])
label_x_valid[object_cols] = oe.transform(x_valid[object_cols])

# Approach 3 : one hot encoding

from sklearn.preprocessing import OneHotEncoder

# apply one hot encoder to each column with categorical data
ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
oh_cols_train = pd.DataFrame(ohe.fit_transform(x_train[object_cols]))
oh_cols_valid = pd.DataFrame(ohe.transform(x_valid[object_cols]))

# one hot encoding removed index so put it back
oh_cols_train.index = x_train.index
oh_cols_valid.index = x_valid.index

# remove categorical columns ( will replace it with one hot encoding)
num_x_train = x_train.drop(object_cols,axis=1)
num_x_valid = x_valid.drop(object_cols,axis=1)

# add one hot encoded columns to numerical features
oh_x_train = pd.concat([num_x_train,oh_cols_train],axis=1)
oh_x_valid = pd.concat([num_x_valid,oh_cols_valid],axis=1)


    

    
