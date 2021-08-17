# Approach 1 : dropping columns with missing values


# getting names of columns with missing value
cols_with_missing = [col for col in x_train.columns
                      if x_train[col].isnull().any()]

# drop columns in training and validation data
reduced_x_train = x_train.drop(cols_with_missing,axis=1)                
reduced_x_val = x_val.drop(cols_with_missing,axis=1)


# Approach 2 : simple imputation


from sklearn.impute import SimpleImputer

# imputation
my_imputer = SimpleImputer()
imputed_x_train = pd.DataFrame(my_imputer.fit_transform(x_train))
imputed_x_val = pd.DataFrame(my_imputer.transform(x_val))

# imputation removed column names, put them back
imputed_x_train.columns = x_train.columns
imputed_x_val.columns = x_val.columns


# Approach 3 : extension of simple imputation


# make a copy to avoid changing original data
x_train_plus = x_train.copy()
x_val_plus = x_val.copy()

# make new columns indicating what will be imputed
for col in cols_with_missing:
    x_train_plus[col + 'was_missing'] = x_train_plus[col].isnull()
    x_val_plus[col + 'was_missing'] = x_val_plus[col].isnull()

# imputation
my_imputer = SimpleImputer()
imputed_x_train_plus = pd.DataFrame(my_imputer.fit_transform(x_train_plus))
imputed_x_val_plus = pd.DataFrame(my_imputer.transform(x_val_plus))

# imputation removed column names , put them back
imputed_x_train_plus.columns = x_train_plus.columns
imputed_x_val_plus.columns = x_val_plus.columns
