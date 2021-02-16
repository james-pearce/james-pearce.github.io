---
layout: post
title:  "New features in scikit-learn part 1 — successive halving"
date:   2021-02-16 09:00:00 +1100
categories: machine-learning sklearn
---

# New features in scikit-learn 0.24

Welcome to a short series of posts where I look at some of the new features introduced in [scikit-learn](https://scikit-learn.org/stable/) version 0.24, a newish release of the popular Python machine learning library.

> You can find a copy of this Jupyter notebook [here](https://drive.google.com/file/d/1T_7sTm0GvMLJFMTQbxtRJM4HTZWqD0-z/view?usp=sharing).

## Feature #1: Successive Halving

The first new feature we will examine is called 'successive halving'. It is suggested as an alternative to [grid searches](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) or [random searches](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) for [hyperparameter tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization).

Note that I prefer random searches in general because

  1. the data scientist has more control over the execution time of the search; and
  2. [random searches can lean to a more accurate result](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) compared with a grid search because they go "in between"
     the grid 'lines'.

The idea with successive halving (which you can read more about [here](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/)  is that early in the search we use fewer 'resources' but search more candidates. 'Resources' usually refers to the number of samples used to git a learner, but _can_ be any parameter. Another sensible choice for multiple-tree-based learners (like [random forests](https://en.wikipedia.org/wiki/Random_forest) or [GBMs](https://en.wikipedia.org/wiki/Gradient_boosting)) is the number of trees fitted.

# Successive halving example

So now we know what the successive halving does, let's try it out on some real-world data. Unfortunately the examples in scikit-learn's examples use only generated data, which I find too clinical to give an indication of what a classification algorithm might be like to use in anger.

The parameter `factor` sets the ‘halving’ or fraction rate. It determines the proportion of candidates that are selected for each subsequent iteration. For example, `factor=3` means that only one third of the candidates are selected.

If we do _not_ specify a specific parameter, the default is to change the size of the sample at each halving iteration. In this way you can select a small sample across the grid space of hyperparameters that gets successively larger as you get closer to the optimum (at least theoretically).

![Illustration of successive halving](/assets/successive-halving-example.png)  
_**Example of successive halving with `factor = 2`**_

## Import the libraries we will need

As usual, I like to import a few standard libraries so I can manipulate data frames and access the underlying OS.


```python
import os

import pandas as pd
import numpy as np
```

I also like to change the configuration of `pandas` to show all the columns of a data frame.


```python
pd.set_option('display.max_columns', None)
```

## Lending club

We will use a sample of the [**Lending Club**](https://www.lendingclub.com) data. This is a view of the data I have manipulated a little bit for education purposes, and you can download it [here](https://drive.google.com/file/d/1Rahuvn8LwKPNnch5lrhKvEyQHkZGZpuu/view?usp=sharing). This data set contains details of loans, and we are interested in classifying loans that have gone 'bad' (that is, not repaid) against others.

The data is stored in CSV format, so we read it in using `pandas`.


```python
data_path = '~/repos/sklearn00/data'
lending_filename = 'loan_stats_course.csv'

lending_df = pd.read_csv(os.path.join(data_path, lending_filename))
```

Examine the data.


```python
lending_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>is_inc_v</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>purpose</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>mths_since_last_delinq</th>
      <th>mths_since_last_record</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>revol_util</th>
      <th>total_acc</th>
      <th>initial_list_status</th>
      <th>out_prncp</th>
      <th>out_prncp_inv</th>
      <th>total_pymnt</th>
      <th>total_pymnt_inv</th>
      <th>total_rec_prncp</th>
      <th>total_rec_int</th>
      <th>total_rec_late_fee</th>
      <th>recoveries</th>
      <th>collection_recovery_fee</th>
      <th>last_pymnt_amnt</th>
      <th>next_pymnt_d</th>
      <th>collections_12_mths_ex_med</th>
      <th>mths_since_last_major_derog</th>
      <th>policy_code</th>
      <th>bad_loan</th>
      <th>credit_length_in_years</th>
      <th>earned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077430</td>
      <td>1314167</td>
      <td>2500</td>
      <td>2500</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>0.0</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>car</td>
      <td>309xx</td>
      <td>GA</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1687</td>
      <td>9.4</td>
      <td>4.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1008.710000</td>
      <td>1008.71</td>
      <td>456.46</td>
      <td>435.17</td>
      <td>0.0</td>
      <td>117.08</td>
      <td>1.11</td>
      <td>119.66</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>12.0</td>
      <td>-1491.290000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077175</td>
      <td>1313524</td>
      <td>2400</td>
      <td>2400</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>small_business</td>
      <td>606xx</td>
      <td>IL</td>
      <td>8.72</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2956</td>
      <td>98.5</td>
      <td>10.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3003.653644</td>
      <td>3003.65</td>
      <td>2400.00</td>
      <td>603.65</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>649.91</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>10.0</td>
      <td>603.653644</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1071795</td>
      <td>1306957</td>
      <td>5600</td>
      <td>5600</td>
      <td>5600.0</td>
      <td>60 months</td>
      <td>21.28</td>
      <td>152.39</td>
      <td>F</td>
      <td>F2</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>OWN</td>
      <td>40000.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>small_business</td>
      <td>958xx</td>
      <td>CA</td>
      <td>5.55</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>0.0</td>
      <td>5210</td>
      <td>32.6</td>
      <td>13.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>646.020000</td>
      <td>646.02</td>
      <td>162.02</td>
      <td>294.94</td>
      <td>0.0</td>
      <td>189.06</td>
      <td>2.09</td>
      <td>152.39</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>7.0</td>
      <td>-4953.980000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1071570</td>
      <td>1306721</td>
      <td>5375</td>
      <td>5375</td>
      <td>5350.0</td>
      <td>60 months</td>
      <td>12.69</td>
      <td>121.45</td>
      <td>B</td>
      <td>B5</td>
      <td>Starbucks</td>
      <td>0.0</td>
      <td>RENT</td>
      <td>15000.0</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>other</td>
      <td>774xx</td>
      <td>TX</td>
      <td>18.08</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>9279</td>
      <td>36.5</td>
      <td>3.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1476.190000</td>
      <td>1469.34</td>
      <td>673.48</td>
      <td>533.42</td>
      <td>0.0</td>
      <td>269.29</td>
      <td>2.52</td>
      <td>121.45</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>7.0</td>
      <td>-3898.810000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1070078</td>
      <td>1305201</td>
      <td>6500</td>
      <td>6500</td>
      <td>6500.0</td>
      <td>60 months</td>
      <td>14.65</td>
      <td>153.45</td>
      <td>C</td>
      <td>C3</td>
      <td>Southwest Rural metro</td>
      <td>5.0</td>
      <td>OWN</td>
      <td>72000.0</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>debt_consolidation</td>
      <td>853xx</td>
      <td>AZ</td>
      <td>16.12</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>4032</td>
      <td>20.6</td>
      <td>23.0</td>
      <td>f</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7677.520000</td>
      <td>7677.52</td>
      <td>6500.00</td>
      <td>1177.52</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1655.54</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>13.0</td>
      <td>1177.520000</td>
    </tr>
  </tbody>
</table>
</div>



Get a list of all the columns. Later we will remove the columns we are not using and identify which colums are categorical and which are numeric.


```python
all_columns = lending_df.columns.tolist()
```

Remove some of the variables that indicate a bad loan and would represent [**target leakage**](https://en.wikipedia.org/wiki/Leakage_(machine_learning)). These columns are in the data set because Lending Club only provides a current snapshot of their loans. To avoid the leakage problem we ideally want time-stamped transactions within a relational database.


```python
to_remove = ['issue_d', 'emp_title', 'zip_code', 'earned', 'total_rec_prncp', 'recoveries', 'total_rec_int',
             'total_rec_late_fee', 'collection_recovery_fee', 'next_pyment_d', 'loan_status', 'pymnt_plan',
             'id', 'member_id', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'last_pymnt_amnt',
             'next_pymnt_d', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code',
             'credit_length_in_years']
```

Identify the columns that should be converted from string to categorical.


```python
cat_features = ['term', 'grade', 'sub_grade', 'home_ownership',
                      'is_inc_v', 'purpose', 'addr_state',
                      'initial_list_status']
```

The target column is `bad_loan`. It takes values of `1` for a bad loan and `0` for one which has not gone bad (or not gone bad _yet_ in some cases).


```python
target = 'bad_loan'

to_remove.append(target)
```

Get a list of all the predictors and numeric features.


```python
predictors = [s for s in all_columns if s not in to_remove]
```


```python
num_features = [s for s in predictors if s not in cat_features]
```

Check the shape of the data so we have an idea of how many rows we have to train and test our classifier with.


```python
lending_df.shape
```




    (36842, 49)



We have 36,842 rows to play with.

### Split into training and test sets

We will train against a separate data set from the one we will test our trained classifier against, using scikit-learn's [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) to partition the data set randomly.


```python
from sklearn.model_selection import train_test_split
```

I am going to fix the test size at 5,000; this seems like a good enough number while retaining plenty of sample to train the classifier.

**Note:** I like to specify parameters as keyword arguments. This way we can load configuration files if we want when we run from a script without having to edit the code. This is much better than having to changing values within our script for a simple change.


```python
# It would be quite easy to stick these in a file and read them in
split_config = dict(
    random_state = 1737,
    test_size = 5000
)
```


```python
X_train, X_test, y_train, y_test = train_test_split(lending_df[predictors], lending_df[target], **split_config)
```

Check that we get the right number of rows back in the test set. I like to check things are as they should be (at least in my mind) so I write tests.


```python
assert X_test.shape[0] == split_config['test_size']
```

Examine the distribution of `bad_loans` in `y_train`.


```python
y_train.value_counts()
```




    0    26693
    1     5149
    Name: bad_loan, dtype: int64



## Build a simple classifier

For this example I will use scikit-learn's [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html). I shall also use the excellent [`Pipeline`](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) capability of scikit-learn to build our classifier. This allows us to combine our preprocessing and classifier into the one composite classifier object. There are many good reasons to do this that I will not go into further here.

One I will mention is that we cab perform a search over preprocessing parameters if we like, though I am not doing that here.

To encode the categorical variables, we will use [`CatBoostEncoder`](http://contrib.scikit-learn.org/category_encoders/catboost.html) from [`category_encoders`](http://contrib.scikit-learn.org/category_encoders/), which you can install using `pip`.

```bash
pip install category_encoders
```


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

import category_encoders as ce
from sklearn.impute import SimpleImputer
```

### Set up the pipelines

The pipeline we will set up has two steps.

1. The preprocessor, which will use `CatBoostEncoder` for categorical features and [`SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) for numeric
   features.

2. A `GradientBoostingClassifier` as the model.

We combine the two preprocessors using [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html), which uses takes a tuple of the format `(name, transformer, columns)`.


```python
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', ce.CatBoostEncoder(), cat_features),
        ('numeric', SimpleImputer(), num_features)
    ]
)
```


```python
check_array = preprocessor.fit_transform(X_train, y_train)
```


```python
assert check_array.shape[1] == len(predictors)
```


```python
feature_names = cat_features + num_features
```

Check whether `check_array` contains `NaN`s.


```python
assert not(np.isnan(check_array).any())
```

Create and test our classifier. Remember, this will become a step in our pipeline.


```python
gbm = GradientBoostingClassifier()
```

Here I test if the `fit` method runs using a `try … except` construct.


```python
try:
    gbm.fit(check_array, y_train)
except:
    assert False, 'Bad gbm fit'
```

Now let us create a pipeline that includes our GBM. We name each step of the pipeline so we can refer to parameters in each of the steps.


```python
gbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('gbm', gbm)
])
```

We can refer to parameters in a pipeline by using the format **`name`**`__`**`parameter`**.


```python
param_grid = {
    'gbm__n_estimators' : 20,
    'gbm__max_depth' : 1,
    'gbm__min_samples_leaf': 30,
    'gbm__learning_rate' : 0.15,
    'gbm__subsample' : 1.0
}
```

To set a parameter for a one-off fit, we use the `set_params` method. Here we are setting the parameters to get a pretty-good model using default parameters. We will compare the results from this with those from successive halving later.


```python
gbm_pipeline.set_params(**param_grid)
```




    Pipeline(steps=[('preprocessor',
                     ColumnTransformer(transformers=[('categorical',
                                                      CatBoostEncoder(),
                                                      ['term', 'grade', 'sub_grade',
                                                       'home_ownership', 'is_inc_v',
                                                       'purpose', 'addr_state',
                                                       'initial_list_status']),
                                                     ('numeric', SimpleImputer(),
                                                      ['loan_amnt', 'funded_amnt',
                                                       'funded_amnt_inv',
                                                       'int_rate', 'installment',
                                                       'emp_length', 'annual_inc',
                                                       'dti', 'delinq_2yrs',
                                                       'inq_last_6mths',
                                                       'mths_since_last_delinq',
                                                       'mths_since_last_record',
                                                       'open_acc', 'pub_rec',
                                                       'revol_bal', 'revol_util',
                                                       'total_acc'])])),
                    ('gbm',
                     GradientBoostingClassifier(learning_rate=0.15, max_depth=1,
                                                min_samples_leaf=30,
                                                n_estimators=20))])



Test that it runs.


```python
try:
    gbm_pipeline.fit(X_train, y_train)
except:
    assert False, 'Bad pipeline fit'
```

### Check the performance of the basic model

To give a baseline of the performance of our basic model that we fit in the previous step, we calculate the [area under the receiver operating characteristic curve (ROC)](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).


```python
y_pred = gbm_pipeline.predict_proba(X_test)[:, 1]
```


```python
baseline_auc = roc_auc_score(y_test, y_pred)
print(f'The baseline AUC performance is {baseline_auc:.3f}.')
```

    The baseline AUC performance is 0.696.


This model is one we expect to be okay, if not spectacular.

## Test successive halving

Successive halving is still an experimental feature, so we need to turn it on explicitly.


```python
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv  

# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

import scipy
```

Check how many rows we have in our training set.


```python
X_train.shape[0]
```




    31842



We have 31,842 rows in our training data set, so we will set the starting number of minimum resources to 256. This will double at each iteration until we hit the limit.

Set the parameter grid for the search.


```python
param_grid = {
    'gbm__n_estimators' : scipy.stats.randint(20, 2001),
    'gbm__n_iter_no_change': [50],
    'gbm__max_depth' : scipy.stats.randint(1, 13),
    'gbm__min_samples_leaf': scipy.stats.randint(1, 51),
    'gbm__learning_rate' : scipy.stats.uniform(0.01, 0.5),
    'gbm__subsample' : [1.0]
}
```

Set some configurable parameters for the [`HalvingRandomSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html) we will run. We will use the number of samples as the resource and use AUC (area under the curve) to evaluate the best model.


```python
# config
hrs_params = dict(verbose=1, cv=5, factor=2, resource='n_samples', min_resources=256, scoring='roc_auc',
                  random_state=1707)
```

I like to note how much time potentially long-running cells take to execute. Do this easily in Jupyter with the [`%%time` cell magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html).


```python
%%time
np.random.seed(hrs_params['random_state'])
sh = HalvingRandomSearchCV(gbm_pipeline, param_grid, **hrs_params)
sh.fit(X_train, y_train)
```

    n_iterations: 7
    n_required_iterations: 7
    n_possible_iterations: 7
    min_resources_: 256
    max_resources_: 31842
    aggressive_elimination: False
    factor: 2
    ----------
    iter: 0
    n_candidates: 124
    n_resources: 256
    Fitting 5 folds for each of 124 candidates, totalling 620 fits
    ----------
    iter: 1
    n_candidates: 62
    n_resources: 512
    Fitting 5 folds for each of 62 candidates, totalling 310 fits
    ----------
    iter: 2
    n_candidates: 31
    n_resources: 1024
    Fitting 5 folds for each of 31 candidates, totalling 155 fits
    ----------
    iter: 3
    n_candidates: 16
    n_resources: 2048
    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    ----------
    iter: 4
    n_candidates: 8
    n_resources: 4096
    Fitting 5 folds for each of 8 candidates, totalling 40 fits
    ----------
    iter: 5
    n_candidates: 4
    n_resources: 8192
    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    ----------
    iter: 6
    n_candidates: 2
    n_resources: 16384
    Fitting 5 folds for each of 2 candidates, totalling 10 fits
    CPU times: user 7min 40s, sys: 1.19 s, total: 7min 41s
    Wall time: 7min 42s





    HalvingRandomSearchCV(estimator=Pipeline(steps=[('preprocessor',
                                                     ColumnTransformer(transformers=[('categorical',
                                                                                      CatBoostEncoder(),
                                                                                      ['term',
                                                                                       'grade',
                                                                                       'sub_grade',
                                                                                       'home_ownership',
                                                                                       'is_inc_v',
                                                                                       'purpose',
                                                                                       'addr_state',
                                                                                       'initial_list_status']),
                                                                                     ('numeric',
                                                                                      SimpleImputer(),
                                                                                      ['loan_amnt',
                                                                                       'funded_amnt',
                                                                                       'funded_amnt_inv',
                                                                                       'int_rate',
                                                                                       'installment',
                                                                                       'emp_length',
                                                                                       'annual_...
                                               'gbm__max_depth': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fa3353f3910>,
                                               'gbm__min_samples_leaf': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fa3354bad00>,
                                               'gbm__n_estimators': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7fa3354ba910>,
                                               'gbm__n_iter_no_change': [50],
                                               'gbm__subsample': [1.0]},
                          random_state=1707,
                          refit=<function _refit_callable at 0x7fa3335f0280>,
                          scoring='roc_auc', verbose=1)



This took 6 iterations and around 8 minutes on my machine.

Note that at each step the number of candidates halved until we got to two in the final step.

Once successive halving has settled on a final candidate, it fits it using the whole sample.

Examine the parameters of the best and final estimator.


```python
sh.best_estimator_
```




    Pipeline(steps=[('preprocessor',
                     ColumnTransformer(transformers=[('categorical',
                                                      CatBoostEncoder(),
                                                      ['term', 'grade', 'sub_grade',
                                                       'home_ownership', 'is_inc_v',
                                                       'purpose', 'addr_state',
                                                       'initial_list_status']),
                                                     ('numeric', SimpleImputer(),
                                                      ['loan_amnt', 'funded_amnt',
                                                       'funded_amnt_inv',
                                                       'int_rate', 'installment',
                                                       'emp_length', 'annual_inc',
                                                       'dti', 'delinq_2yrs',
                                                       'inq_last_6mths',
                                                       'mths_since_last_delinq',
                                                       'mths_since_last_record',
                                                       'open_acc', 'pub_rec',
                                                       'revol_bal', 'revol_util',
                                                       'total_acc'])])),
                    ('gbm',
                     GradientBoostingClassifier(learning_rate=0.37976865555958783,
                                                max_depth=1, min_samples_leaf=16,
                                                n_estimators=1500,
                                                n_iter_no_change=50))])



We can check how the scoring progressed in each iteration. Recall our baseline AUC score is 0.696.


```python
res_df = pd.DataFrame(sh.cv_results_)
res_df.groupby('iter')['mean_test_score'].max()
```




    iter
    0    0.618091
    1    0.652617
    2    0.648898
    3    0.695115
    4    0.717969
    5    0.725628
    6    0.727079
    Name: mean_test_score, dtype: float64



According to the cross-validation AUC score on the _training_ set, successive halving produced a better model than the naive GBM after iteration 4.

Check the performance of the final model against the test set. This will show us whether the performance has improved compared with the initial baseline model we fit against the same test set.


```python
preds_test = sh.best_estimator_.predict_proba(X_test)[:,1]
baseline_auc = roc_auc_score(y_test, y_pred)

rsh_auc = roc_auc_score(y_test, preds_test)
rsh_auc
```




    0.7233991664802961



We got an AUC score of 0.723 which is better than 0.696 — a definite improvement.

# Conclusion

The successive halving approach is a reasonable algorithm that does the same sort of thing that I advise learning data scientists to do. That is, perform searches using smaller samples at first to cut down processing time. Once we have a good idea of the best parameters, fit against a large sample to improve the precision of the estimators.

Of course, one could argue the position that the parameters of the successive halving are additional hyperparameters that need to be optimised on. But that would send you down the **rabbit hole** and distract you from this fact: for most purposes, a good model is not significantly better than the best model. Your time is usually better spent ensuring the entire problem solving process, end-to-end pipeline and experimental design are also 'good enough'.
