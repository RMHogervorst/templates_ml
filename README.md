# Templates for machine learning projects

This is a set of templates for machine learning projects. 
This is a work in progress, and def. not a final or best version.

| example | python | R|
|---------|--------|--|
|simple categorical  | |[categorical_prediction_rf.R](categorical_prediction_rf.R)|
|simple categorical with lightgbm | |[categorical_prediction_lgbm.R](categorical_prediction_lgbm.R)|
|cross validation | |[cross_validation](cross_validation.R)|
|hyperparameter tuning | |[hyperparameter_tuning](hyperparameter_tuning.R)|


# Comparison R and Python templates
The R language was designed with data analysis in mind and when ML came on the scene it flowed nicely into the language. Python is a general purpose language with ML and some stats packages bolted on. Python is super readable even for people who never program in python. R has data.frames, missing values and statistical functionality built in, but it looks a bit weird. 

## Catagorical data
R has support for categorical data (factors), and tree based models like lightGBM and random forests can make efficient use of that. In python, it is possible to use the 'categorical' option as column and some models (like lightGBM) can make use of that. But this need attention and you have to perform work!


## Object oriented programming
Python loves to be **encapsulated** object oriented: methods are part of objects. and objects need to be activated/initialized.
You call a fit function from a sklearnmodel-object. `modelobject.fit(args)`. R loves to be
**functional** object oriented: methods belong to generic functions and a call looks like a normal function `fit(modelobject, args)`. 


In practice that makes the code look and feel different.

So in python (sklearn) you import a modelobject and manipulate that, the modelobject keeps state.

```python
# instantiate RandomForest classifier instance
clf = RandomForestClassifier(max_depth=2, random_state=0)
# fit that instance with features (X) and results (y)
clf.fit(X, y)
# the clf object contains the trained model now.
# use that object to predict new values.
clf.predict([[0, 0, 0, 0]]
```

in R (tidymodels) you setup a model and write the result to a new object,
and if you want, you can pipe the steps after each other even. 

```R
# instantiate a random forest model
rf_mod <-
  rand_forest(trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger") %>% 
# train the model (rf_mod), 
# with species as target 
# and all the other variables as features
# use the trainingset
trained_model <- fit(rf_mod, species~., trainingset)
# the trained_model now contains enough information to predict new data
predictions <- predict(trained_model, testset)
```


## Concepts in different languages
What is it called ( or Help me Search)? [(*see also this excellent article by tim mastny*)](https://timmastny.com/blog/tuning-and-cross-validation-with-tidymodels-and-scikit-learn/)

| Concept                                      | Python (sklearn)                                                                                                | R (tidymodels)                                                            |
|------------------|----------------------------------|--------------------|
| Combine feature engineering & modeling steps | Pipeline                                                                                                        | workflow                                                                  |
| split data into training and test set        | `train_test_split()`                                                                                            | `initial_split()`                                                         |
| feature engineering                          | `sklearn.preprocessing`                                                                                         | `recipes::step_*` functions                                               |
| tuning                                       | create a dictionary yourself with tunable hyperparameters `{"decisiontreeclassifier__max_depth":[1, 4, 8, 11]}` | `dials::grid_*` functions. (max_entropy, latin_hypercube, random, regular |
| cross-validation                             | `from sklearn.model_selection import GridSearchCV`                                                              | `vfold_cv()`                                                              |
