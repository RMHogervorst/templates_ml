# Templates for machine learning projects

This is a set of templates for machine learning projects. Some are optimized for kaggle style competitions others are more generic for working with tidymodels. This is a work in progress, and def. not a final or best version.

### Comparison R and Python templates

R has support for categorical data, and tree based models like lightGBM and random forests can make efficient use of that. In python, it is possible to use the 'categorical' option as column and some models (like lightGBM) can make use of that.

What is it called ( or Help me Search)? [(*see also this excellent article by tim mastny*)](https://timmastny.com/blog/tuning-and-cross-validation-with-tidymodels-and-scikit-learn/)

| Concept                                      | Python (sklearn)                                                                                                | R (tidymodels)                                                            |
|------------------|----------------------------------|--------------------|
| Combine feature engineering & modeling steps | Pipeline                                                                                                        | workflow                                                                  |
| split data into training and test set        | `train_test_split()`                                                                                            | `initial_split()`                                                         |
| feature engineering                          | `sklearn.preprocessing`                                                                                         | `recipes::step_*` functions                                               |
| tuning                                       | create a dictionary yourself with tunable hyperparameters `{"decisiontreeclassifier__max_depth":[1, 4, 8, 11]}` | `dials::grid_*` functions. (max_entropy, latin_hypercube, random, regular |
| cross-validation                             | `from sklearn.model_selection import GridSearchCV`                                                              | `vfold_cv()`                                                              |
