## hyper parameter tuning
# now is using gridsearch, but other options are possible
# <https://scikit-learn.org/stable/modules/grid_search.html#grid-search>
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline(
    [("median_imputer", median_imputer), ("classification", classifier)]
)

import numpy as np

grid_params = {"rf__n_estimators": [120, 140], "rf__max_depth": [30, 50]}


clf = GridSearchCV(pipeline, grid_params)
clf.fit(X_train, y_train)
