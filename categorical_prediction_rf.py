# categorical prediction with sklearn and random forest
# <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>

import numpy as np

# imports ----
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

## Split file into training and test set ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

## FEATURE engineering ----
median_imputer = SimpleImputer(missing_values=np.nan, strategy="median")
## model specification
classifier = RandomForestClassifier(max_depth=2, random_state=0)

## combine feature engineering and model
# <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline
clf = Pipeline([("median_imputer", median_imputer), ("classification", classifier)])
## fit on trainingset
clf.fit(X_train, y_train)
## evaluate on testset
clf.predict(X_test)
