## how to use cross validation
#
# Most of this is contained in 
# <https://rsample.tidymodels.org/>
# 
# You can bootstrap, vfold crossvalidate, nested crossvalidate etc.
# https://rsample.tidymodels.org/reference/vfold_cv.html

# Create folds ----
mtcars_vfold <- vfold_cv(mtcars, v = 10)

# instead of using fit() on a workflow, use fit_resamples
# (it will fit on every analysis set in the fold and predict 
# on the assessmentset in that that fold)
vfold_results <- 
    workflow_object() %>% 
    fit_resamples(mtcars_vfold)

collect_metrics(vfold_results)
