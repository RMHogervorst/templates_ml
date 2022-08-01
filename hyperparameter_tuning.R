# tuning your model for best performance
# 
# <https://www.tidymodels.org/start/tuning/>
# 

# Give the model a set of values to try.

# provide model with tune-able options ----
tune_spec <- 
    decision_tree(
        cost_complexity = tune(),
        tree_depth = tune()
    ) %>% 
    set_engine("rpart") %>% 
    set_mode("classification")

# supply values to try ---
# The {dials} package chooses sensible values to try for each hyperparameter.
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5)

# use cross validation folds to determine optimal hyperparameters
set.seed(234)
cell_folds <- vfold_cv(cell_train)

# combine tuneable model and preprocessing into workflow
tuning_results <- 
    workflow_object %>% 
    tune_grid(
        resamples = cell_folds,
        grid = tree_grid
        # you can supply metric here too. 
    )

# evaluate performance ----
tuning_results  %>% 
    collect_metrics()

# use ggplot2 for example to evalute results, or use autoplot.

# pick best model ----
best_tree <- tuning_results %>%
    select_best("accuracy")

# finalize model to use in production ----
# that is: set the hyperparameters of best model in produciton model
final_wf <- 
    workflow_object %>% 
    finalize_workflow(best_tree)

# train the best parameter model on entire trainingset and evalute on
# testset 
final_fit <- 
    final_wf %>% 
    last_fit(cell_split) 
