# generic lightgbm script
# from https://www.tychobra.com/posts/2020-05-19-xgboost-with-tidymodels/
# Their design and description is very sound and I iterate on their work.
# data
library(AmesHousing)

# data cleaning
library(janitor)

# data prep
library(dplyr)

# tidymodels
library(rsample)
library(recipes)
library(parsnip)
library(tune)
library(dials)
library(workflows)
library(yardstick)
library(treesnip)
# need this too later.
library(ggplot2)

# speed up computation with parrallel processing (optional)
library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE) 
# in xgboost the cores were not optimally used on my mac, but lgbm is filling
# it tot the brim
registerDoParallel(cores = all_cores) 

# set the random seed so we can reproduce any simulated results.
set.seed(1234)

# load the housing data and clean names
ames_data <- make_ames() %>%
    janitor::clean_names()

# split into training and testing datasets. Stratify by Sale price 
ames_split <- rsample::initial_split(
    ames_data, 
    prop = 0.8, 
    strata = sale_price
)

# Pre processing 
preprocessing_recipe <- 
    recipes::recipe(sale_price ~ ., data = training(ames_split)) %>%
    # convert categorical variables to factors
    #recipes::step_string2factor(all_nominal()) %>%
    # combine low frequency factor levels
    recipes::step_other(all_nominal(), threshold = 0.01) %>%
    # remove no variance predictors which provide no predictive information 
    recipes::step_nzv(all_nominal()) %>%
    prep()

# Cross validate 
ames_cv_folds <- 
    recipes::bake(
        preprocessing_recipe, 
        new_data = training(ames_split)
    ) %>%  
    rsample::vfold_cv(v = 5)

## /// changing from xgboost to catboost
# catboost model specification
catboost_model <- 
    parsnip::boost_tree(
        mode = "regression",
        trees = 1000,
        min_n = tune(),
        tree_depth = tune(),
        learn_rate = tune()
        # removed loss reduction here, because that is not in catboost
    ) %>%
    set_engine("catboost",  loss_function = "squarederror") 
    # had to change objective: reg:squarederror to loss_function = "squarederror"

# ///grid specification by dials package to fill in the model above
# grid specification
catboost_params <- 
    dials::parameters(
        min_n(),
        tree_depth(),
        learn_rate()
    )

# ///and the grid to look in 
# Experimental designs for computer experiments are used
# to construct parameter grids that try to cover the parameter space such that
# any portion of the space has an observed combination that is not too far from
# it.
catboost_grid <- 
    dials::grid_max_entropy(
        catboost_params, 
        size = 20
    )
# To tune our model, we perform grid search over our xgboost_grid’s grid space
# to identify the hyperparameter values that have the lowest prediction error.

# Workflow setup
# /// (contains the work)
catboost_wf <- 
    workflows::workflow() %>%
    add_model(catboost_model
             ) %>% 
    add_formula(sale_price ~ .)

# /// so far little to no computation has been performed except for
# /// preprocessing calculations

# Step 7: Tune the Model

# Tuning is where the tidymodels ecosystem of packages really comes together.
# Here is a quick breakdown of the objects passed to the first 4 arguments of
# our call to tune_grid() below:
#
# “object”: xgboost_wf which is a workflow that we defined by the parsnip and
# workflows packages “resamples”: ames_cv_folds as defined by rsample and
# recipes packages “grid”: xgboost_grid our grid space as defined by the dials
# package “metric”: the yardstick package defines the metric set used to
# evaluate model performance
# 
# hyperparameter tuning
# //// this is where the machine starts to smoke!
catboost_tuned <- tune::tune_grid(
    object = catboost_wf,
    resamples = ames_cv_folds,
    grid = catboost_grid,
    metrics = yardstick::metric_set(rmse, rsq, mae),
    control = tune::control_grid(verbose = TRUE)
)

# In the above code block tune_grid() performed grid search over all our 60 grid
# parameter combinations defined in xgboost_grid and used 5 fold cross
# validation along with rmse (Root Mean Squared Error), rsq (R Squared), and mae
# (Mean Absolute Error) to measure prediction accuracy. So our tidymodels tuning
# just fit 60 X 5 = 300 XGBoost models each with 1,000 trees all in search of
# the optimal hyperparameters. Don’t try that on your TI-83! 
# /// this is just taking way too long! set it back to 30

# These are the
# hyperparameter values which performed best at minimizing RMSE.
catboost_tuned %>%
    tune::show_best(metric = "rmse") 

# Next, isolate the best performing hyperparameter values.
catboost_best_params <- catboost_tuned %>%
    tune::select_best("rmse")
catboost_best_params

# Finalize the lgbm model to use the best tuning parameters.

catboost_model_final <- catboost_model%>% 
    finalize_model(catboost_best_params)

# Evaluate Performance on Test Data We use the rmse (Root Mean Squared Error),
# rsq (R Squared), and mae (Mean Absolute Value) metrics from the yardstick
# package in our model evaluation.

# First let’s evaluate the metrics on the training data

train_processed <- bake(preprocessing_recipe,  new_data = training(ames_split))

final_model <- catboost_model_final %>%
    # fit the model on all the training data
    fit(
        formula = sale_price ~ ., 
        data    = train_processed
    ) 
train_prediction <- final_model %>% 
    # predict the sale prices for the training data
    predict(new_data = train_processed) %>%
    bind_cols(training(ames_split))

catboost_score_train <- 
    train_prediction %>%
    yardstick::metrics(sale_price, .pred) %>%
    mutate(.estimate = format(round(.estimate, 2), big.mark = ","))

catboost_score_train

# And now for the test data:
    
test_processed  <- bake(preprocessing_recipe, new_data = testing(ames_split))

test_prediction <- final_model %>%
    # use the training model fit to predict the test data
    predict(new_data = test_processed) %>%
    bind_cols(testing(ames_split))

# measure the accuracy of our model using `yardstick`
catboost_score_test <- 
    test_prediction %>%
    yardstick::metrics(sale_price, .pred) %>%
    mutate(.estimate = format(round(.estimate, 2), big.mark = ","))
catboost_score_test

# To quickly check that there is not an obvious issue with our model’s
# predictions, let’s plot the test data residuals.

house_prediction_residual <- test_prediction %>%
    arrange(.pred) %>%
    mutate(residual_pct = (sale_price - .pred) / .pred) %>%
    select(.pred, residual_pct)

ggplot(house_prediction_residual, aes(x = .pred, y = residual_pct)) +
    geom_point() +
    xlab("Predicted Sale Price") +
    ylab("Residual (%)") +
    scale_x_continuous(labels = scales::dollar_format()) +
    scale_y_continuous(labels = scales::percent)


# ///looks good enough to me, not a trend in residuals.
