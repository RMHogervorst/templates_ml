# A generic template for tidymodels crossvalidation
# based on https://www.tychobra.com/posts/2020-05-19-xgboost-with-tidymodels/
# 
# Split data into train and test. 
# Do feature engineering 
# Create cross validation folds
# Setup model and tuneable parameters
# Use cv_folds to determine the best parameters
# retrain the model on train set with best parameters
# predict on test set, report performance.
# =====================  Libraries & scripts ====================
library(dplyr)
library(readr)
library(ggplot2)

# tidymodels
library(rsample)
library(recipes)
library(parsnip)
library(tune)
library(dials)
library(workflows)
library(yardstick)
library(collapse)

# ===================== Load data ===============================
dataset <- read_csv("")

# =====================  Set up parallel ========================
# optional
library(doParallel)
all_cores <- parallel::detectCores(logical = TRUE) 
registerDoParallel(cores = all_cores)  
 

# 
# ============ Split into train and test set ====================
training_splits <- rsample::initial_split(
    dataset, 
    prop = 0.8, 
    strata = yvariable
)
# =====================   Applying learning curve  ==============
source("R/learning_curve.R")
# Create incremental set to see if your model is fitting.
incr_train <- incremental_set(training(training_splits), 10,min_data_size = 30)
# ===============================================================



# ===================  Feature engineering  =====================
# recipe on only the training data
preprocessing_recipe <- 
    recipe(sale_price~., data = training(training_splits)) %>% 
    ## convert categorical variables to factors
    #recipes::step_string2factor(all_nominal()) %>%
    ## combine low frequency factor levels
    #recipes::step_other(all_nominal(), threshold = 0.01) %>%
    ## remove no variance predictors which provide no predictive information 
    #recipes::step_nzv(all_nominal()) %>%
    ## transform y-variable with YeoJohnson.
    #step_YeoJohnson()
    ## transform data with collapse, lags, leads, etc.
    # fgroup_by(Variable, Country) %>% fdiff(c(1, 10), 1:2, Year)
    ## We do the preprocessing on the entire trainingset
    prep()
    
# =================  Cross validation ==========================
cv_folds <- 
    recipes::bake(
        preprocessing_recipe, 
        new_data = training(training_splits)
    ) %>%  
    rsample::vfold_cv(v = 5)

# ========================= model setup =========================
# model specification
lightgbm_model<- 
    parsnip::boost_tree(
        mode = "regression",
        trees = 1000,
        min_n = tune(),
        tree_depth = tune(),
        learn_rate = tune(),
        loss_reduction = tune()
    ) %>%
    set_engine("lightgbm", objective = "reg:squarederror")

#   flexible model parameters
lightgbm_params <- 
    dials::parameters(
        min_n(),
        tree_depth(),
        learn_rate(),
        loss_reduction()
    )
#    model grid
lgbm_grid <- 
    dials::grid_max_entropy(
        lightgbm_params, 
        size = 20
    )
# Workflow
lgbm_wf <- 
    workflows::workflow() %>%
    add_model(lightgbm_model) %>% 
    add_formula(sale_price ~ .)

# ============== hyper parameter tuning over cv folds  =========
## THIS IS THE TIME CONSUMING PART
lgbm_tuned <- tune::tune_grid(
    object = lgbm_wf,
    resamples = cv_folds,
    grid = lgbm_grid,
    metrics = yardstick::metric_set(rmse, rsq, mae),
    control = tune::control_grid(verbose = TRUE)
)

# Best performant hyper parameters
lgbm_tuned %>%
    tune::show_best(metric = "rmse") 

# Next, isolate the best performing hyperparameter values.
lgbm_best_params <- lgbm_tuned %>%
    tune::select_best("rmse")
lgbm_best_params

# Finalize the lgbm model to use the best tuning parameters.

lgbm_model_final <- lightgbm_model%>% 
    finalize_model(lgbm_best_params)

# ==========  Evaluate performance ============================
# create train and test set
train_processed <- bake(preprocessing_recipe,  new_data = training(training_splits))
test_processed  <- bake(preprocessing_recipe, new_data = testing(training_splits))
# fit model on entire trainset
trained_model_all_data <- lgbm_model_final %>%
    # fit the model on all the training data
    fit(
        formula = sale_price ~ ., 
        data    = train_processed
    )
# predict on trainset
train_prediction <- 
    trained_model_all_data %>% 
    predict(new_data = train_processed) %>%
    bind_cols(training(training_splits))



# predict on testset
test_prediction <- 
    trained_model_all_data %>% 
    # use the training model fit to predict the test data
    predict(new_data = test_processed) %>%
    bind_cols(testing(training_splits))

# measure the accuracy of our model using `yardstick`
lgbm_score_train <- 
    train_prediction %>%
    yardstick::metrics(sale_price, .pred)
lgbm_score_train
lgbm_score_test <- 
    test_prediction %>%
    yardstick::metrics(sale_price, .pred)
lgbm_score_test

## Check residuals to see if nothing obvious is wrong
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
