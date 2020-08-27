# explore if categorical features are indeed picked up in model.
library(AmesHousing)

# data cleaning
library(janitor)

# data prep
library(dplyr)

# tidymodels
library(rsample)
library(recipes)

# model packages
library(lightgbm)
library(catboost)

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

# get dataset
train_set <- recipes::bake(
    preprocessing_recipe, 
    new_data = training(ames_split)
) 
test_set <- recipes::bake(
    preprocessing_recipe, 
    new_data = testing(ames_split)
)

get_categorical_column_numbers <- function(dataset){
    vars = dataset %>% select(all_nominal()) %>% names()
    which(names(dataset) %in% vars)
}
categorical_features <- get_categorical_column_numbers(train_set)
## make lgb data
lgb_data <- lgb.Dataset(
    data = train_set %>% select(-sale_price) %>% as.matrix()
    , label = train_set$sale_price
    , categorical_feature = categorical_features,
    feature_pre_filter = FALSE
)
ctbst_data <- catboost.load_pool(
    data = train_set %>% select(-sale_price),
    label = train_set$sale_price
)
ctbst_data2 <- catboost.load_pool(
    data = train_set %>% select(-sale_price),
    label = train_set$sale_price,
    cat_features = categorical_features
)

# train quick lgbm model on it
lgbm_params <- list(
    objective = "regression"
    , min_data = 1L
    , learning_rate = 0.01
    , min_data = 1L
    , min_hessian = 1.0
    , max_depth = 2L
)
## without specifying categorical
lgb_mod_1 <- lgb.train(data = lgb_data,params = lgbm_params, verbose = 3, num_threads=1)
## with specifying categorical
lgb_mod_2 <- lgb.train(params= lgbm_params, data = lgb_data,categorical_feature = categorical_features, verbose = 3)
# train quick catboost model on it
ctbst_params <- list(
    
)
## without specifying categorical
ctbst_mod_1 <- catboost.train(learn_pool = ctbst_data,params = list(logging_level="Verbose", thread_count = 4))
## with specifying categorical
ctbst_mod_2 <- catboost.train(learn_pool = ctbst_data2,params= list(logging_level="Verbose", thread_count=4)) 

# inspect the models
lgb_mod_1_importance <- lgb.importance(lgb_mod_1, percentage=TRUE)
lgb_mod_2_importance <- lgb.importance(lgb_mod_2, percentage=TRUE)
lgb.plot.importance(lgb_mod_1_importance, top_n = 5L, measure = "Gain")
lgb.interprete(lgb_mod_1,data = as.matrix(test_set %>% select(-sale_price)),idxset = 1L:5L)
# see if you can train model on data without specifying categorical in data 

#  loading step.
