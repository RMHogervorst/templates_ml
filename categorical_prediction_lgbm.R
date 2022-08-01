# this is a categorical prediction template
# 
# This package, bonsai enables light gradient boosted models or conditional
# inference trees as possible options.
# These are the best in class models for tabular data. They are unreasonably
# effective with very little tuning.
#  <https://bonsai.tidymodels.org/articles/bonsai.html>
#  
# libraries
library(tidymodels)
library(bonsai) # for tree based models
library(palmerpenguins) # for the dataset here

## Split file into training and test set
split_file <- initial_split(penguins,strata = "species")

## FEATURE engineering ----
# Do only on training data
# also look at Feature engineering and Selection: 
# <https://bookdown.org/max/FES/>
lgbm_recipe <-
    recipe(formula = species ~ ., data = training(split_file)) %>% 
    step_impute_knn(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g,sex,impute_with = c("island", "sex", "body_mass_g"))# impute missing values



## model specification lightgbm ----
bt_mod <- 
    boost_tree() %>%
    set_engine(engine = "lightgbm") %>%
    set_mode(mode = "classification") 

## put the recipe and model together into one workflow ----
bt_workflow <-
    workflow() %>%
    add_recipe(lgbm_recipe) %>%
    add_model(bt_mod)

## fit on training set ----
trained_lgbm_model <- fit(bt_workflow,training(split_file))

# evaluate on testset ----
predicted_classes <- predict(trained_lgbm_model, testing(split_file))

result <- testing(split_file) %>% add_column(predicted_classes)
## evaluating metrics
# see <https://yardstick.tidymodels.org/articles/metric-types.html>
class_metrics <- metric_set(accuracy, kap)
result %>% class_metrics(species, estimate=.pred_class)

