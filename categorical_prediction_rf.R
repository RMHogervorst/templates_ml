# this is a categorical prediction template
# You can also use library(usemodels) with their use_* to spit out templates for your model
#
# libraries
library(tidymodels)
library(palmerpenguins) # for the dataset here

## Split file into training and test set
split_file <- initial_split(penguins,strata = "species")

## FEATURE engineering ----
# Do only on training data
# also look at Feature engineering and Selection: 
# <https://bookdown.org/max/FES/>
ranger_recipe <-
  recipe(formula = species ~ ., data = training(split_file)) %>% 
    step_impute_knn(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g,sex,impute_with = c("island", "sex", "body_mass_g"))# impute missing values



## model specification ranger ----
ranger_spec <-
  rand_forest(trees = 1000) %>%
  set_mode("classification") %>%
  set_engine("ranger")

## put the recipe and model together into one workflow ----
ranger_workflow <-
  workflow() %>%
  add_recipe(ranger_recipe) %>%
  add_model(ranger_spec)

## fit on training set ----
trained_ranger_model <- fit(ranger_workflow,training(split_file))

# evaluate on testset ----
predicted_classes <- predict(trained_ranger_model, testing(split_file))

result <- testing(split_file) %>% add_column(predicted_classes)
## evaluating metrics
# see <https://yardstick.tidymodels.org/articles/metric-types.html>
class_metrics <- metric_set(accuracy, kap)
result %>% class_metrics(species, estimate=.pred_class)

