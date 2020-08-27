# Variable exploration template ----
# load libraries
library(tidyverse)
library(janitor)
library(corrplot)
library(skimr)
library(GGally)
library(lubridate)
library(ggExtra) # ggMarginal
library(naniar)
# load scripts -------

## Description of problem, and first ideas --------

# Load data  ----------------

# Distribution of target / label ----------

# overview of data -------
skimr::skim(training_data)

# variation
## Don't put too many variables in this massive plot.
ggpairs()

# missings -------------
# https://cran.r-project.org/web/packages/naniar/vignettes/naniar-visualisation.html
# overview of missings
vis_miss(airquality)
# 
ggplot(airquality,
       aes(x = Ozone,
           y = Solar.R)) +
    geom_miss_point() + 
    facet_wrap(~Month)
# relations between missingness
gg_miss_upset(train_data)
# correlation  ------------
## select the numeric variables and run correlations
corrplot(cor(tmp, use="complete.obs"),type="lower")
# 

# Probable actions to undertake -------------
## mark missings / impute them
## Most probable strong candidates
## transformations to try out
