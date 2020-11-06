#' Learning curve
#'
#' A method described by Andrew Ng in Coursera Machine Learning videos.
#' Train model on a few points, predict add a few points, predict again
#' all the way to the total size of training set.
#' Plot train error and test error over sample size. These plots can
#' indicate bias or variance.
#' High bias: errors are high, not even good performance on the training set.
#' Almost the same errors on training and cv set. More data does not really help.
#' High variance: low train error. High CV error.
#'
#' ## Details
#' We're reusing the infrastructure created by rsample package here, because
#' frankly it is very elegant and works incredibly well.
#' rsample makes use of the fact that a dataset will only be copied if you
#' modify it. Rsample objects contain the same dataset and indices of the
#' rows we will use. That way there will be no unneccessary copies of data.

set_max_s_tr <- function(max_data_size = NULL, trainingsetsize) {
  if (is.null(max_data_size)) {
    max_s_tr <- trainingsetsize
  } else {
    max_s_tr <- min(trainingsetsize, max_data_size)
  }
  max_s_tr
}


###
sample_sizes <- function(start_size, length_vec, steps) {
  indices <- seq.int(
    start_size,
    length_vec,
    by = as.integer((length_vec - start_size) / (steps - 1))
  )
  sample(indices, size = steps, replace = FALSE) # hack
}

sample_from_ind <- function(sample_size, idx) {
  sample(x = idx, size = sample_size, replace = FALSE)
}

create_rsample_obj <- function(analysis_, assessment_, data, class = NULL) {
  res <- structure(
    list(
      data = data,
      in_id = analysis_,
      out_id = assessment_
    ),
    class = "rsplit"
  )
  if (!is.null(class)) {
    res <- rsample:::add_class(res, class)
  }
  res
}


print_sizes <- function(split_obj) {
  in_l <- length(split_obj$in_id)
  out_l <- length(split_obj$out_id)
  paste0("in:", in_l, "/out:", out_l)
}

training_size <- function(split_obj) {
  length(split_obj$in_id)
}


#' Create incremental dataset
#'
#' Create 'learning curve' dataset. with incrementally
#' larger training and validation sets.
#' @export
incremental_set <- function(dataset, max_steps = 15, min_data_size = NULL, max_data_size = NULL, prop = 3 / 4, strata = NULL) {
  # We use initial split indices for train and test set to split the rows
  # into train and test indices.
  splitted <- rsample::initial_split(dataset, prop = prop, strata = strata)
  max_s_tr <- set_max_s_tr(max_data_size = max_data_size, trainingsetsize = training_size(splitted))
  max_s_test <- nrow(dataset) - max_s_tr
  min_data_size <- ifelse(is.null(min_data_size), 1, min_data_size)
  max_steps <- ifelse(max_steps > max_s_test,
    {
      max_steps <- max_s_test
      warning(paste0("max_steps reset to ", max_steps))
    },
    max_steps
  )

  full_train_idx <- splitted$in_id
  full_test_idx <- rsample::complement(splitted)
  sample_sizes_test <- incremental_indices(min_data_size, max_s_test, max_steps)
  sample_sizes_tr <- incremental_indices(min_data_size, max_s_tr, max_steps)
  analysis <- purrr::map(sample_sizes_tr, subset_idx, idx = full_train_idx)
  assessment <- purrr::map(sample_sizes_test, subset_idx, idx = full_test_idx)
  
  # sample_sizes_test <- sample_sizes(start_size = min_data_size, max_s_test, max_steps)
  # sample_sizes_tr <- sample_sizes(start_size = min_data_size, max_s_tr, length(sample_sizes_test))
  # analysis <- purrr::map(sample_sizes_tr, sample_from_ind, idx = full_train_idx)
  # assessment <- purrr::map(sample_sizes_test, sample_from_ind, idx = full_test_idx)

  split_objs <- purrr::map2(analysis, assessment, create_rsample_obj, data = dataset, class = "incremental_splits")
  tibble::tibble(
    splits = split_objs,
    id = rsample:::names0(length(split_objs), "Increment")
  )
}

## SOMETHING IS WRONG, BECAUSE THE LAST ROW IS WAY SMALLER

### test, sort of
# iris2 <- incremental_set(iris,min_data_size = 10)
# iris2$splits[[1]]$in_id


## autoplot method?


### Extract indices turn into dataframe

make_dframe <- function(indices, type = c("training", "test"), incrementlabel) {
  data.frame(
    stringsAsFactors = FALSE,
    ind = indices,
    type = type,
    label = incrementlabel
  )
}

indice_per_row <- function(dfrow) {
  stopifnot(nrow(dfrow) == 1)
  dplyr::bind_rows(
    make_dframe(dfrow$splits[[1]]$in_id, "training", dfrow$id),
    make_dframe(dfrow$splits[[1]]$out_id, "test", dfrow$id)
  )
}
# indice_per_row(iris2[11,])


# a function that takes a min, a max, total_length, steps
# (capture that at the top, this is not a user facing function)
#  if(is.null(max)) max <- total_length
#  # total lenght, steps and min cannot be empty, steps must be integer
#  indices <- ifelse(
# max < total_length,
# sample(1:total_length, size = max, replace = FALSE),
# 1:total_length
# )
# capture if steps are possible at top
incremental_indices <- function(min , max, steps) {
    stepsize <- max(1,floor((max-min)/steps))
    #length(seq(from=min,to=max,by=stepsize))
    incremental_indices<- list()
    for (step in seq_len(steps)) {
        incremental_indices[[step]] <- seq_builder(step, stepsize, min)
    }
    incremental_indices
}

incremental_indices(1,10, 10) # should be step by step
incremental_indices(1,10, 8) # better to have exactly the number of steps than
# full coverage.

seq_builder <- function(step, stepsize, min){
    1:( stepsize*(step-1) + min)
}
seq_builder(1,1,1) ==1
all(seq_builder(2,1,1)== c(1,2))
all(seq_builder(1,1,3)== c(1,2,3))
all(seq_builder(2,1,3)== c(1,2,3,4))
all(seq_builder(2,1,3)== c(1,2,3,4))
seq_builder(6,2,1)


subset_idx <- function(idx, seq){
    idx[seq]
}
subset_idx(c(99, 80,70,60), c(1,2,3)) == c(99,80,70)
a <- list(a =c(1,2,3))
purrr::map(a, subset_idx, idx = c(99, 80,70,60)) # $a = 99,80, 70
