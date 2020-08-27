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



#' Create incremental dataset
#'
#' Create 'learning curve' dataset. with incrementally
#' larger training and validation sets.
#' @export
incremental_set <- function(dataset,max_steps=15, min_data_size = NULL, max_data_size=NULL){
    splitted <- rsample::initial_split(dataset)
    if(is.null(max_data_size)){
        max_s_tr = length(splitted$in_id)
    }else {
        max_s_tr = min(length(splitted$in_id), max_data_size)
    }
    max_s_test = nrow(dataset)-max_s_tr
    if(is.null(min_data_size)) min_data_size <- 1

    if(max_steps > max_s_test){
        max_steps <- max_s_test
        warning(paste0("max_steps reset to ",max_steps))
    }
    ind <- list()
    full_train_idx = splitted$in_id
    full_test_idx = rsample::complement(splitted)
    sample_sizes_test <- sample_sizes(start_size = min_data_size, max_s_test, max_steps)
    sample_sizes_tr <- sample_sizes(start_size = min_data_size, max_s_tr, length(sample_sizes_test))
    ind$analysis <- purrr::map(sample_sizes_tr, sample_from_ind, idx=full_train_idx)
    ind$assessment <- purrr::map(sample_sizes_test, sample_from_ind, idx=full_test_idx)

    split_objs <- purrr::map2(ind$analysis, ind$assessment, create_rsample_obj, data = dataset, class = "incremental_splits")
    tibble::tibble(splits = split_objs,
                   id = rsample:::names0(length(split_objs), "Increment"))
}

###
sample_sizes <- function(start_size, length_vec, steps){
    seq.int(start_size, length_vec, by= as.integer((length_vec - start_size)/(steps - 1)))
}

sample_from_ind <- function(sample_size, idx){
    sample(x=idx,size=sample_size, replace = FALSE)
}

create_rsample_obj <- function(analysis_, assessment_, data, class = NULL){
    res <- structure(
        list(
            data = data,
            in_id = analysis_,
            out_id = assessment_
        ),
        class = "rsplit"
    )
    if (!is.null(class))
        res <- rsample:::add_class(res, class)
    res
}


print_sizes <- function(split_obj){
    in_l = length(split_obj$in_id)
    out_l = length(split_obj$out_id)
    paste0("in:",in_l,"/out:",out_l)
}

training_size <- function(split_obj){
    length(split_obj$in_id)
}
