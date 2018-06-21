
# load packages
library("tidyverse")

# XGBoost confusion matrix based on the caret package
xgb_confusion = function(xgb.cv_model,true_labels)
{
  # Out Of Fold prediction
  OOF_prediction = max.col(xgb.cv_model$pred)-1
  
  # confusion matrix and error metrics
  confusion_matrix = confusionMatrix(factor(true_labels ), 
                                     factor(OOF_prediction),
                                     mode = "everything")
  return(confusion_matrix)
}

# XGBoost hyperparameter grid search
xgb_gridsearch = function(niter=1000,
                          cv.nround = 500,
                          cv.nfold = 10,
                          seed=NULL,
                          objective="multi:softprob",
                          eval_metric = "mlogloss",
                          num_class = NULL,
                          max_depth_lims = c(2,10),
                          eta_lims = c(0.01,0.3),
                          gamma_lims = c(0.0,0.2), 
                          subsample_lims = c(0.6,0.9),
                          colsample_bytree_lims = c(0.5,0.8), 
                          min_child_weight_lims = c(1,40),
                          max_delta_step_lims = c(1:10)){
  # initialize lists
  params_list = list()
  cv_run_list = list()
  eval_metric_list = list()
  eval_metric_list_idx = list()
  seed_number_list = list()
  
  for (iter in 1:niters) {
    
    param = list(objective = objective,
                 eval_metric = eval_metric,
                 num_class = num_class,
                 max_depth = sample(max_depth_lims[1]:max_depth_lims[2], 1),
                 eta = runif(1, .01, 0.3),
                 gamma = runif(1,eta_lims[1],eta_lims[2]), 
                 subsample = runif(1,subsample_lims[1],subsample_lims[2]),
                 colsample_bytree = runif(1,colsample_bytree_lims[1],colsample_bytree_lims[2]), 
                 min_child_weight = sample(min_child_weight_lims[1]:min_child_weight_lims[2], 1),
                 max_delta_step = sample(max_delta_step_lims[1]:max_delta_step_lims[2], 1))

    if (!is.null(seed))
    {
      seed = sample.int(10000, 1)[[1]]
    }
    
    set.seed(seed)
    
    mdcv = xgb.cv(data=final_data_matrix ,
                  params = param, 
                  nfold=cv.nfold,
                  nrounds=cv.nround,
                  verbose = FALSE,
                  early_stopping_rounds=10,
                  maximize=FALSE,
                  prediction = FALSE)
    
    min_logloss = min(mdcv$evaluation_log$train_mlogloss_mean)
    
    min_logloss_index = which.min(mdcv$evaluation_log$train_mlogloss_mean)
    
    # populate lists
    params_list[[iter]] = param
    cv_run_list[[iter]] = mdcv
    eval_metric_list[[iter]] = min_logloss
    eval_metric_list_idx[[iter]] = min_logloss_index
    seed_number_list[[iter]] = seed
    
  }
  
  final_list = list(params_list=params_list,
                    cv_run_list=cv_run_list,
                    eval_metric_list=eval_metric_list,
                    eval_metric_list_idx=eval_metric_list_idx,
                    seed_number_list=seed_number_list)
  
  return(final_list)
  }

# shuffle a data frame row-wise
shuffle_df = function(df,seed=NULL)
{
  if (!is.null(seed))
  {
    set.seed(seed)
  }
  
  idx = 1:nrow(df)
  
  idx_sample = sample(idx,length(idx))
  
  df = df[idx_sample,]
  
  output_list = list(shuffled_df=df,shuffled_idx=idx_sample,seed=seed)
  
  return(output_list)
}

# given a an array of class labels
# calculates weights based on class proportions
# or with thresholds and values specified by the user
calculate_weights = function(in_labels,thresholds=NULL,weights=NULL)
{
  array_length = length(in_labels)
  
  weights_array = rep(1,array_length)
  
  idx = 1:array_length
  
  counts_df = data.frame(table(in_labels),stringsAsFactors = FALSE)
  
  for (i in 1:nrow(counts_df))
    {
      
      counts_of_class = counts_df$Freq[i]
      
      sub_idx = idx[in_labels==(as.numeric(counts_df$in_labels[i])-1)]
      
      if (is.null(thresholds) |is.null(weights))
      {
      weights_array[sub_idx]= 1-length(sub_idx)/array_length
      }
      else
      {
       for(j in 1:length(thresholds))
       {
         threshold=thresholds[j]
         if (counts_of_class<=threshold)
         {
           weights_array[sub_idx]=weights[j]
         }
       }
      }
      
  }
  
  return(weights_array)
}

# given a feature matrix and a corresponding labels array
# samples classes in labels to obtain at least n obs of each
sample_class_matrix = function(in_matrix,in_labels,n)
{
  idx = 1:length(in_labels)
  
  counts_df = data.frame(table(in_labels),stringsAsFactors = FALSE)

  sample_idx = c()
  
  for (i in 1:nrow(counts_df))
  {
    
    counts_of_class = counts_df$Freq[i]
    
    sub_idx = idx[in_labels==(as.numeric(counts_df$in_labels[i])-1)]

    if (counts_of_class>n)
    {
      sub_idx = sample(sub_idx,n)
    }
    
    sample_idx = c(sample_idx,sub_idx)
  }
  
  return(sample_idx)
}

# sort vector based on another vector
sort_vector = function(vector_to_sort,order_vector)
{
  vector_to_sort_factor = factor(vector_to_sort, levels = order_vector, ordered=TRUE)
  
  vector_to_sort_factor = as.numeric(vector_to_sort_factor)
  
  return(vector_to_sort_factor)
}
