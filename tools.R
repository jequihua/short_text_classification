
# load packages
library("tidyverse")

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

# given a feature anlabels array
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
