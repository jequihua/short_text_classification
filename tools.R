
# load packages
library("tidyverse")

# shuffle a data frame row-wise
shuffle_df = function(df)
{
  df = df[sample(nrow(df)),]
  
  return(df)
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
