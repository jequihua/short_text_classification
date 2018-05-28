# load packages
library("Matrix")
library("tm")
library("tidyverse")
library("purrrlyr")
library("text2vec")
library("qdap")
library("slam")
library("xgboost")
#library("caret")
library("dummies")


# load full_table
load("./data/full_corpus.RData")

# shuffle rows
full_table = shuffle_df(full_table)

# class counts imbalanced and a little bit dirty
table(full_table$Tipo)

# clean "Tipo" variable
full_table$Tipo[full_table$Tipo=="au"]="AU"
full_table$Tipo[full_table$Tipo=="ed"]="ED"
full_table$Tipo[full_table$Tipo=="re"]="RE"
full_table$Tipo[full_table$Tipo=="S"]="SA"

# create numerical dependent variable for classification
# as we will use XGBOOST classes must begin at 0
full_table$target = as.numeric(as.factor(full_table$Tipo))-1

# clean "Estado" variable

length(table(full_table$Estado))

full_table$Estado[full_table$Estado=="Veracruz"]="Veracruz de Ignacio de la Llave"
full_table$Estado[full_table$Estado=="Michoacán"]="Michoacán de Ocampo"
full_table$Estado[full_table$Estado=="AGS"]="Aguascalientes"
full_table$Estado[full_table$Estado=="Quintana Roo."]="Quintana Roo"
full_table$Estado[full_table$Estado=="Distrito Federal"]="Ciudad de México"
full_table$Estado[full_table$Estado=="Coahuila"]="Coahuila de Zaragoza"

# make dummy variables from "Estado"
dummies_matrix = dummy(full_table$Estado,sep="_")

### build model training data table

# tokenize
it = itoken(full_table$Destino, tokenizer = word_tokenizer,
            ids = 1:nrow(full_table),
            progressbar = TRUE)

# create vocabulary
vocab = create_vocabulary(it)

vectorizer = vocab_vectorizer(vocab)

# create document-term matrix
dtm = create_dtm(it, vectorizer)

# define tf-idf model
tfidf = TfIdf$new()

### fit the model to data and transform it with the fitted model
dtm_tfidf = fit_transform(dtm, tfidf)

# add estado dummies to matrix
dtm_tfidf = cbind(dtm_tfidf,dummies_matrix)

# split into unlabeled and labeled feature matrix
labeled_matrix = dtm_tfidf[!is.na(full_table$Tipo),] 
unlabeled_matrix = dtm_tfidf[is.na(full_table$Tipo),] 

total_clean = as.matrix(full_table$Total)
total_clean = total_clean[!is.na(full_table$Tipo)]

labeled_matrix_gasto = cbind(labeled_matrix,total_clean)
labeled_matrix_gasto = labeled_matrix_gasto[total_clean!=0,]

labels = full_table$target[!is.na(full_table$Tipo)]
labels = labels[total_clean!=0]

sample_idx = sample_class_matrix(labeled_matrix_gasto,labels,100)

labeled_matrix_gasto_s = labeled_matrix_gasto[sample_idx,]
labels_s = labels[sample_idx]

# build XGBoost matrix for training
final_data_matrix = xgb.DMatrix(data = labeled_matrix_gasto_s,
                                 label = labels_s)

### grid search

# initialize lists
params_list = list()
cv_run_list = list()
eval_metric_list = list()
eval_metric_list_idx = list()
seed_number_list = list()

for (iter in 1:2000) {

  print(iter)
  
  param = list(objective = "multi:softprob",
                eval_metric = "mlogloss",
                num_class = 32,
                max_depth = sample(3:10, 1),
                eta = runif(1, .01, 0.3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  
  cv.nround = 200
  cv.nfold = 10
  
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  
  mdcv = xgb.cv(data=final_data_matrix ,
                 params = param, 
                 nfold=cv.nfold,
                 nrounds=cv.nround,
                 verbose = FALSE,
                 early_stopping_rounds=10,
                 maximize=FALSE,
                 prediction = TRUE)
  
  mdcv$evaluation_log
  

  min_logloss = min(mdcv$evaluation_log$train_mlogloss_mean)
  
  min_logloss_index = which.min(mdcv$evaluation_log$train_mlogloss_mean)
  
  # populate lists
  params_list[[iter]] = param
  cv_run_list[[iter]] = mdcv
  eval_metric_list[[iter]] = min_logloss
  eval_metric_list_idx[[iter]] = min_logloss_index
  seed_number_list[[iter]] = seed.number
  
}

final_list = list(params_list=params_list,
                  cv_run_list=cv_run_list,
                  eval_metric_list=eval_metric_list,
                  eval_metric_list_idx=eval_metric_list_idx,
                  seed_number_list=seed_number_list)

save(final_list, file="./grid_search_results/grid_search_v1_mlogloss.RData")

#nround = best_logloss_index
#set.seed(best_seednumber)
#md <- xgb.train(data=dtrain, params=best_param, nrounds=nround)

