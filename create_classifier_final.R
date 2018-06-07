# load packages
library("Matrix")
library("tm")
library("tidyverse")
library("purrrlyr")
library("text2vec")
library("qdap")
library("skmeans")
library("slam")
library("xgboost")
library("caret")
library("dummies")

# loud source functions
source("./tools.R")

# load full_table
load("./data/full_corpus.RData")

set.seed=666

# shuffle rows
shuffled_list = shuffle_df(full_table,seed=666)

full_table = shuffled_list$shuffled_df

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

# build XGBoost matrix for training
final_data_matrix = xgb.DMatrix(data = labeled_matrix_gasto,
                                label = labels)


### train model

# set XGBoost parameters
xgb_params = list("objective" = 'multi:softprob',
                  "eval_metric" = "mlogloss",
                  'num_class'= 32,
                  'max_depth'= 10,  # the maximum depth of each tree
                  'eta'= 0.05,
                  'nthread'=6,
                  'gamma'=1) # the training step for each iteration

nround=500 # number of XGBoost rounds

cv.nfold=10 # number of cross-validation folds

#t1 <- Sys.time()

# Fit cross-validation.nfold * cross-validation.nround XGB models and save Out Of Fold predictions
cv_model = xgb.cv(params = xgb_params,
                  data = final_data_matrix, 
                  nrounds = nround,
                  nfold = cv.nfold,
                  verbose = FALSE,
                  prediction = TRUE)



#print(difftime(Sys.time(), t1, units = 'mins'))

# Out Of Fold prediction
OOF_prediction = cv_model$pred
OOF_prediction = max.col(OOF_prediction)-1

true_labels = labels

# confusion matrix and error metrics
confusion_matrix = confusionMatrix(factor(true_labels ), 
                                   factor(OOF_prediction),
                                   mode = "everything")

confusion_matrix

cvresults = list(cv_model=cv_model,
                 confusion_matrix=confusion_matrix,
                 true_labels=true_labels,
                 idx_shuffled=shuffled_list$shuffled_idx,
                 random_seed=shuffled_list$seed)

save(cvresults,file='./models/xgboost_model_cvresults_v8.model')



# save full model and calculate var importance

xgboost_model_v8 <- xgb.train(data=final_data_matrix, xgb_params, nrounds=nround)

xgb.save(xgboost_model_v8,'./models/xgboost_model_v8.model')