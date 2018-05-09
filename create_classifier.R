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

# load full_table
#load("./data/full_table.RData")

# class counts imbalanced and a little bit dirty
table(full_table$Tipo)

# clean "Tipo" variable
full_table$Tipo[full_table$Tipo=="au"]="AU"
full_table$Tipo[full_table$Tipo=="ed"]="ED"
full_table$Tipo[full_table$Tipo=="re"]="RE"

# create numerical dependent variable for classification
# as we will use XGBOOST classes must begin at 0
full_table$target = as.numeric(as.factor(full_table$Tipo))-1

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

# split into unlabeled and labeled feature matrix
labeled_matrix = dtm_tfidf[!is.na(full_table$Tipo),] 
unlabeled_matrix = dtm_tfidf[is.na(full_table$Tipo),] 

# build XGBoost matrix for training
final_data_matrix = xgb.DMatrix(data = labeled_matrix,
                                 label = full_table$target[!is.na(full_table$Tipo)])

# number of classes
length(table(full_table$target[!is.na(full_table$Tipo)]))

### train model

# set XGBoost parameters
xgb_params = list("objective" = 'multi:softprob',
                   "eval_metric" = "merror",
                   'num_class'= 32,
                   'max_depth'= 3,  # the maximum depth of each tree
                   'eta'= 0.3) # the training step for each iteration

nround=175 # number of XGBoost rounds

cv.nfold=5 # number of cross-validation folds

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

# confusion matrix and error metrics
confusion_matrix = confusionMatrix(factor(full_table$target[!is.na(full_table$Tipo)]), 
                   factor(OOF_prediction),
                   mode = "everything")


names(confusion_matrix)

# write to disk
write_csv(as.data.frame.matrix(confusion_matrix$table),"./evaluation_stats/confusion_matrix.csv")

write_csv(data.frame(statistics=names(confusion_matrix$overall),
                     values=confusion_matrix$overall),"./evaluation_stats/overall_accuracies.csv")

write_csv(as.data.frame.matrix(confusion_matrix$byClass),"./evaluation_stats/byClass_accuracies.csv")


