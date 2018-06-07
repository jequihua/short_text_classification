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
library("pROC")
library("Rfast")

# loud source functions
source("./tools.R")

# load full_table
load("./data/full_corpus_shuffled.RData")

# separate unlabeled data for later
unlabeled_table = full_table[is.na(full_table$Tipo),]

# class names and codes
class_names = names(table(full_table$Tipo))
class_ids = names(table(full_table$target))

# clean "Estado" variable

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

labels = full_table$target[!is.na(full_table$Tipo)]
labels = labels[total_clean!=0]

# load model (CV results and trained model)
load(file='./models/xgboost_model_cvresults_v8.model')
xgb_model = xgb.load('./models/xgboost_model_v8.model')

# reconstruct CV accuracy assesment
cv_model = cvresults$cv_model 

# Out Of Fold prediction
OOF_prediction = cv_model$pred
OOF_prediction = max.col(OOF_prediction)-1

##### ROC curves and thresholds

# initialize output df
thresholds_df = data.frame(matrix(0,length(unique(labels)),4))
colnames(thresholds_df) = c("class","class_id","threshold_Spec99","threshold_Spec95")

for(i in 1:length(unique(labels)))
{
  print(i)
  
  thresholds_df[i,"class"]=class_names[i]
  thresholds_df[i,"class_id"]=class_ids[i]
  
  label_idx = labels == class_ids[i]
  
  labels_aux = labels[label_idx]
  
  labels_aux[label_idx]=1
  labels_aux[!label_idx]=0
  
  OOF_prediction_prob_aux = cv_model$pred[,i]
  
  OOF_roc <- roc(labels_aux, 
                 OOF_prediction_prob_aux)
  
  roc_df = data.frame(Spec=OOF_roc$specificities,
                      Sens=OOF_roc$sensitivities,
                      SumSpecSens=OOF_roc$specificities+OOF_roc$sensitivities,
                      threshold=OOF_roc$thresholds)
  
  # set 99 threshold
  roc_df_subset = roc_df[roc_df$Spec>=0.999,]
  thresholds_df[i,"threshold_Spec99"] = roc_df_subset$threshold[which.max(roc_df_subset$SumSpecSens)]
  
  # set 95 threshold
  roc_df_subset = roc_df[roc_df$Spec>=0.99,]
  thresholds_df[i,"threshold_Spec95"] = roc_df_subset$threshold[which.max(roc_df_subset$SumSpecSens)]
  
  # plot roc
  ## Now plot
  #plot(OOF_roc, print.thres = c(opt_threshold), type = "S",
  #     print.thres.pattern = "%.3f (Spec = %.2f, Sens = %.2f)",
  #     print.thres.cex = .8, 
  #     legacy.axes = TRUE)
  
}


##### prediction

unlabeled_matrix = dtm_tfidf[is.na(full_table$Tipo),] 

total_clean = as.matrix(full_table$Total)
total_clean = total_clean[is.na(full_table$Tipo)]

unlabeled_matrix_gasto = cbind(unlabeled_matrix,total_clean)
#unlabeled_matrix_gasto = unlabeled_matrix_gasto[total_clean!=0,]

pred <- predict(xgb_model,unlabeled_matrix_gasto)


pred_matrix <- matrix(pred, nrow = 32, ncol = length(pred) / 32) %>% 
               t() %>%
               data.frame() 

Final_prediction = max.col(pred_matrix)-1
Final_prediction_prob = rowMaxs(as.matrix(pred_matrix),value=TRUE)

unlabeled_table$pred_proba = Final_prediction_prob
unlabeled_table$pred_id = Final_prediction
unlabeled_table$pred_Tipo = ""
unlabeled_table$pred_confianza = 0
unlabeled_table$row_id_original = cvresults$idx_shuffled[is.na(full_table$Tipo)]

head(thresholds_df$class)

for (i in 1:nrow(thresholds_df))
{
  print(i)
  
  class_id = thresholds_df$class_id[i]
  
  class_ = thresholds_df$class[i]
  
  unlabeled_table$pred_Tipo[unlabeled_table$pred_id==class_id]=class_
  
  # assign confidence level
  unlabeled_table$pred_confianza[(unlabeled_table$pred_id==class_id)&(unlabeled_table$pred_proba>thresholds_df$threshold_Spec95[i])]=1
  unlabeled_table$pred_confianza[(unlabeled_table$pred_id==class_id)&(unlabeled_table$pred_proba>thresholds_df$threshold_Spec99[i])]=2
}

View(thresholds_df)

table(unlabeled_table$pred_confianza)

table(unlabeled_table$pred_Tipo)
table(full_table$Tipo)

# save as RData
save(unlabeled_table, file="./data/unlabeled_table_prediction_v1.RData")

load("./data/unlabeled_table_prediction_v1.RData")

# save a csv
write_csv(unlabeled_table,"./data/unlabeled_table_prediction_v1.csv")

head(unlabeled_table)
