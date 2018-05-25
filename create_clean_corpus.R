# Load packages
library("tm")
library("textreg")
library("readxl")
library("tidyverse")

# list sheets of an excel file
sheets = excel_sheets("./data/2011-2016 todo 11_10_UKPF cities.xlsx")

sheets # some repeated sheets for no reason

# load first sheet
sheet = read_excel("./data/2011-2016 todo 11_10_UKPF cities.xlsx",sheet=sheets[1])

# no olvidar cambiar nombres de campos
# Entidad -> Estado
# FOLIO -> Folio
# Presupuesto -> Total
# Nombre de Proyecto -> Destino

# select only our variables of interest (textid, description, class)
full_table = sheet %>% select(Folio,Destino,Estado,Total,Tipo)

# vertically concatenate next sheets (ignoring repeated ones)
for (i in seq(3,11,2))
{
  sheet = read_excel("./data/2011-2016 todo 11_10_UKPF cities.xlsx",sheet=sheets[i]) %>% select(Folio,Destino,Estado,Total,Tipo)
  
  print(nrow(sheet))
  
  print(head(sheet))
  
  full_table = rbind(full_table,sheet)
}

# clean cases (no missing values in any field)
full_table = full_table %>% filter(complete.cases(.))

### add (unlabeled) 2016 data

# dont forget to manually delete the first 9 rows of the 2017 excel file

# read only sheet for 2016
sheet = read_excel("./data/presupuesto 2017.xlsx",sheet=1) 

head(sheet)

# data table has a bad row as names delete first row and select only columns we want
sheet = sheet %>% select(Folio,Destino,Estado,Total)

# clean cases (no missing values in any field)
sheet = sheet %>% filter(complete.cases(.))

# add a variable for "Tipo" even though it's not present (remember these cases are unlabeled)
sheet$Tipo=NA

# concatenate with full table
full_table = rbind(full_table,sheet)

# make corpus from sms_messages and clean (to lowercase, remove numbers, punctuation and extra spaces)
corpus = Corpus(VectorSource(full_table$Destino))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, stripWhitespace)

# make all messages completely lowercase
full_table$Destino = iconv(get("content", corpus), to='ASCII//TRANSLIT')

# save as RData
save(full_table, file="./data/full_corpus.RData")

head(full_table)
