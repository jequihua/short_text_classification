## Short Text Classification

This repository provides tools to perform automated classification of public spending based on its text description. To this end, we use xgboost, and the doc2vec algorithm to perform text feature learning. 

### Cleaning Data

We use several years of manually labeled examples (+200000), and we pretend to predict on data from the year 2017. 

First of all, columns names must match between years. To achieve this we perform the following substitutions:

* Entidad -> Estado
* FOLIO/Clave de Proyecto -> Folio
* Presupuesto -> Total
* Nombre de Proyecto -> Destino
* Tipo de gasto -> Tipo

