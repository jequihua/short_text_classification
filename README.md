## Short Text Classification

This repository provides tools to perform automated classification of public spending based on its text description. To this end, we use xgboost, and the doc2vec algorithm to perform text feature learning. 

### Data preparation

We use several years of manually labeled examples:

* 2011 - 51500 examples
* 2012 - 47777 examples
* 2013 - 31890 examples
* 2014 - 47260 examples
* 2015 - 51101 examples
* 2016 - 27655 examples 

For a total of 256764 labeled examples.

Each of these examples include a paragraph describing the expenditure. A total amount for it. The state in which it took place. And a categorical variable indicating the "type" of expenditure it belongs to.

Aditionally, a data table corresponding to the year 2017 is available. It includes all the previously mentioned fields except the "type" (the examples are unlabeled). A total of 240178 belong to this data set.

First of all, columns names must match between years. To achieve this we perform the following substitutions:

* Entidad -> Estado
* FOLIO/Clave de Proyecto -> Folio
* Presupuesto -> Total
* Nombre de Proyecto -> Destino
* Tipo de gasto -> Tipo

After this, all the tables are stacked (naturally the "type" variable for the year 2017 is left empty: NA).

Then we proceed to clean the corpus (the expenditure description paragraph) by:

* Making everything lowe case
* Removing numbers
* Removing punctuation
* Removing unnecessary white spaces
* Removing all special characters

### Feature engineering

First, the "type" variable has some typos that must be corrected:

* "au" -> "AU"
* "ed" -> "ED"
* "re" -> "RE"
* "S" -> "SA"

Second, the "state" variable shows some differences that also need to be corrected:

* "Veracruz" -> "Veracruz de Ignacio de la Llave"
* "Michoacán" -> "Michoacán de Ocampo"
* "AGS" -> "Aguascalientes"
* "Quintana Roo." -> "Quintana Roo"
* "Distrito Federal" -> "Ciudad de México"
* "Coahuila" -> "Coahuila de Zaragoza"

Then the "state" variable is transformed into a set of 32 dummy variables to be able to enter the XGBoost model.




