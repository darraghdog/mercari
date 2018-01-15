# This removes anything from a previous session
rm(list=ls())
# First we set our working directory
# Please change this to your working directory where the data is
setwd("/Users/dhanley2/Documents/mercari/sub")
getwd()
library(data.table)
subdh =fread("myBag_1201.csv")
sublb = fread("submit.csv")
test = fread("../data/test.tsv", sep = "\t")
test$sublb =  sublb$price
test$subdh =  subdh$price
cor(subdh$price, sublb$price)
test$diff = abs(log1p(subdh$price) - log1p(sublb$price)) 
View(tail(test[order(diff)][,.(name, diff, sublb, subdh)], 30))
View(tail(test[order(diff)], 50))
?order
