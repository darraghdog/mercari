rm(list=ls())
gc()
setwd("/home/darragh/mercari/data")
library(Hmisc)
library(Metrics)
library(data.table)
train = fread("train.tsv", sep="\t")
test = fread("test.tsv", sep="\t")


# 
nchar( " ", keepNA = F)
substr(" ", 0, 20)

l2Category = function(var) unlist(lapply(strsplit(var, "/"), function(x) paste(x[1:2], collapse = "/")))
train[,category_name_l2 := l2Category(category_name)]
test[,category_name_l2 := l2Category(category_name)]

train[,brand_cat_cond_name_ship := paste(brand_name, item_condition_id, name, category_name_l2, shipping, sep = "_")]#category_name,
test[,brand_cat_cond_name_ship := paste(brand_name, item_condition_id, name, category_name_l2, shipping, sep = "_")]#category_name,
train[is.na(item_description), item_description := " "]
test[is.na(item_description), item_description := " "]
train[,brand_cat_cond_name_ship_descr := paste(brand_name,item_condition_id, name, shipping, category_name_l2, substr(item_description, 1, 20), sep = "_")]# category_name,
test[,brand_cat_cond_name_ship_descr := paste(brand_name, item_condition_id, name, shipping, category_name_l2, substr(item_description, 1, 20), sep = "_")]#category_name,

train[(nchar(name)>15&nchar(item_description)>5) ,ct:=.N, by =brand_cat_cond_name_ship_descr ]
train[ct>1,loo_price:=(sum(price)-price)/(length(price)-1), by =brand_cat_cond_name_ship_descr ]
train[ct>1,loo_logerror:= abs(log1p(price)-log1p(loo_price))]

train[nchar(name)>20 ,ctnm:=.N, by =brand_cat_cond_name_ship ]
train[ctnm>1,loo_nm_price:=(sum(price)-price)/(length(price)-1), by =brand_cat_cond_name_ship ]
train[ctnm>1,loo_nm_logerror:= abs(log1p(price)-log1p(loo_nm_price))]

table(cut2(train[ct>1]$loo_logerror, g = 50))
nrow(train[ct>1])/nrow(train)
rmsle(train[ct>1]$price, train[ct>1]$loo_price)

# [1] 0.01844004
# [1] 0.1957173

table(cut2(train[ctnm>1]$loo_nm_logerror, g = 50))
nrow(train[ctnm>1])/nrow(train)
rmsle(train[ctnm>1]$price, train[ctnm>1]$loo_nm_price)
# [1] 0.03098004
# [1] 0.2752512

rmsle(train[ctnm>1 & ct==1]$price, train[ctnm>1 & ct==1]$loo_nm_price)
# [1] 0.3254487


nrow(test[brand_cat_cond_name_ship_descr %in% unique(train[ct>1]$brand_cat_cond_name_ship_descr)])/nrow(test)
nrow(test[brand_cat_cond_name_ship %in% unique(train[ctnm>1]$brand_cat_cond_name_ship)])/nrow(test)
# [1] 0.006200251
# [1] 0.01290096

par(mfrow=c(2,1))
hist(log1p(train[ctnm>1]$price), xlim = c(0, 8))
hist(log1p(train[ct>1]$price), xlim = c(0, 8))
