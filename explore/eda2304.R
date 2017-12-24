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


train[(nchar(name)>25 ,ctnm:=.N, by =brand_cat_cond_name_ship ]
train[ct>1,loo_price:=(sum(price)-price)/(length(price)-1), by =brand_cat_cond_name_ship_descr ]
train[ct>1,loo_logerror:= abs(log1p(price)-log1p(loo_price))]

table(cut2(train[ct>1]$loo_logerror, g = 50))
nrow(train[ct>1])/nrow(train)
rmsle(train[ct>1]$price, train[ct>2]$loo_price)

# [1] 0.01699994
# [1] 0.1974794

nrow(test[brand_cat_cond_name_ship_descr %in% unique(train[ct>1]$brand_cat_cond_name_ship_descr)])/nrow(test)


?nchar
# check for leak
View(train[grep("$", name,fixed=TRUE)][grep("for", tolower(name))][!grep("retail", tolower(name))][,.(name, price)])

# Length of descriptions etc. 
train[,name_len := unlist(lapply(strsplit(tolower(name), " "), length))]
train[,descr_len := unlist(lapply(strsplit(tolower(item_description), " "), length))]
train[,category_name1 := lapply(strsplit(tolower(category_name), "/"), paste)]
train[,cat_len := unlist(lapply(strsplit(tolower(category_name1), " "), length))]

table(train$name_len)
hist(train$name_len)
table(train$descr_len)
hist(train$descr_len)
table(train$cat_len)
hist(train$cat_len)

train[cat_len>6]
plot(table(round(train$name_len), round(train$descr_len/10)))
