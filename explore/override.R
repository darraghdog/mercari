library(data.table)
train <- fread("~/mercari/data/train.tsv", sep = "\t")
test <- fread("~/mercari/data/test.tsv", sep = "\t")
head(train)
gc()

train[,ct_name:=.N, by = name]
train[,ct_namepr:=length(unique(price)), by = name]

gotrn = train[ct_name>1][ct_namepr==1]
gotst = test[name %in% gotrn$name]
gotst$price = -1
cols = intersect(colnames(gotrn), colnames(gotst))
go = rbind(gotrn[,..cols,], gotst[,..cols])

View(go)
