rm(list = ls())

# library packages
{library(jpeg)
library(imager)
library(OpenImageR)
library(e1071)
library(caret) }

# define LGB features
# R*0.299 + G*0.587 + B*0.114
toGray <- function(img){
  nr=nrow(img)
  nc=ncol(img)
  dim(img) <- c(nr*nc,3)
  img <- img %*% c(299,587,144)/1000
  dim(img) <- c(nr,nc)
  return(img)
}
lbp <- function (img) {
  if(length(dim(img))==3){
    img <- toGray(img)
  }
  trans <- c(-1, 0, -1, 1, 0, 1, 1, 1, 1, 0, 1, -1, 0, -1, -1, -1)
  r <- 1
  trans <- matrix(trans, 2, 8)
  x <- nrow(img)
  y <- ncol(img)
  o.img <- list()
  t.img <- list()
  o.img.1 <- img[(r + 1):(x - r), (r + 1):(y - r)]
  for (i in 1:ncol(trans)) {
    dx <- trans[1, i]
    dy <- trans[2, i]
    o.img[[i]] <- img[(r + 1 + dx):(x - r + dx), (r + 1 + dy):(y - r + dy)] - o.img.1
  }
  ulst.d <- unlist(o.img)
  ulst.d <- ifelse(ulst.d >= 0, 1, 0)
  bin.data <- matrix(ulst.d, length(ulst.d)/8, 8)
  o.img.f <- matrix(apply(bin.data, 1, bin2dec), x - 2 * r, y - 2 * r)
  return(as.vector(table(cut(o.img.f,c(0:256)))))
}

# initialize data set
{setwd("C:/Users/DELL/Desktop/R_project/Train")
listtrainset=list.files("C:/Users/DELL/Desktop/R_project/Train",pattern = "*.jpg$",recursive = T)
hancetrain <- trainset <- matrix(0,length(listtrainset),54)

# get Hog features
train_y=vector()
for (index in 1:length(listtrainset)) {
  tempname=listtrainset[index]
  temp=readJPEG(tempname)[1:145,,]
  hog=HOG(temp)
  temp <- temp[,ncol(temp):1,][2:143,,]
  hancehog=HOG(temp)
  train_y=c(train_y,!grepl("Not",tempname))
  trainset[index,] <- hog
  hancetrain[index,] <- hancehog
}
train_y <- as.factor(train_y)
trainset <- data.frame(trainset)
hancetrain <- data.frame(hancetrain)
# data enhance
trainset <- cbind(trainset,train_y)
subhancetrain <- cbind(hancetrain,train_y)[1:150,]
conbine <- rbind(trainset,subhancetrain)

setwd("C:/Users/DELL/Desktop/R_project/Validation")
listtestset=list.files("C:/Users/DELL/Desktop/R_project/Validation",pattern = "*.jpg$",recursive = T)
testset=matrix(0,length(listtestset),54)

# test set
test_y=vector()
for (index in 1:length(listtestset)) {
  tempname=listtestset[index]
  temp=readJPEG(tempname)[1:145,,]
  hog=HOG(temp)
  test_y=c(test_y,!grepl("Not",tempname))
  testset[index,] <- hog
}
test_y <- as.factor(test_y)
testset <- data.frame(testset)
testset <- cbind(testset,test_y)}

adjusted_features <- function(filepath,labeled=FALSE,label=TRUE){
  sample=readJPEG(filepath)
  remain=(dim(sample)[1] %/% 3)*2
  hog=HOG(sample)
  feature=data.frame(t(hog))
  if(labeled){
    train_y <- as.factor(label)
    feature <- cbind(feature,train_y)
  }
  return(feature)
}

# test svm
test_model <- function(model,testset,test_y){
  svm_pred<-predict(model,testset)
  svm_table=table(svm_pred,test_y)
  confusionMatrix(svm_table)
}

{# train SVM
model<- svm(train_y~.,data =conbine,kernel="radial",probability = TRUE)
svm_pred<-predict(model,testset[,1:54])
# model assignment
svm_table=table(svm_pred,test_y)
confusionMatrix(svm_table)}

# further train
#{tuned<-tune.svm(train_y~.,data =conbine,gamma = 10^(-8:-2),cost = (2:5))
#tuned<-tune.svm(train_y~.,data =conbine,gamma = 10^(-2:1),cost = (5:7))
#summary(tuned)
#model<- svm(train_y~.,data =conbine,kernel="radial",probability = TRUE,cost=1.25,gamma=0.015)
#svm_pred<-predict(model,testset[,1:54])
# model assignment
#svm_table=table(svm_pred,test_y)
#confusionMatrix(svm_table)}

model_get_acc <- function(model,testset,test_y){
  pred<-predict(model,testset)
  # model assignment
  mt_table=table(pred,test_y)
  return(confusionMatrix(mt_table)$overall[1])
}
model_get_acc(model,testset[,1:54],test_y)

svm_test_pic <- function(model,filepath,asnum = F){
  test <- adjusted_features(filepath)
  predict(model,test,probability = asnum)
}

svm_test_pic(model,"C:/Users/DELL/Desktop/R_project/Test/test2.jpg",asnum = T)
svm_test_pic(model,"C:/Users/DELL/Desktop/R_project/Test/test1.jpg",asnum = T)



# *** I-SVM

# test support vector
{model<- svm(train_y~.,data = trainset[model$index,],kernel="radial",probability = T)
svm_pred_cont<-predict(model,testset[,1:54],probability = T)
# model assignment
svm_pred_cont == svm_pred}

# test increment learning
{sample <- adjusted_features("C:/Users/DELL/Desktop/R_project/Test/test2.jpg",TRUE,TRUE)
suppVec <- rbind(trainset[model$index,],sample)
sample <- adjusted_features("C:/Users/DELL/Desktop/R_project/Test/test1.jpg",TRUE,FALSE)
suppVec <- rbind(trainset[model$index,],sample)
newmodel<- svm(train_y~.,data = suppVec,kernel="radial",probability = T)

model_get_acc(newmodel,testset,test_y)
model_get_acc(model,testset,test_y)}

# I-SVM steps
svm_Increment_Learning <- function(model,trainset,filepath,label){
  index = model$index
  trainset <- trainset[model$index,]
  sample = adjusted_features(filepath,TRUE,label)
  trainset <- rbind(trainset,sample)
  return(trainset)
}

trainset <- svm_Increment_Learning(model,trainset,"C:/Users/DELL/Desktop/R_project/Test/test2.jpg",FALSE)
model <- svm(train_y~.,data = trainset,kernel="radial",probability = T)

