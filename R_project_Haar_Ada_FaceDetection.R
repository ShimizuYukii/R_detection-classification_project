{library(jpeg)
  library(imager)
  library(OpenImageR)
  library(e1071)
  library(caret)
  library(rpart)
  library(fastAdaboost)
  library(reticulate)}

# Haar feature
intergraMat <- function(img){
  h=nrow(img)
  w=ncol(img)
  output <- matrix(0,h,w)
  colsum = vector("numeric",w)
  for (i in 1:h) {
    output[i,] <- vector("numeric",w)
    for (j in 1:w) {
      if(j==1) colsum[j] <- img[i,j]
      else colsum[j] <- img[i,j] + colsum[(j-1)]
      if(i==1) output[i,j] <- colsum[j]
      else output[i,j] <- output[(i-1),j] + colsum[j]
    }
  }
  return(output)
}
haar_edge_vert <- function(intimg,size=2,deep=2){
  h=nrow(intimg)
  w=ncol(intimg)
  dst=matrix(0,h-deep+1,w-2*size+1)
  for (i in 1:(h-deep+1)) {
    dst[i,] <- vector("numeric",w-2*size+1)
    for (j in 1:(w-2*size+1)) {
      white =0
      black=0
      if (i==1 && j==1) white = intimg[1,1]
      else if(i!=1 && j==1) white = intimg[i+deep-1,j+size-1] - intimg[i,j+size-1]
      else if(i==1 && j!=1) white = intimg[i+deep-1,j+size-1] - intimg[i+deep-1,j]
      else white = intimg[i+deep-1,j+size-1] - intimg[i,j+size-1]- intimg[i+deep-1,j] + intimg[i,j]
      inew = i
      jnew = j+size
      if (inew ==1) black = intimg[inew+deep-1,jnew+size-1] - intimg[inew+deep-1,jnew]
      else black = intimg[inew+deep-1,jnew+size-1] - intimg[inew,jnew+size-1]- intimg[inew+deep-1,jnew] + intimg[inew,jnew]
      dst[i,j] <- black - white
    }
  }
  return(dst)
}
haar_edge_hori <- function(intimg,size=2,deep=2){
  h=nrow(intimg)
  w=ncol(intimg)
  dst=matrix(0,h-2*size+1,w-deep+1)
  for (i in 1:(h-2*size+1)) {
    dst[i,] <- vector("numeric",w-deep+1)
    for (j in 1:(w-deep+1)) {
      white =0
      black=0
      if (i==1 && j==1) white = intimg[1,1]
      else if(i!=1 && j==1) white = intimg[i+size-1,j+deep-1] - intimg[i,j+deep-1]
      else if(i==1 && j!=1) white = intimg[i+size-1,j+deep-1] - intimg[i+size-1,j]
      else white = intimg[i+size-1,j+deep-1] - intimg[i,j+deep-1]- intimg[i+size-1,j] + intimg[i,j]
      inew = i +size
      jnew = j
      if (jnew ==1) black = intimg[inew+size-1,jnew+deep-1] - intimg[inew,jnew+deep-1]
      else black = intimg[inew+size-1,jnew+deep-1] - intimg[inew,jnew+deep-1]- intimg[inew+size-1,jnew] + intimg[inew,jnew]
      dst[i,j] <- black - white
    }
  }
  return(dst)
}
get_haar_feature <- function(img,size=2,deep=2){
  if(length(dim(img))>2) img <- toGray(img)
  intimg <- intergraMat(img)
  return(c(haar_edge_vert(intimg=intimg,size = size,deep = deep),haar_edge_hori(intimg=intimg,size = size,deep = deep)))
}
imageShow(haar_edge_vert(toGray(readJPEG(listtrainset[1])),3,6))

# dataset
{
  setwd("C:/Users/DELL/Desktop/R_project/FaceDetect/Train")
  fdlisttrainset=list.files("C:/Users/DELL/Desktop/R_project/FaceDetect/Train",pattern = "*.jpg$",recursive = T)
  fdtrainset=matrix(0,length(fdlisttrainset),15486)

  # train set
  fdtrain_y=vector()
  for (index in 1:length(fdlisttrainset)) {
    tempname=fdlisttrainset[index]
    temp=readJPEG(tempname)
    if(index < 3155) temp <- temp[1:200,,]
    temp <- resizeImage(temp,90,90)
    haar = get_haar_feature(temp)
    fdtrain_y=c(fdtrain_y,!grepl("Not",tempname))
    fdtrainset[index,] <- haar
  }
  fdtrain_y <- as.factor(fdtrain_y)
  fdtrainset <- data.frame(fdtrainset)
  fdtrainset <- cbind(fdtrainset,fdtrain_y)
  
  
  setwd("C:/Users/DELL/Desktop/R_project/FaceDetect/Validation")
  fdlisttestset=list.files("C:/Users/DELL/Desktop/R_project/FaceDetect/Validation",pattern = "*.jpg$",recursive = T)
  fdtestset=matrix(0,length(fdlisttestset),15486)
  
  # test set
  fdtest_y=vector()
  for (index in 1:length(fdlisttestset)) {
    tempname=fdlisttestset[index]
    temp=readJPEG(tempname)
    if(index < 415) temp <- temp[1:200,,]
    temp <- resizeImage(temp,90,90)
    haar = get_haar_feature(temp)
    fdtest_y=c(fdtest_y,!grepl("Not",tempname))
    fdtestset[index,] <- haar
  }
  fdtest_y <- as.factor(fdtest_y)
  fdtestset <- data.frame(fdtestset)
  fdtestset <- cbind(fdtestset,fdtest_y)
}

test_adamodel <- adaboost(fdtrain_y~.,data =fdtrainset,1)
a <- pred_testmodel <- predict(test_adamodel,fdtestset[1,])
a

fdmodel_strong <- boosting(fdtrain_y~.,data =fdtrainset,mfinal = 4)
fdmodel_weaker <- boosting(fdtrain_y~.,data =fdtrainset,mfinal = 1)
fdmodel_stronger <- boosting(fdtrain_y~.,data =fdtrainset,mfinal = 8)
fdmodel_weak <- boosting(fdtrain_y~.,data =fdtrainset,mfinal = 2)
pred_strong <- predict.boosting(fdmodel_strong,fdtestset)
pred <- vector("logical",length(pred_strong$class))
for (i in 1:length(pred)) {
  pred[i] <- ((pred_weak$class[i]==T || pred_strong$class[i]==T) && pred_stronger$class[i]==T && pred_weaker$class[i]==T)
}
pred
confusionMatrix(table(pred,fdtest_y))

cas_pred <- function(weakermodel,weakmodel,strongmodel,strongermodel,savemodel){
  if(ppredict.boosting(weakermodel,fdtestset)==FALSE) return(FALSE)
  if(ppredict.boosting(weakmodel,fdtestset)==FALSE) return(FALSE)
  if(ppredict.boosting(strongmodel,fdtestset)==FALSE) return(FALSE)
  if(ppredict.boosting(strongermodel,fdtestset)==FALSE) return(FALSE)
  return(TRUE)
}

memory.limit(80000)
load(".Rdata")

