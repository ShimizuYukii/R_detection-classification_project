# library packages
{library(jpeg)
  library(imager)
  library(OpenImageR)
  library(e1071)
  library(caret)}




sample <- readJPEG(listtrainset[1]) 
# initialize data set
{
  setwd("C:/Users/DELL/Desktop/R_project/FaceDetect/Train")
  fdlisttrainset=list.files("C:/Users/DELL/Desktop/R_project/FaceDetect/Train",pattern = "*.jpg$",recursive = T)
  fdtrainset=matrix(0,length(fdlisttrainset),54)
  tempname=fdlisttrainset[index]

  # get Hog features
  fdtrain_y=vector()
  for (index in 1:length(fdlisttrainset)) {
    tempname=fdlisttrainset[index]
    temp=readJPEG(tempname)
    hog=HOG(temp)
    fdtrain_y=c(fdtrain_y,!grepl("Not",tempname))
    fdtrainset[index,] <- hog
  }
  fdtrain_y <- as.factor(fdtrain_y)
  fdtrainset <- data.frame(fdtrainset)
  fdtrainset <- cbind(fdtrainset,fdtrain_y)
  
  
  setwd("C:/Users/DELL/Desktop/R_project/FaceDetect/Validation")
  fdlisttestset=list.files("C:/Users/DELL/Desktop/R_project/FaceDetect/Validation",pattern = "*.jpg$",recursive = T)
  fdtestset=matrix(0,length(fdlisttestset),54)
  
  # test set
  fdtest_y=vector()
  for (index in 1:length(fdlisttestset)) {
    tempname=fdlisttestset[index]
    temp=readJPEG(tempname)
    hog=HOG(temp)
    fdtest_y=c(fdtest_y,!grepl("Not",tempname))
    fdtestset[index,] <- hog
  }
  fdtest_y <- as.factor(fdtest_y)
  fdtestset <- data.frame(fdtestset)
  fdtestset <- cbind(fdtestset,fdtest_y)
}

fdmodel <- svm(fdtrain_y~.,data =fdtrainset,kernel="radial",probability = TRUE)
test_model(fdmodel,testset = fdtestset,test_y = fdtest_y)