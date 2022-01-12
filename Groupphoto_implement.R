memory.limit(800000)
#Import package
library(jpeg) # To read the picture
library(OpenImageR)# *Use three function: 1.HOG() to get the hog feature;2.Imageresize() to resize the picture; imageshow() to visualizing the picture
library(e1071)# SVM
library(caret)# Use function confusionMatrix()

# Is_bald SVM, In here, you should update the train_set, validation_set and test_data path. 
#It is convenient for you to search these by "ctrl+f (svm_path)"
{
  toGray <- function(img){
    nr=nrow(img)
    nc=ncol(img)
    dim(img) <- c(nr*nc,3)
    img <- img %*% c(299,587,144)/1000
    dim(img) <- c(nr,nc)
    return(img)
  }

  # initial data svm_path
  {setwd("C:/Users/huhongwei/Desktop/BC/Bald_Classification/Train")
    listtrainset=list.files("C:/Users/huhongwei/Desktop/BC/data/Bald_Classification/Train",pattern = "*.jpg$",recursive = T)
    hancetrain <- trainset <- matrix(0,length(listtrainset),54)
    
    # get Hog features
    train_y=vector()
    for (index in 1:length(listtrainset)) {
      tempname=listtrainset[index]
      temp=readJPEG(tempname)[1:178,,]
      hog=HOG(temp)
      temp <- temp[,ncol(temp):1,][2:175,,]
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
    
    
    #svm_path
    setwd("C:/Users/huhongwei/Desktop/BC/data/Bald_Classification/Validation")
    listtestset=list.files("C:/Users/huhongwei/Desktop/BC/data/Bald_Classification/Validation",pattern = "*.jpg$",recursive = T)
    testset=matrix(0,length(listtestset),54)
    listtestset[1]
    
    # test set
    test_y=vector()
    for (index in 1:length(listtestset)) {
      tempname=listtestset[index]
      temp=readJPEG(tempname)[1:178,,]
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
  
  model<- svm(train_y~.,data =conbine,kernel="radial",probability = TRUE,cost=1.25,gamma=0.015)
  
  svm_test_pic <- function(model,filepath,asnum = F,adjusted=FALSE){
    test <- adjusted_features(filepath)
    if(adjusted){
      adjusted_predict(model,test,probability = asnum)
    }else{predict(model,test,probability = asnum)
    }
  }
  
  svm_Increment_Learning <- function(model,trainset,filepath,label){
    index = model$index
    trainset <- trainset[model$index,]
    sample = adjusted_features(filepath,TRUE,label)
    trainset <- rbind(trainset,sample)
    return(trainset)
  }
  
  test_model <- function(model,testset,test_y,adjusted=FALSE){
    if(adjusted==TRUE){
      svm_pred<-adjusted_predict(model,testset) 
    }else{
      svm_pred<-predict(model,testset) 
    }
    svm_table=table(svm_pred,test_y)
    confusionMatrix(svm_table)
  }
  
  adjusted_predict <- function(model,testset){
    raw=as.vector(attr(predict(model,testset,probability = TRUE),"probabilities")[,2])
    adj= raw<0.556
    return(adj)
  }
  
  test_model(model,testset,test_y,TRUE)
  
  #svm_path
  setwd("C:/Users/huhongwei/Desktop/Rtest")
  svm_test_pic(model,"yuqian.jpg",asnum = T)
  svm_test_pic(model,"myhead.jpg",asnum = T)
}

# Is_human SVM, In here, you shold update the train_set, validation_set and test_data path.
#It is convenient for you to search these by "ctrl+f (SVM_path)"
{ 
  greyit=function(img){
    R = img[,,1]
    G = img[,,2]
    B = img[,,3]
    new_pic = 0.3*R+0.59*G+0.11*B
    return(new_pic)
  }# greying picture function
  set.seed(123)
  #SVM_path the human data set
  setwd("C:/Users/huhongwei/Desktop/BC/data/Bald_Classification/Train")
  files=list.files("C:/Users/huhongwei/Desktop/BC/data/Bald_Classification/Train",pattern = "*.jpg$",recursive = T)
  length(files)
  human=matrix(0,length(files),55)
  index=1
  files[1]
  for(file in files){
    temp = readJPEG(file)
    temp_matrix = greyit(temp)
    temp_hogvector = HOG(temp_matrix)
    temp_vector = c(temp_hogvector,TRUE)
    human[index, ]=temp_vector
    index = index+1
  }
  human=human[c(2000:6000),]
  
  #SVM_path the not_human data set
  setwd("C:/Users/huhongwei/Desktop/flower")
  flower_files=list.files("C:/Users/huhongwei/Desktop/flower",pattern = "*.jpg$",recursive = T)
  length(flower_files)
  not_human=matrix(0,length(flower_files),55)
  index=1
  for(file in flower_files){
    temp = readJPEG(file)
    temp_matrix = greyit(temp)
    temp_hogvector = HOG(temp_matrix)
    temp_vector = c(temp_hogvector,FALSE)
    not_human[index, ]=temp_vector
    index = index+1
  }
  df = rbind(human,not_human)
  df=as.data.frame(df)
  names(df)=c(c(1:54),"human")
  df$human=factor(df$human,levels=c(1,0),labels=c("True","false"))
  dim(df)
  train=sample(nrow(df), 0.7*nrow(df))
  df.train <- df[train,]
  df.validate <- df[-train,]
  fit.svm <- svm(human~., data=df.train,probability=TRUE)
  fit.svm
  
  svm.pred <- predict(fit.svm, na.omit(df.validate))
  dim(df.validate)
  length(svm.pred)
  svm.perf <- table(na.omit(df.validate)$human,
                    svm.pred, dnn=c("Actual", "Predicted"))
  svm.perf
  
  
  #SVM_path test one another picture
  testimg = readJPEG("C:/Users/huhongwei/Desktop/Rtest/myhead.jpg")
  testimg1 = HOG(greyit(testimg))
  testimg2 = data.frame(t(testimg1))
  names(testimg2) = c(c(1:54))
  test = list()
  test$Pred_Class <- predict(fit.svm, testimg2, probability = TRUE)
  test$Pred_Prob <- attr(test$Pred_Class, "probabilities")[,1]
  test
}


##here to update the group img path
setwd("C:/Users/huhongwei/Desktop/Rtest")
files=list.files()
groupimg = readJPEG(files[7])
#files[7]
#dim(groupimg)
imageShow(groupimg)
windowlength=90
##Some functions necessary to implement the window_sliding
###1 window sliiding
#img = readJPEG(path); scale = int, Image magnification,usually scale=2; cutstep =int, the speed of the window; prob=int<1, the probability of "True"
window_cutting = function(img,scale,cutstep,prob){
  kuan0 = dim(img)[1]
  chang0 = dim(img)[2]
  parameter = matrix(1,4,1)
  for(i in c(1:scale)){
    testkuan = i*kuan0
    testchang = i*chang0
    img1 = resizeImage(img,testkuan,testchang)
    roundnum = (round((testkuan-windowlength+1)/cutstep,digits = 0))*(round((testkuan-windowlength+1)/cutstep,digits = 0))
    parameterk = matrix(0,4,roundnum)
    index =1
    half = dim(img1)[1]/3
    for(j in seq(1,(testkuan-windowlength*i-1),cutstep)){
      for(k in seq(1,(testchang-windowlength*i-1),cutstep)){
        if((j)<half){
          window = img1[c(j:(j+windowlength-1)),c(k:(k+windowlength-1)),]
          testimg1 = HOG(greyit(window))
          testimg2 = data.frame(t(testimg1))
          names(testimg2) = c(c(1:54))
          A=predict(fit.svm,testimg2)
          test = list()
          test$Pred_Class <- predict(fit.svm, testimg2, probability = TRUE)
          pro=test$Pred_Prob <- attr(test$Pred_Class, "probabilities")[,1]
          if("True" %in% A[1]){
            if(pro>prob){
              parameterk[1,index]=j
              parameterk[2,index]=k
              parameterk[3,index]=i
              parameterk[4,index]=pro
              index = index+1 
            }
          }
        }
      }
    }
    parameter = cbind(parameter,parameterk)
  }
  n0 <- apply(parameter, 2, sum)
  i0 <- which(n0 > 0)
  return(parameter[,i0])
}


###2 draw windows function
#img = readJEPG(path); (x,y) is the window left up point location, scale=int, Image magnification
#draw red window
redwindow = function(img,x,y,scale){
  max_x=dim(img)[1]
  max_y=dim(img)[2]
  newlen = round(windowlength/scale,digits=0)
  if(((x+newlen)<max_x) & (y+newlen<max_y)){
    img[c(x:(x+newlen)),c(y:(y+3)),1]=1
    img[c(x:(x+newlen)),c(y:(y+3)),2]=0
    img[c(x:(x+newlen)),c(y:(y+3)),3]=0
    img[c(x:(x+newlen)),c((y+newlen):(y+newlen+3)),1]=1
    img[c(x:(x+newlen)),c((y+newlen):(y+newlen+3)),2]=0
    img[c(x:(x+newlen)),c((y+newlen):(y+newlen+3)),3]=0
    img[c(x:(x+3)),c(y:(y+newlen)),1]=1
    img[c(x:(x+3)),c(y:(y+newlen)),2]=0
    img[c(x:(x+3)),c(y:(y+newlen)),3]=0
    img[c((x+newlen):(x+newlen+3)),c(y:(y+newlen),1),1]=1
    img[c((x+newlen):(x+newlen+3)),c(y:(y+newlen),1),2]=0
    img[c((x+newlen):(x+newlen)),c(y:(y+newlen),1),3]=0
    return(img)
  }
} 
#draw green window
greenwindow = function(img,x,y,scale){
  newlen = round(windowlength/scale,digits=0)
  img[c(x:(x+newlen)),c(y:(y+3)),1]=0
  img[c(x:(x+newlen)),c(y:(y+3)),2]=1
  img[c(x:(x+newlen)),c(y:(y+3)),3]=0
  img[c(x:(x+newlen)),c((y+newlen):(y+newlen+3)),1]=0
  img[c(x:(x+newlen)),c((y+newlen):(y+newlen+3)),2]=1
  img[c(x:(x+newlen)),c((y+newlen):(y+newlen+3)),3]=0
  img[c(x:(x+3)),c(y:(y+newlen)),1]=0
  img[c(x:(x+3)),c(y:(y+newlen)),2]=1
  img[c(x:(x+3)),c(y:(y+newlen)),3]=0
  img[c((x+newlen):(x+newlen+3)),c(y:(y+newlen),1),1]=0
  img[c((x+newlen):(x+newlen+3)),c(y:(y+newlen),1),2]=1
  img[c((x+newlen):(x+newlen)),c(y:(y+newlen),1),3]=0
  #imageShow(img)
  return(img)
} 


###3 draw all red windows by the parameter matrix(4*n)
#img=readJEPG(path); matrix0=parameter which dim =(4*n)
windowsliding = function(img,matrix0){
  col = dim(matrix0)[2]
  for (i in c(2:col)) {
    scale = matrix0[,i][3]
    x = round(matrix0[,i][1]/scale,digits = 0)
    y = round(matrix0[,i][2]/scale,digits = 0)
    scale = matrix0[,i][3]
    img = redwindow(img,x,y,scale)
  }
  return(img)
}

###4 combine123 to get the window_sliding
ws_function = function(img,windowlength,prob,step,scale){
  windowlength = windowlength
  temp = window_cutting(img,scale,step,prob)
  answer_img = windowsliding(img,temp)
  imageShow(answer_img)
  return(temp)
}

###5 Calculate the coincidence rate of the two windows
#(x1,y1) and (x2,y2) are the two windows left up point location
nms_IOU = function(x1,y1,x2,y2){
  difference_x = abs(x1-x2)
  difference_y = abs(y1-y2)
  distance = sqrt(difference_x^2+difference_y^2)
  if(distance>windowlength*sqrt(2)){
    return(0)
  }
  else{
    intersection= (windowlength-difference_x)*(windowlength-difference_y)
    union = windowlength*windowlength*2-intersection
    iou = intersection/union
    return(iou)
  }
}

###6 NMS drop the overlapping windows
## parameter_matrix is the matrix(4*n), ioumax = max_overlapping ratio
nms_repeat = function(parameter_matrix,ioumax){
  answer = matrix(1,4,1)
  temp_matrix = parameter_matrix
  index=2
  while (dim(answer)[2] < dim(temp_matrix)[2]) {
    temp_matrix = temp_matrix[ ,order(temp_matrix[4,],decreasing = T)]
    answer=cbind(answer,temp_matrix[,index])
    col = dim(temp_matrix)[2]
    x1 = round(temp_matrix[1,index]/(temp_matrix[3,index]),digits=0)
    y1=  round(temp_matrix[2,index]/(temp_matrix[3,index]),digits=0)
    for (i in c(3:col)) {
      x2 = round(temp_matrix[1,i]/(temp_matrix[3,i]),digits=0)
      y2 = round(temp_matrix[2,i]/(temp_matrix[3,i]),digits=0)
      iou = nms_IOU(x1,y1,x2,y2)
      if(iou>ioumax){
        temp_matrix[,i]=0
      }
    }
    n0 <- apply(temp_matrix, 2, sum)
    i0 <- which(n0 > 0)
    temp_matrix=(temp_matrix[,i0])
    index = index+1
  }
  return(answer)
} 

###7 face detection
Face_detection = function(img,windowlength,prob,step,scale,ioumax){
  windowlength=windowlength
  t1 = proc.time()
  temp = ws_function(img,windowlength,prob,step,scale)
  answer0 = nms_repeat(temp,ioumax)
  answer1 = windowsliding(img,answer0)
  t2 = proc.time()
  t0=t2-t1
  sprintf("FD RunTime:%.2f√Î",t0[3][[1]])
  imageShow(answer1)
  return(answer0)
}

###8 choose the picture to svm_isbald
recognition_window = function(img,x,y,scale){
  if(x<=44/scale){
    return(img)
  }
  else if(y<=44/scale){
    return(img)
  }
  else{
    max_x = dim(img)[1]
    max_y = dim(img)[2]
    newpointx = x -44/scale
    newpointy = y -44/scale
    finalx = x + 44/scale+90
    finaly = y+ 44/scale+90
    if(finaly<max_y & finalx<max_x){
      newimg = img[c(newpointx:finalx),c(newpointy:finaly),]
    }
    else if(finaly>max_y&finalx<max_x){
      newimg = img[c(newpointx:finalx),c((max_y-88/scale-90):max_y),]
    }
    else if(finaly<max_y&finalx>max_x){
      newimg = img[c((max_x-88/scale-90):max_x),c(newpointy:finaly),]
    }
    
    return(newimg)
  }
}

###9 using the svm to recognize the answer
svm_isbald = function(svm.model,matrix,recognition_img){
  num=dim(matrix)[2]
  if(num>=2){
    for (i in c(2:num)) {
      scale = matrix[3,i]
      x = round(matrix[1,i]/scale,digits=0)
      y = round(matrix[2,i]/scale,digits=0)
      img0 = recognition_window(recognition_img,x,y,scale)
      testimg = img0
      testimg1 = HOG(greyit(testimg))
      testimg2 = data.frame(t(testimg1))
      test = list()
      test$Pred_Class <- predict(svm.model, testimg2, probability = TRUE)
      test$Pred_Prob <- attr(test$Pred_Class, "probabilities")[,1]
      if(test$Pred_Prob<0.5){
        recognition_img=greenwindow(recognition_img,x,y,scale)
      }
      if(test$Pred_Prob>=0.5){
        recognition_img=redwindow(recognition_img,x,y,scale)
      }
    }
  }
  return(recognition_img)
}

###10 combine1~9 to get the final answer
whoisbald_amongus = function(img,windowlength,prob,step,scale,ioumax,svm.model){
  ta=proc.time()
  temp = Face_detection(img,windowlength,prob,step,scale,ioumax)
  answer= svm_isbald(svm.model,temp,img)
  imageShow(answer)
  tb = proc.time()
  t0=tb-ta
  sprintf("BC RunTime:%.2f√Î",t0[3][[1]])
}




Face_detection(img=groupimg,windowlength=90,prob = 0.8,step = 5,scale=2,ioumax = 0.05) ##example for "test.jpg, human face location detection"
whoisbald_amongus(groupimg,90,0.8,5,2,0.05,model) ## example for"test.jpg, to find who is bald?"



