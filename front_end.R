library(jpeg)
library(imager)
library(OpenImageR)
library(e1071)
library(caret)
library(shiny)
library(DT)
library(ggplot2)
library(shinydashboard)

setwd("C:/Users/12444/Desktop/TMY/R/proj/data/Bald_Classification/Validation/Bald")
setwd("C:/Users/12444/Desktop/TMY/R/proj")
setwd("C:/Users/12444/Desktop/TMY/R/proj/data")

ui<-dashboardPage(
  dashboardHeader(title = "Balding recognizer"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Outcome", tabName = "Outcome", icon = icon("dashboard")),
      menuItem("Recognize", tabName = "Recognize",icon = icon("th"), badgeColor = "green")
    )),
  dashboardBody(
    tabItems(
      tabItem(
        tabName = "Outcome",
        fluidPage(
          fluidRow("We could know this people is bald or not in this page."),
          box(fileInput("file", "Choose JPG File"),width=12),
          box(imageOutput("image"),
              width=5,height = 240),
          box(textOutput("text1"),
              textOutput("text2"),
              textOutput("text3")),
          br(),
          br(),
          br(),
          box("The probability after increment learning",
              textOutput("text4"),
              textOutput("text5"))
        )
      ),
      tabItem(
        tabName = "Recognize",
        fluidPage(
          fluidRow("We could recognize all people's face and they are bald or not in this page."),
          box(fileInput("filer", "Choose JPG File"),width=12),
          box(plotOutput("imager"),
              width=12,height = 480)
        )
      )
    )  
  )
)

server=function(input,output){
  output$image<-renderImage({
   if (is.null(input$file))
     return(NULL)
   imgjpg=input$file
   imgjpg0=load.image(imgjpg$datapath)
   imagename=imgjpg$name
   par(mar=c(3, 3, 0.5, 0.5), mgp=c(2, 0.5, 0))
   paste(imgjpg0)
   dev.off()
   list(src=imagename)
  })
  output$text1<-renderText({
    if (is.null(input$file))
      return(NULL)
      tof=svm_test_pic(model = model_Stay,input$file$datapath)
      if (tof==FALSE) {
        if (length(Not_Bald==1))
        Not_Bald[1]=input$file$name
        else Not_Bald[length(Not_Bald)+1]=input$file$name
        result="Not Bald"
      }
      else {
        if (length(Bald)==1)
        Bald[1]=input$file$name
        else Bald[length(Bald)+1]=input$file$name
        result="Bald"
      }
    paste("The result of identification is ",result)
  })
  output$text2<-renderText({
    if (is.null(input$file))
      return(NULL)
    tof=svm_test_pic(model = model_Stay,input$file$datapath)
    prob=attr(tof,"prob")
    paste("Bald's probability --",prob[1])
  })
  output$text3<-renderText({
    if (is.null(input$file))
      return(NULL)
    tof=svm_test_pic(model = model_Stay,input$file$datapath)
    prob=attr(tof,"prob")
    paste("Not Bald's probability --",prob[2])
  })
  output$text4<-renderText({
    if (is.null(input$file))
      return(NULL)
    tof=svm_test_pic(model = model,input$file$datapath)
    prob=attr(tof,"prob")
    paste("Bald's probability --",prob[1])
  })
  output$text5<-renderText({
    if (is.null(input$file))
      return(NULL)
    tof=svm_test_pic(model = model,input$file$datapath)
    prob=attr(tof,"prob")
    paste("Not Bald's probability --",prob[2])
  })
  output$imager<-renderPlot({
    if (is.null(input$filer))
      return(NULL)
    par(mar=c(3, 3, 0.5, 0.5), mgp=c(2, 0.5, 0))
    #imageShow(load.image(input$filer$datapath))
    groupimg_new=readJPEG(input$filer$datapath)
    #imageShow(groupimg_new)
    windowlength=90
    dim(groupimg_new)
    stepnew_new=window_cutting(groupimg_new,2,5,0.8)
    dim(stepnew_new)
    img_slidenew_new=windowsliding(groupimg_new,stepnew_new)
    dim(stepnew_new)
    finalmatrix_new=nms_repeat(stepnew_new,0.05)
    dim(finalmatrix_new)
    b_new=windowsliding(groupimg_new,finalmatrix_new)
    #imageShow(b_new)
    dim(groupimg_new)
    b_new=recognition_window(groupimg_new,55,383,2)
    dim(b_new)
    #imageShow(b_new)
    b_new=svm_isbald(model_Stay,finalmatrix_new,groupimg_new)
    imageShow(b_new)
  })
}
shinyApp(ui,server)

library(jpeg)
a=0
setwd("C:/Users/12444/Desktop/TMY/R/proj/data/Bald_Classification/Train/Bald")
files=list.files(pattern = ".jpg")
a=matrix(0,54,length(files))
for (i in 1:length(files)){
a[((i-1)*54+1):(i*54)]=HOG(readJPEG(files[i]))
}
hogmatrix_bald=t(a)
hogmatrix_bald=cbind(hogmatrix_bald,rep(1,nrow(hogmatrix_bald)))

setwd("C:/Users/12444/Desktop/TMY/R/proj/data/Bald_Classification/Train/NotBald")
files=list.files(pattern = ".jpg")
b=matrix(0,54,length(files))
for (i in 1:length(files)){
  b[((i-1)*54+1):(i*54)]=HOG(readJPEG(files[i]))
}
hogmatrix_notbald=t(b)
hogmatrix_notbald=cbind(hogmatrix_notbald,rep(0,nrow(hogmatrix_notbald)))
hogmatrix=rbind(hogmatrix_bald,hogmatrix_notbald)

trainset0=trainset

trainset=trainset0
model=model_Stay

setwd("C:/Users/12444/Desktop/TMY/R/proj/data/Bald_Classification/Validation/Bald")
files_V=list.files(path="C:/Users/12444/Desktop/TMY/R/proj/data/Bald_Classification/Validation/Bald",pattern = ".jpg")
testing=adjusted_features(files_V[1],labeled = TRUE,label = "TRUE")
for (i in 2:470){
  testing=rbind(testing,adjusted_features(files_V[i],labeled = TRUE,label = "TRUE"))
}

setwd("C:/Users/12444/Desktop/TMY/R/proj/data/Bald_Classification/Validation/NotBald")
files_V=list.files(path="C:/Users/12444/Desktop/TMY/R/proj/data/Bald_Classification/Validation/NotBald",pattern = ".jpg")
testingnot=adjusted_features(files_V[1],labeled = TRUE,label = "FALSE")
for (i in 2:470){
  testingnot=rbind(testingnot,adjusted_features(files_V[i],labeled = TRUE,label="FALSE"))
}

model=svm_Increment_Learning(model,trainset,testing,label=TRUE)
model=svm_Increment_Learning(model,trainset,testingnot,label=FALSE)


test1=test_model(model_Stay,testset,test_y)
test2=test_model(model,testset,test_y)

test1$overall
test2$overall

testing1=testing[1:235,]
testing2=testing[236:470,]
testingnot1=testingnot[1:235,]
testingnot2=testingnot[236:470,]
model=svm_Increment_Learning(model,trainset,testing1,label=TRUE)
model=svm_Increment_Learning(model,trainset,testingnot1,label=FALSE)
model=svm_Increment_Learning(model,trainset,testing2,label=TRUE)
model=svm_Increment_Learning(model,trainset,testingnot2,label=FALSE)

pred_initial=svm_test_pic_nonpath(model_Stay,testset)
pred_after=svm_test_pic_nonpath(model,testset)
library(ROCR)
roc_initial=prediction(pred_initial,test_y)
roc_after=prediction(pred_after,test_y)
perf=performance(roc_initial,"tpr","fpr")
perf=performance(roc_after,"tpr","fpr")
plot(perf)

load("best_r.RData")
