library(keras)
memory.limit(40000)
{setwd("C:/Users/DELL/Desktop/R_project/Train")
  x_train <- array(0,dim = c(8154,178,178,1))
  # get Hog features
  y_train=vector()
  for (index in 1:length(listtrainset)) {
    tempname=listtrainset[index]
    temp=readJPEG(tempname)[1:178,,]
    temp <- toGray(temp)
    dim(temp) <- c(178,178,1)
    x_train[index,,,] <- temp
    y_train=c(y_train,0+!grepl("Not",tempname))
  }
  
  setwd("C:/Users/DELL/Desktop/R_project/Validation")

  # get Hog features
  x_test <- array(0,dim = c(940,178,178,1))
  y_test=vector()
  for (index in 1:length(listtestset)) {
    tempname=listtestset[index]
    temp=readJPEG(tempname)[1:178,,]
    temp <- toGray(temp)
    dim(temp) <- c(178,178,1)
    x_test[index,,,] <- temp
    y_test=c(y_test,0+!grepl("Not",tempname))
  }
}


y_train <- to_categorical(y_train,2)
y_test <- to_categorical(y_test,2)
img_rows <- img_cols <- 178
input_shape <- c(img_rows,img_cols,1)


{model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 16, kernel_size = c(20,20),input_shape = input_shape,activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(3,3)) %>%
  layer_conv_2d(filters = 8, kernel_size = c(10,10),activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128,name = "cnn_feature") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 2, activation = 'sigmoid')
model
model %>% compile(loss='categorical_crossentropy',
                  optimizer = optimizer_adagrad(),
                  metrics = c('accuracy'))

history <- model %>% fit(x_train,y_train,epochs = 40,validation_split=0.2)

model %>% evaluate(x_test,y_test)


model$get_layer("cnn_feature")

pre_sVM <- k_function(inputs = model$input,outputs = model$get_layer("cnn_feature")$output)
sample=x_train[1,,,]
dim(sample) <- c(1,178,178)

cnn_svm_trainset = data.frame(matrix(0,8154,128))
for (i in 1:4077) {
  j=2*i
  cnn_svm_trainset[(j-1):j,] <- pre_sVM(x_train[(j-1):j,,,])
}

cnn_svm_testset = data.frame(matrix(0,940,128))
for (i in 1:470) {
  j=2*i
  cnn_svm_testset[(j-1):j,] <- pre_sVM(x_test[(j-1):j,,,])
}

cnn_svm_trainset <- cbind(cnn_svm_trainset,train_y)

cnn_svm_model <- svm(train_y~.,data = cnn_svm_trainset,kernel="radial",probability = T)

test_model(cnn_svm_model,cnn_svm_testset,test_y )}

save_model_hdf5(model,"beat_model.h5")
