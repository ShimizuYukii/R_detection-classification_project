***1.SPARE MORE SPACE for R than initialization to run our codes (method: memory.limit(size=80000)), and need a 64 bit version R.
2. To reshow the results in the report, you can ask us for RData files. But to run the front end, you need its RData of course
3. To get the trained CNN model, you show load the .h5 file (method: keras::load_model_hdf5("filepath") )
4. You may need a python environment with TensorFlow implemented to R to run CNN model
5. If you want to read the dataset and train the model by yourself, be careful about the path
6. Sometimes loading, training, predicting actions may cost a large period of time or even cause R breaking down, SORRY FOR THIS.
7. To implement our model to videos, you should firstly derive its all frames by other languege, like python, then read them into R.
8. We have marked the path you should check and change, use“ctrl+F”+（svm_path）and (SVM_path) and (check_path)
9. Group_implement.R has set the best parameter to run test samples，if want to check other photos, you may need change the parameters.
10. I have the habbit to divide code blocks by {}, watch out for this

If you have other questions about our code, please contact QQ: 1043453541 Mail: 11911317@mail.sustech.edu.cn