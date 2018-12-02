To run the file, upload it to colab and run it.

Last cell of notebook contains main function. In main function, you can:

1. choose task to run by comment/uncomment the task variables. Two tasks are 'classify' and 'detection'.

2. To train the model

   You can choose to train or not to train a model by comment/uncomment the train function.

   If you uncomment the train function, code will train the model and save it in the folder with name of task given by task variable. Before you test the model you train, Please change the folder path in saver.restore function correspondingly. Current path is '.' (current folder path) since I assume you will first run my model first and it cannot upload a folder in colab.

3. To test the model.

   If you want to test the model, please upload files in classify or detection to the colab and change the task variable name correspondingly. You can only test one model at a time. Before you want to test another model, please remove the model you uploaded for the former task. Again, the default path of model is current folder path.

4. Test file.

   The name of test file is controled by variable prefix and it should has format of "prefix+_X.npy" as it indicated in the first cell in the note book

5. Accuracy:

   The model I uploaded for classify should has accuracy around 98.52%. The iou for detection should be around 0.7508 (should means 75.08%). If the accuracy or iou is below them, please consider to re-train the model. The iteration I used to achieve 98.52 is 100 for n_epochs and 100 for n_batches. The iteration I used to achieve 0.7508 is 300 for n_epochs and 300 for n_batches.