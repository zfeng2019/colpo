## colpo
The model is divided into two parts: Segmentation and Classification

- Segmentation:
The test graph was drawn after training five different reconstruction models, and the best model was selected for segmentation.
For training, we process the aceticTrain_differentModel file first and we will get five different model weights when the training is complete.
For prediction, we should extract and mark the test image using the xceptionTestModel model test file. For the other five different models, we do the same way with vgg16TestModel, resNet101V2TestModel, resNet50V2TestModel, densenet169TestModel.
For test, we use 50testTrueLabel to get 50 standard answers marked by physicians.

- Classification:
We use the getTestColorCharacter, getTestTextureCharacter, getTrainColorCharacter, getTrainTextureCharacter files to get the texture and color characteristics of the training and test data.
SMOTE_RBFSVM_BAYESModel file is our model, input training data to get our classification model.
In the first stage, the input are significant texture and color features, and its output is the preliminary prediction. 
In the second stage, the output from the first stage, together with the age, HPV, TVT, are the input. The output is the final prediction.
All the other files are models for comparison.

## Reference
