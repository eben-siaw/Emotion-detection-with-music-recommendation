# Facial-Emotion detection system with machine learning

This is an application which detects our emotions at real time using webcam feed and  smartly classifies your playlist into genres, at last playing a song that suits the mood specified by the facial analysis.


The application provides a pre-trained model for emotion or mood recognition which has been trained on Kaggle's 'Fer2013' dataset.
It also provides a pre-trained model for music classification which has been trained on GTZAN Genre collection.

For running emotion recognition only the file "emotions.py" can be run which takes input from the webcam feed.

For running the music genre classification the file "audioAnalysis.py can be run with a full command like:-
'python audioAnalysis.py classifyFolder -i<inputfolder> --model <model_name> --classifier <location> --details'

Instead of a folder name a file name can also be given for classification by changing the argument 'classifyFolder' to 'classifyFile'.
As for the model it only supports 'svm' or 'gradient boosting', SVM beimng better.

To run the application as a whole 'main.py' is to be used and it will invoke emotional recognition. After you have a definite emotion 'q' is to be pressed to capture that emotion and play a suitable song.


This will make the emotion capture window exit.

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

Training the model

- audioTraintest file
