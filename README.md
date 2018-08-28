# faec_rec
project for general face recognition using a trained CNN on a general DB of faces

The goal of this project was to create a model which can be deployed in any app to allow face comparison and recognition with respect to a local database built by the app.

The idea is to train a model on a dataset of faces which are classified by the id of the person in the picture, and then use the output of an intermediate layer to compare 2 given faces.

There are 2 sets of files used to reach this goal:

1. Simple CNN classifier- 
  Implemented in tensorflow using the Estimator model, using a softmax classifier loss.
  
  Relevant files: 
  FERET_main_gray_scale.ipynb- notebook used for training the model
  app/ace_rec_app.py- application file
  
2. Multi-loss CNN classifier-
  Implemented in tensorflow using eager execution. Model is trained to classify images and also compare them.
  Loss is the comprised from the classification objective (softmax) and the comparison objective (L2 distance).
  
  Relevant files: 
  FERET_main_calass_compare.ipynb- notebook used for training the model
  face_rec_app_advanced_model.py- application file
  
Note that it is required to run a training session and copying the saved model/checkpoints to the app folder before running the application (files were to big to upload an example).
In addition the feret_data_dict_gs.pickle.bz2 must be uncompressed as well.
  
