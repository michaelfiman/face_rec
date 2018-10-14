# faec_rec
Project for general face recognition using a trained CNN on a general DB of faces

This project was done as part of a self training after completing stanford's CS231N course meant to practice different tensorflow tools and levels (Estimator, Keras, Eager)/

The goal of this project was to create a model which can be deployed in any app to allow face comparison and recognition with respect to a local database built by the app.

The idea is to train a model on a dataset of known faces and then use the output of an intermediate layer to compare 2 given faces.
The dataset of faces used for this project was taken from FERET and includes 700 individuala with ~10 pictures per person.

During my work I attempted several solutions which were built on each other:

Solution No. 1:
  Built and trained a simple CNN face image classifier. Once trained, use an intermediate layer as a comparison between any 2 given images.
  This solution was done using the TF Estimator API.
  Training notebook can be found in: FERET_main_gray_scale.ipynb. APP can be found in: app/face_rec_app.py.
  The solution gave a score of ~70% while testing similarity.
  Though this score is not bad, unfortunately when running this model in the app, the differences between different faces were very small and it was impossible to point out "Unknown" faces (meaning faces which don't appear in the local database).
  Points worth mentioning:
    
    1. While training on the dataset, I found that some of my images are not fit for training, some cropped wrong, some unable to detect face using basic CV algorithms etc. After cleaning the DS manually, reults were dramaticlly improved.
    
    2. It's probalby a very obvious point, but looking for the points where your training overfits can teach you a lot about what you can do next to cope with the problem (dropout layer, less parameters, where to stop the training, etc.)
    
    3. The most important thing, although the intermediate layer gives a good represantation of the picture features, it doesn't mean that comparing it to other images will give a good distinction between them.
  
Solution No. 2:
  Built and trained a similar CNN model yet the objective of the training was divided into 2, having 2 losses:
    
    1. Classify the face in the image- softmax loss.
    
    2. Compare one image to another- A loss which penalizes the score according to distance between the intermediate layers depending if both images are the of the same class.
    
  This solution was done using Eager execution with Keras layers.
  Training notebook can be found in: FERET_main_class_compare.ipynb. APP can be found in: app/face_rec_advanced_app.py.
  The solution gave a score of ~70% once again.
  When testing it in the APP, the difference between different images in the DB were much bigger (20% and more), yet there were still alot of misses when the face is not pointing straight at the camera. The ability to find "Unkown" faces, is much better yet still not that good.
  
  Points worth mentioning:
  
    1. While building the comparison loss, I found that it was very problematic to get it to converge for 2 main reasons:
      
      a. The minibatches (which are created randomly by using the DS tensorflow shuffle options) were very unbalanced. There were much more "different" training examples than "same", sometimes even having no "same" examples. This created a very unbalanced loss pulling the variables to make the "distances" as big as possible.
      
     b. Attempting to train the model for both losses at the same time took a lot of tweeking until getting both losses to converge. Good practice would be to choose weights on the different losses so the the most important goal is achived.
    
    2. Even when it looks like you an overall OK score on your evaluation set, you should analyze what it actually means. When looking into the ~70% accuracy, I suddenly saw that all guesses where "different" and since the dataset is unbalanced, I got a 70% success. It's important to say that the training did actually lead to an ability to differ between 2 faces, yet it took aome tweeking with the "distance" which is expected.

Solution No. 3:
  Built and trained another similar CNN mode with the same objectives, yet this time the 2 objectives were done sequentially and not parallel. The model was first trained only while using the classification objective until reaching the best possible result. After that, all variables are frozen except for the fully-connected layer before the feature output layer which is used for comparisson. In addition I created a class which builds mini-batches, in a way where the "same" and "different" examples will be more balanced.
  
  This solution was done with Eager as well.
  Training notebook can be found in: face_main_class_compare_dist_data.ipynb. APP is the same as solution 2.
  This solution was much better than the earlier 2 reaching an accuracy of ~83%.
  While testing it in the APP it seems the destinction is better (especially when using a model which tries to differ strongly between differnt images), yet the ability to point out an "Unknown" face has stayed the more or less the same. I should add that changing the comparison parameters and amount of images used for each object can affect the scores by a bit.
  
  Points worth mentioning:
  
    1. It seems that performing the training one after the other gave better results and also made it easier to fine tune the hyper-parameters for the similarity loss.
    
    2. An interesting insight which I noticed is that when gettin the parameters are working on making a big "diff" for images which are of different individuals, they also "hurt" and make a bigger distance for the same images. This is probalby due to the fact that some features which were being given more distance for a "diff" couple, can also be the same for a "same" couple (I hope that made sense). That is why the threshold used for the "diff" examples, should not be too big.
    
    3. It would probably be wiser to take a pre-trained net which is good for face feature extraction as a basic model, train it for the classification goal, freeze all parameters except for the relevant FC layer and train the comparisson objective. This might have given a better accuracy as the model was already trained to get features from images.
    
  
Note that it is required to run a training session and copying the saved model/checkpoints (depending on the app you are using) to the app folder before running the application (files were to big to upload an example).
In addition the feret_data_dict_gs.pickle.bz2 must be uncompressed as well.
  
