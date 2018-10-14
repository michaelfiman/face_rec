'''
    Author: 
        Michaelfi
        
    Date: 
        20.8.18
    
    Description: 
        This app uses a webcam to recognize similarity between faces which are held in a local databse of photos.
        The idea is to run a prediction based on a pre-trained CNN used to classify faces, and use its dense output as a comparison
        object.
        In case the face is not reckognized, face will be tagged with "Unknown".
        This app can be used to add faces to datase (see help for further instructions).
    
    Usage:
        find usage by running --help option
    
    Python Version:
        3.5
'''

from feret_utils import extract_faces_gs
import numpy as np
import cv2
import tensorflow as tf
import pickle
import time
import argparse
import os
import time
from keras import backend as K

#### GLOABLS ####
mean_image, std_image, num_of_ids = 103.659775, 52.517044, 700 # These actually should be given by part of the model

#------------------------------------------------------------#          

class CombinedClassifier(tf.keras.Model):
    '''
    This is a duplicate of the classifier which training was done on
    '''
class CombinedClassifier(tf.keras.Model):
    def __init__(self, num_of_ids, loss_type="Class", similarity_threshold=5, class_loss=0.5*1e-4):
        super().__init__()
        self.loss_type = loss_type
        self.similarity_threshold = similarity_threshold
        self.num_of_ids = num_of_ids
        self.flag_for_combined_loss = False
        
        # Conv layer 1 + Pooling
        self.conv1a = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=[4, 4],
                                            strides=(1, 1),
                                            padding='valid',
                                            activation=tf.nn.leaky_relu,
                                            use_bias=True,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                           )
        
        self.pool1a = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                               strides=(2, 2),
                                               padding='valid'
                                              )
        
        # Conv layer 2 + Pooling
        self.conv2a = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=[3, 3],
                                            strides=(1, 1),
                                            padding='same',
                                            activation=tf.nn.leaky_relu,
                                            use_bias=True,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                           )
        
        self.pool2a = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                               strides=(2, 2),
                                               padding='valid'
                                              )
        
        # Conv layer 3 + Pooling
        self.conv3a = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=[3, 3],
                                            strides=(1, 1),
                                            padding='same',
                                            activation=tf.nn.leaky_relu,
                                            use_bias=True,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                           )
        
        self.pool3a = tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                               strides=(2, 2),
                                               padding='valid'
                                              )
        
        # Dense output layer
        self.fc1a = tf.keras.layers.Dense(4096, activation=tf.nn.relu)
        
        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(rate=0.2)
        
        # Dense layer for classes
        self.fc2a = tf.keras.layers.Dense(num_of_ids)
        
        # Optimizers
        self.optimizer1 = tf.train.AdamOptimizer(learning_rate=class_loss)
        self.optimizer2 = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.85)
        
    def call(self, inputs, training=True, **kwargs):
        half_batch_size = int(int(inputs.shape[0]) / 2)
        batch_size = int(half_batch_size * 2)
        
        # Input Layer
        input_layer = tf.reshape(inputs, [-1, 96, 96, 1])
        
        # Flow for classifier
        x1 = self.conv1a(input_layer)
        x1 = self.pool1a(x1)
        x1 = self.conv2a(x1)
        x1 = self.pool2a(x1)
        x1 = self.conv3a(x1)
        x1 = self.pool3a(x1)
        x1 = tf.reshape(x1, [x1.shape[0], -1])
        x1_id_layer = self.fc1a(x1)
        
        # On training when we are doing the classification training, use dropout and get distances
        if (training):
            if self.loss_type is "Class":
                x1_dropout = self.dropout(x1_id_layer)
                x1_logits = self.fc2a(x1_dropout)
            else:
                x1_logits = self.fc2a(x1_id_layer)
            distances = tf.reduce_mean((x1_id_layer[0:half_batch_size] - x1_id_layer[half_batch_size:batch_size])**2, axis=1)

        else:
            x1_logits = self.fc2a(x1_id_layer)
            distances = 0

        
        return x1_logits, x1_id_layer, distances 
    
    def loss(self, logits1, labels1, distances, batch_size):        
        half_batch_size = int(batch_size / 2)
        
        # To make sure that both halves are equal
        batch_size = int(half_batch_size * 2)
        
        # Calculate losses according to classification requirments and comparison requirement
        # Loss 1: classification requirement
        onehot_labels = tf.one_hot(indices=tf.cast(labels1, tf.int32), depth = num_of_ids)
        loss_1 = tf.losses.softmax_cross_entropy(onehot_labels, logits1)
            
        # Loss 2: comparison requirement
        same = np.where(np.array(labels1[0:half_batch_size] - labels1[half_batch_size:batch_size]) == 0)[0]
        diff = np.arange(half_batch_size)
        diff = np.delete(diff, same)
        if (len(same) > 0):
            loss_same = tf.reduce_mean((tf.gather(distances, same)))
        loss_diff = tf.reduce_mean(tf.maximum(0, self.similarity_threshold - tf.gather(distances, diff)))
        loss_2 = loss_same * 0.5 + loss_diff * 0.5
        
        return loss_1, loss_2
    
    def optimize(self, inputs, labels):
        with tf.GradientTape(persistent=True) as tape:
            x1_logits, x1_id_layer, distances = self(inputs)
            loss_1, loss_2 = self.loss(x1_logits, labels, distances, int(inputs.shape[0]))
        
        # Apply gradinets according to loss for the training phase we are in
        if self.loss_type is "Class":
            gradients = tape.gradient(loss_1, self.variables)
            self.optimizer1.apply_gradients(zip(gradients, self.variables))
        if self.loss_type is "Similarity":
            gradients = tape.gradient(loss_2, [self.variables[6]])
            self.optimizer2.apply_gradients(zip(gradients, [self.variables[6]]))
        del(tape)
        return loss_1, loss_2
    
    #def extract_features (self, inputs):
    #    x1_logits, x1_id_layer, distance = self(inputs)
    #    return x1_id_layer
    
    def test(self, inputs, labels, similarity_test=False, training_set=False, debug_print=False):
        x1_logits, x1_id_layer, distance = self(inputs, training=False)
        test_class_1, test_class_2, test_compare = 0, 0, 0
        
        # Score of predecting the labels of the images
        pred_labels = tf.argmax(x1_logits, axis=-1)
        pred_labels = tf.cast(pred_labels, tf.float64)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
        
        #Scores for the similarity test
        correct = 0
        incorrect = 0
        similarity = 0
        
        size = int(inputs.shape[0]) if (int(inputs.shape[0]) % 2 == 0) else int(inputs.shape[0]) - 1
        
        if (similarity_test):
            if (training_set):
                for i in range(0, int(size/2)):
                    if int(labels[i]) == int(labels[int(size/2) + i]):
                        if (tf.reduce_mean(tf.abs((x1_id_layer[i] - x1_id_layer[int(size/2) + i])**2))) < self.similarity_threshold*0.35:
                            correct += 1
                        else:
                            incorrect += 1
                    else:
                        if (tf.reduce_mean(tf.abs((x1_id_layer[i] - x1_id_layer[int(size/2) + i])**2))) > self.similarity_threshold*0.35:
                            correct += 1
                        else:
                            incorrect += 1
            else:
                for i in range(0, size, 2):
                    if int(labels[i]) == int(labels[i + 1]):
                        if (tf.reduce_mean(tf.abs((x1_id_layer[i] - x1_id_layer[i + 1])**2))) < self.similarity_threshold*0.35:
                            correct += 1
                        else:
                            if(debug_print):
                                distance = tf.reduce_mean(tf.abs((x1_id_layer[i] - x1_id_layer[i + 1])**2))
                                print("Got similar pics wrong: distance = {} thresh = {}".format(distance, self.similarity_threshold*0.35))
                            incorrect += 1
                    else:
                        if (tf.reduce_mean(tf.abs((x1_id_layer[i] - x1_id_layer[i + 1])**2))) > self.similarity_threshold*0.35:
                            correct += 1
                        else:
                            if(debug_print):
                                distance = tf.reduce_mean(tf.abs((x1_id_layer[i] - x1_id_layer[i + 1])**2))
                                print("Got different pics wrong: distance = {} thresh = {}".format(distance, self.similarity_threshold*0.35))
                            incorrect += 1
            similarity = correct/(correct + incorrect)
            

        return acc, similarity
    def extract_features(self, inputs):
        x1_logits, x1_id_layer, distance = self(inputs, training=False)
        return x1_id_layer

#------------------------------------------------------------#          
        
def check_if_matched(img_dense, dense_dict):
    '''
    This function will compare the ouptut dense layer from the tested image to a dictionary with known images and return the name
    of the closest image which fits the given input and the score.
    Note that the lower the score, the more the images are similar.
    
    param img_dense:
        Numpy array which is the output if the dense layer which was predicted by the trained CNN.
        
    param dense_dict:
        A dictionary with names of known faces as keys and dense output layers as values. Each name may have one or more values 
        depending on the amount of photos used to create the face entry.
        
    retunrs:
        a tuple with the lowest (best) scoring name and the lowest score.
    '''
    scores_dict = {}
    
    # Iterate over the known images dense layer output and compare them to the image dense layer output
    for name_itr in dense_dict.keys():   
        sum_score, score = 0, 0
        for dense_item in dense_dict[name_itr]:
            diff = np.mean((np.abs(img_dense - dense_item))**2)
            sum_score += diff
            break
        score = sum_score/len(dense_dict[name_itr])
        scores_dict[name_itr] = score
    
    print_v(scores_dict)
    
    # Get the lowest score from the scores dict
    lowest_name = min(scores_dict, key=scores_dict.get)
    
    print_v("Matched image is {} with score {}".format(lowest_name, scores_dict[lowest_name]))
    
    return(lowest_name, scores_dict[lowest_name])

#------------------------------------------------------------#

def create_dense_for_pic(img):
    '''
    This function will run an image through the trained CNN and return the output from the last dense layer.
    
    param img:
        Gray scale image which is a 2D numpy array size (96, 96).
              
    retunrs:
        a numpy array holding the output of the dense layer when inputting the img through the CNN
    '''
    # Standarize image
    img = ((img - mean_image)/std_image).astype(np.float32)
    # Reshape and put image through model prediction
    img = tf.reshape(img, (1, 96, 96, 1))
    dense_layer = model.extract_features(img).numpy()
                
    # Return the dense layer output using keras backend to convert from tensor to array
    return dense_layer

#------------------------------------------------------------#

def print_v(msg):
    '''
    Wrapper to print message only if verbosity is true
    
    param msg:
        string of given message
    '''
    if (args.verbose):
        print(msg)

#------------------------------------------------------------#        
        
def get_new_faces_pics(name, source=0):
    '''
    function used to capture samples of faces for a given subject and save them as files in the samples directory.
    Files will be saved as name_<numer>.jpg.
    Note, name for each subject must be different, in case it isn't earlier photos might be deleted.
    
    param name:
        name of given subject
    param source:
        source of video if not given will use webcam as default.
    '''
    # Create capture object from webcam (cha
    cap = cv2.VideoCapture(source)
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    i = 0
    images_added = 0
   
    # We run this to sample 5 pictures within around 5 seconds
    try:
        while(i < 101):

            # Capture frame-by-frame
            ret, frame = cap.read()
            i += 1

            # Change to gray scale and get faces in frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.09,
                minNeighbors=3,
                minSize=(96, 96)
            )
            if (len(faces) == 1) and (i % 100 == 0):
                (x, y, w, h) = faces[0]

                crop_img = gray[y:y+h, x:x+w]
                crop_img = cv2.resize(crop_img, dsize=(96, 96), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(('samples/%s_%s.jpg' % (name, i)),crop_img)
                print_v("Created file samples/%s_%s.jpg" % (name, i))
                images_added += 1

            # Display the resulting frame
            cv2.imshow('frame',gray)
    except Exception as err:
        print("Error, received exception: {}". format(err))
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows() 
    
    print("{} new images were added for subject:{}".format(images_added, name))

#------------------------------------------------------------#    
    
def create_pickle_face_rec(dir):
    '''
    function used to update the known faces pickle file from a given directory.
    Saves the pickle file in known_faces/dense_dict.pickle.
    
    param dir:
        directory where sample photos are held

    '''
    output_dict = {}
    
    # Iterate over directory and dictionary of files to faces
    directory = dir
    for file in os.listdir(directory):
        file_p = ("%s/%s" % (directory, file))
        faces_pics = extract_faces_gs(file_p)
        if faces_pics is not None and faces_pics.shape[0] == 1:
            output_dict[file] = faces_pics
    

    # Create a dictionary holding names as keys and dense output scores as values
    # We will keep the dictionary saved as a pickle file which will be updated/created only when adding new photos
    dense_dict = {}

    # For each picture get the dense layer output from the model and save it in the dense_dict with the subject name as key
    for file, pic in output_dict.items():
        for i, letter in enumerate(file):
            if not letter.isalpha():
                break
        name = file[:i]
        pic = (pic - mean_image)/std_image
        # Reshape and put image through model prediction
        img = tf.reshape(pic, (1, 96, 96, 1))
        dense_layer = model.extract_features(img).numpy()
        dense_dict.setdefault(name, []).append(dense_layer)
    
    # Save output_dict so we can unpickle it when needed
    data_to_pickle = dense_dict
    
    with open('known_faces/dense_dict.pickle', 'wb') as handle:
        pickle.dump(data_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

#------------------------------------------------------------#        
        
def add_new_face(name):
    '''
    Wrapper used to call all functions needed to update the known faces db.
    Includes getting required samples and updating the known faces dense output pickle files.
    
    param name:
        name of given subject
    '''
     # First get new images
    print("New faces will be captured in 3 seconds")
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("Action!")

    get_new_faces_pics(name)
        
    # Next update the general pickle file holding all of the known faces
    create_pickle_face_rec("samples/")

#------------------------------------------------------------#

def run_face_recognizer(source=0):
    '''
    Function running an infinite loop which will attempt to recognize faces from a given video source according to a self built
    faces database.
    
    param source:
        Video source to run the face recognition on. If not given will use webcam.
    '''
    try:
        cap = cv2.VideoCapture(0)

        face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        found_time = 0
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.09,
                minNeighbors=3,
                minSize=(96, 96),
            )
            if (len(faces) > 0):
                if time.time() - found_time > 1:
                    positions = {}
                    for (x, y, w, h) in faces:         
                        crop_img = gray[y:y+h, x:x+w]
                        crop_img = cv2.resize(crop_img, dsize=(96, 96), interpolation=cv2.INTER_NEAREST)
                        dense_for_pic = create_dense_for_pic(crop_img)

                        # Do predicition
                        name, score = check_if_matched(dense_for_pic, dense_dict)
                        print(score)
                        if (score > 0.9):
                            name = ("Unknown")

                        # Keep position data to continue drawing square when there is no prediction
                        positions[(x, y, w, h)] = name

                        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(gray, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        found_time = time.time()
                else:
                    for pos, name in positions.items():
                        x, y, w, h = pos
                        cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(gray, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('frame', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as err:
        print("Error, received exception: {}". format(err))
        raise
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()




### MAIN ###
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='This app is used to recognize faces compared to a local dataset of faces using the local webcam. To add additional faces to the dataset, use the -a option when facing the webcam')
    parser.add_argument("-a", "--add_to_face_ds", help="String with the name of the subject whos photo will be added to dataset",
                        type=str, default=None)
    parser.add_argument("-v", "--verbose", help="Run with extra prints for debug", type=bool ,default=False)
    args = parser.parse_args()
    
    print_v("Verbose mode is on")
    
    # If verbose mode is on, set tf verbosity to info
    if (args.verbose == True):        
        tf.logging.set_verbosity(tf.logging.INFO)

    # Create model
    tf.enable_eager_execution()
    model = CombinedClassifier(num_of_ids = num_of_ids)
    model.load_weights('checkpoints/cpt_final_5')
    
    # If we are in adding additional faces mode, run face addition and update known faces pickle file. When done exit.
    if args.add_to_face_ds is not None:
        print_v("Adding new faces to DB")
        add_new_face(args.add_to_face_ds)
    else: # We are in operation mode, running an inifinte loop until 'q' is hit, which will recognize faces from video
        # First unpickle the dense dict
        with open('known_faces/dense_dict.pickle', 'rb') as handle:
            dense_dict = pickle.load(handle)
            
        run_face_recognizer()
            
        
       
    exit(0)
    
    