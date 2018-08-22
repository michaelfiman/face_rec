'''
    Author: 
        Michaelfi
        
    Date: 
        10.7.18
    
    Description: 
        small script to create create a pickled dictionary containing files names and images of faces
    
    Python Version:
        3.5
'''
import argparse
from feret_utils import extract_faces, extract_faces_gs
import numpy as np
import pickle
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="directory containing images with names of known people", type=str)
    parser.add_argument("--gs", help="if True will create a grayscale dataset", type=bool)
    args = parser.parse_args()
    
    output_dict = {}
    
    if args.gs:
        print("GS!!")
    
    directory = args.dir
    for file in os.listdir(directory):
        file_p = ("%s/%s" % (directory, file))
        if args.gs:
            faces_pics = extract_faces_gs(file_p)
        else:
            faces_pics = extract_faces(file_p)
        if faces_pics is not None and faces_pics.shape[0] == 1:
            output_dict[file] = faces_pics
            
    with open('known_faces/faces.pickle', 'wb') as handle:
        pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            