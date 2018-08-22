'''
    Author: 
        Michaelfi
        
    Date: 
        10.7.18
    
    Description: 
        small script to get all images from bzip files and create a readable csv with tagged data.
        Requires having the feret db on disk
        
        Note: need to change FIRST_RUN to true on first run :)
    
    Python Version:
        3.5
'''

import os
import shutil
import pickle

FIRST_RUN = False

if __name__ == "__main__":
    
    if FIRST_RUN:
        directory = os.fsencode('/home/shared/feret/colorferet/colorferet/dvd1/data/smaller')
        for file in os.listdir(directory):
            dirname = os.fsdecode(file)
            sub_directory = ('/home/shared/feret/colorferet/colorferet/dvd1/data/smaller/%s' %(dirname))
            os.system('sudo bunzip2 %s/00*' %(sub_directory))
        print(curr)
    

    if FIRST_RUN:
        files_dict = {}
        directory = os.fsencode('/home/shared/feret/colorferet/colorferet/dvd1/data/smaller')
        for file in os.listdir(directory):
            dirname = os.fsdecode(file)
            sub_directory = ('/home/shared/feret/colorferet/colorferet/dvd1/data/smaller/%s' %(dirname))
            for file_t in os.listdir(sub_directory):
                file_p = ('%s/%s' % (sub_directory, file_t))
                files_dict[file_t] = int(dirname)
                shutil.copy(file_p, 'pics/%s' % (file_t))

        with open('feret_dict.pickle', 'wb') as handle:
            pickle.dump(files_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    with open('feret_dict.pickle', 'rb') as handle:
        unpickled_files_dict = pickle.load(handle)
    print(unpickled_files_dict)

    
   