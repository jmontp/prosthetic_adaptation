import os, sys

#Detertmine directory to add
relative_path = '../'
dir_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

#Oputput debug text if called as main
if __name__ == "__main__":
    print(f"Attempting to import {dir_to_add}")
    file_loc = str(os.path.dirname(__file__))+"/context.py"
    print(f"Context.py: Succesfully imported libraries \n {file_loc}")

#Insert directory into path
sys.path.insert(0, dir_to_add)

#Import the relevant modules
import k_model
import function_bases
