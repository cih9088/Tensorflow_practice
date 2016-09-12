import os
import shutil

def delete_and_create_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
