import os 
import logging
import datetime
import hdf5storage
import h5py 
import numpy as np 
import cv2
import matplotlib.pyplot as plt

def initiate_logger(file_dir, name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter("[(%(asctime)s] %(message)s")

    handler = logging.FileHandler(file_dir)
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    

    logger.addHandler(handler)

    return logger 


def create_folders_for(saved_dir, logged_dir, reconstructed_dir, model_name = "", camera_type = "", external_dir = None):
    experiment_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if external_dir is not None:
        saved_dir = f"{external_dir}/{saved_dir}"
        logged_dir = f"{external_dir}/{logged_dir}"
        reconstructed_dir = f"{external_dir}/{reconstructed_dir}"

    if (not os.path.exists(logged_dir)):
        os.mkdir(logged_dir)

    if (not os.path.exists(saved_dir)):
        os.mkdir(saved_dir)

    if (not os.path.exists(reconstructed_dir)):
        os.mkdir(reconstructed_dir)

    experiment_name = f"exp-{model_name}-{camera_type}"
    exp_logged_dir = f"{logged_dir}/{experiment_name}"   # to log the experiment process
    if (not os.path.exists(exp_logged_dir)):
        os.mkdir(exp_logged_dir) 

    exp_saved_dir = f"{saved_dir}/{experiment_name}" # save the model according to each experiment instead
    if (not os.path.exists(exp_saved_dir)):
        os.mkdir(exp_saved_dir)

    exp_reconstructed_dir = f"{reconstructed_dir}/{experiment_name}" # save the model according to each experiment instead
    if (not os.path.exists(exp_reconstructed_dir)):
        os.mkdir(exp_reconstructed_dir)

    return exp_logged_dir, exp_saved_dir, exp_reconstructed_dir

def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

def save_jpg(path, rgb, quality):
    rgb = rgb.clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

def loadcube(path):
    with h5py.File(path, 'r') as f:
        cube = np.squeeze(np.float32(np.array(f['cube'])))
        cube = np.transpose(cube, [2,1,0]) 
        f.close()
    return cube

def visualize_save_cube32(cube, bands, rgb_file, saved_path, MSI = False):
    plt.figure(figsize=(20, 10))
    if MSI:
        rgb = loadcube(rgb_file)[:,:,:3]
    else:
        rgb = plt.imread(rgb_file)

    plt.subplot(4,8,1), plt.imshow(rgb), plt.title('RGB')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    for k in range(2, 33):
        plt.subplot(4,8,k)
        plt.imshow(cube[k-2]), plt.title(f'Band {bands[k-2]}')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{saved_path}.png')

