import numpy as np
import matplotlib.pyplot as plt 
import cv2
import h5py
import os
import glob
import pickle
import torch
from scipy.io import loadmat



################################################################
# NTIRE2022 data
class NTIRE2022(torch.utils.data.Dataset):
    def __init__(self, 
                image_dir, 
                spectral_dir, 
                do_discard = False, 
                  quality_log_dir = "",
                  quality_thres = -1, 
                do_crop = False,
                  stride = 8, 
                  crop_size = 128, 
                do_aug = False, 
                do_shuffle = False,
                do_shift = False,
                to_chw = False,
                load_img_type = 'rgb'):

        # attributes of NTIRE data
        self.bands= np.arange(400, 710, 10)
        self.bgr_bands = np.array([450, 550, 700])
        self.height = 482
        self.width = 512

        # files location
        self.image_dir = image_dir
        self.spectral_dir = spectral_dir
        self.img_files = sorted(glob.glob(self.image_dir + "/*.jpg"))
        self.cube_files = sorted(glob.glob(self.spectral_dir + "/*.mat"))

        # discard data based on quality threshold (do it for training data only)    
        if do_discard:
            self.discardData(quality_log_dir, quality_thres)

        # total data
        self.img_files = np.asarray(self.img_files)
        self.cube_files = np.asarray(self.cube_files)
        self.total_files = len(self.img_files)

        ####################
        if do_shuffle:
            id = np.random.permutation(self.total_files)
            self.img_files = self.img_files[id]
            self.cube_files = self.cube_files[id]

        if do_crop:
            self.crop_size, self.stride, \
              self.patch_per_line, self.patch_per_colume, self.patch_per_img \
                = defineCrop(stride, crop_size, self.height, self.width)
        else:
            self.patch_per_img = 1
            self.crop_size = -1
            self.stride = -1
        
        self.do_aug = do_aug
        self.do_shift = do_shift
        self.to_chw = to_chw
        self.load_img_type = load_img_type

    def discardData(self, quality_log_dir, quality_thres):
        if quality_thres < 0:
            print('To discard data, must set the quality_thres above 0')
            return -1
        else:
          self.quality_thres = quality_thres 

        quality_filename = "quality" + "".join(str(quality_thres).split('.'))
        quality_filedir = f'{quality_log_dir}/{quality_filename}'
        del_filename = "qdel" + "".join(str(quality_thres).split('.'))

        k_todel = []
        self.files_del = []
        self.quality = {}

        if os.path.exists(quality_filedir):
            with open(quality_filedir, 'rb') as f:
                self.quality = pickle.load(f)
            with open(f'{quality_log_dir}/{del_filename}', "rb") as f:  
                self.files_del = pickle.load(f)

            for fd in self.files_del:
                temp = int(fd.split('\\')[-1].split('.')[0].split('_')[-1]) - 1
                k_todel.append(temp)

        else:
            for k, cube_path in enumerate(self.cube_files):
                print(f'\r[{k}] check >> {cube_path}', end="")
                if self.checkQuality(cube_path) == -1:
                      k_todel.append(k)
                      self.files_del.append(cube_path)

            # save the quality files
            with open(quality_filedir, 'wb') as f:
                pickle.dump(self.quality, f)
            with open(f'{quality_log_dir}/{del_filename}', "wb") as f:  
                pickle.dump(self.files_del, f)
            print(f'save the quality files at {quality_filedir}')

        k_todel.append(530)
        k_todel.append(339)
        k_todel.append(313)
        k_todel.sort(reverse=True)
        print(f'Discarding ... >> {k_todel}')
        for k in k_todel:
            del self.img_files[k]
            del self.cube_files[k]

    def checkQuality(self, cube_path):
        self.quality[cube_path] = []

        cube = self.loadCube(cube_path)
        cube = np.transpose(cube, [0, 2, 1])    

        for k in range(30):
            ss = self.ssim(cube[k], cube[k+1])
            self.quality[cube_path] = []
            if ss < self.quality_thres:
                print(f' del : {ss}')
                self.quality[cube_path].append(k)
                return -1
                break
        return 1     

    def loadCube(self, cube_path):
        '''
        return cube in (h, w, c=31)
        range: (0, 1)
        '''
        with h5py.File(cube_path, 'r') as f:
            cube =np.float32(np.array(f['cube']))
            cube[cube<0] = 0
            
            # if self.quality_thres > 0:
            #     cube = self.masked_data(cube, cube_path)        
            f.close()
        cube = np.transpose(cube, [2, 1, 0])  # (h, w, c=31)
        return cube

    def loadData(self, img_path, cube_path):
        # load image file
        if self.load_img_type == 'rgb':
            img = loadRGB(img_path)
        elif self.load_img_type == 'gray':
            img = loadGray(img_path)
        elif self.load_img_type == 'hsv':
            img = loadHSV(img_path)

        img = (img - img.min()) / (img.max() - img.min())

        if self.do_shift:
            img = shift(img, 0, 1)

        # load cube file
        cube = self.loadCube(cube_path)

        return img, cube

    def __getitem__(self, idx):
        img_idx = idx // self.patch_per_img
        img, cube = self.loadData(self.img_files[img_idx], self.cube_files[img_idx])

        if self.crop_size > 0:
            img, cube = crop(idx, img, cube, self.patch_per_img, self.patch_per_line, self.stride, self.crop_size)
        
        if self.do_aug:  
            img, cube = augumentation(img, cube)

        if self.to_chw:
            img = np.transpose(img, [2, 0, 1])  # (c, h, w)
            cube = np.transpose(cube, [2, 0, 1])  # (c, h, w)

        return (np.ascontiguousarray(img), np.ascontiguousarray(cube))

    def __len__(self):
        return self.patch_per_img*self.total_files  
   
    def masked_data(self, cube, cube_path):
        mask = np.ones(shape=cube.shape, dtype = np.float32)
        kk = self.quality[cube_path]
        if len(kk) >0:
            for k in kk:
                if k == 0 or k == 30:
                    mask[k,:, :] = 0
                else:
                    mask[k-1:k+1, :, :] = 0
        cube *= mask
        return cube

    def ssim(self, img1, img2):
        C1 = (0.01)**2
        C2 = (0.03)**2

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def generateCieRGB(self, cube):
        rgb_c = np.dot(np.transpose(cube, [1, 2, 0]), self.cie_filter)
        rgb_c = np.float32(rgb_c)
        rgb_c /= 255 
        return np.transpose(rgb_c, [2, 0, 1])


################################################################
# NTIRE2020 data
class NTIRE2020(torch.utils.data.Dataset):
    def __init__(self, 
                image_dir, 
                spectral_dir, 
                do_crop = False,
                  stride = 8, 
                  crop_size = 128, 
                do_aug = False, 
                do_shuffle = False,
                do_shift = False,
                to_chw = False,
                load_img_type = 'rgb'):

        # attributes of NTIRE data
        self.bands= np.arange(400, 710, 10)
        self.bgr_bands = np.array([450, 550, 700])
        self.height = 482
        self.width = 512

        # files location
        self.image_dir = image_dir
        self.spectral_dir = spectral_dir
        self.img_files = sorted(glob.glob(self.image_dir + "/*.jpg"))
        self.cube_files = sorted(glob.glob(self.spectral_dir + "/*.mat"))

        # total data
        self.img_files = np.asarray(self.img_files)
        self.cube_files = np.asarray(self.cube_files)
        self.total_files = len(self.img_files)

        ####################
        if do_shuffle:
            id = np.random.permutation(self.total_files)
            self.img_files = self.img_files[id]
            self.cube_files = self.cube_files[id]

        if do_crop:
            self.crop_size, self.stride, \
              self.patch_per_line, self.patch_per_colume, self.patch_per_img \
                = defineCrop(stride, crop_size, self.height, self.width)
        else:
            self.patch_per_img = 1
            self.crop_size = -1
            self.stride = -1
        
        self.do_aug = do_aug
        self.do_shift = do_shift
        self.to_chw = to_chw
        self.load_img_type = load_img_type
 

    def loadCube(self, cube_path):
        '''
        return cube in (h, w, c=31)
        range: (0, 1)
        '''
        cube = np.float32(np.array(loadmat(cube_path)['cube']))
        cube[cube<0] = 0
        return cube

    def loadData(self, img_path, cube_path):
        # load image
        if self.load_img_type == 'rgb':
            img = loadRGB(img_path)
        elif self.load_img_type == 'gray':
            img = loadGray(img_path)
        elif self.load_img_type == 'hsv':
            img = loadHSV(img_path)

        img = (img - img.min()) / (img.max() - img.min())

        if self.do_shift:
            img = shift(img, 0, 1)

        # load cube
        cube = self.loadCube(cube_path)

        return img, cube


    def __getitem__(self, idx):
        img_idx = idx // self.patch_per_img
        img, cube = self.loadData(self.img_files[img_idx], self.cube_files[img_idx])

        if self.crop_size > 0:
            img, cube = crop(idx, img, cube, self.patch_per_img, self.patch_per_line, self.stride, self.crop_size)
        
        if self.do_aug:  
            img, cube = augumentation(img, cube)

        if self.to_chw:
            img = np.transpose(img, [2, 0, 1])  # (c, h, w)
            cube = np.transpose(cube, [2, 0, 1])  # (c, h, w)

        return (np.ascontiguousarray(img), np.ascontiguousarray(cube))

    def __len__(self):
        return self.patch_per_img*self.total_files  


################################################################
# CAVE data
class CAVE(torch.utils.data.Dataset):
    def __init__(self, 
                image_dir, 
                spectral_dir, 
                do_discard = False, 
                  quality_log_dir = "",
                  quality_thres = -1, 
                do_crop = False,
                  stride = 8, 
                  crop_size = 128, 
                do_aug = False, 
                do_shuffle = False,
                do_shift = False,
                to_chw = False,
                load_img_type = 'rgb'):

        # attributes of  data
        self.bands= np.arange(400, 710, 10)
        self.bgr_bands = np.array([450, 550, 700])
        self.height = 482
        self.width = 512

        # # files location
        # self.image_dir = image_dir
        # self.spectral_dir = spectral_dir
        # self.img_files = sorted(glob.glob(self.image_dir + "/*.jpg"))
        # self.cube_files = sorted(glob.glob(self.spectral_dir + "/*.mat"))

        # # discard data based on quality threshold (do it for training data only)    
        # if do_discard:
        #     self.discardData(quality_log_dir, quality_thres)

        # # total data
        # self.img_files = np.asarray(self.img_files)
        # self.cube_files = np.asarray(self.cube_files)
        # self.total_files = len(self.img_files)

        # ####################
        # if do_shuffle:
        #     id = np.random.permutation(self.total_files)
        #     self.img_files = self.img_files[id]
        #     self.cube_files = self.cube_files[id]

        # if do_crop:
        #     self.crop_size, self.stride, \
        #       self.patch_per_line, self.patch_per_colume, self.patch_per_img \
        #         = defineCrop(stride, crop_size, self.height, self.width)
        # else:
        #     self.patch_per_img = 1
        #     self.crop_size = -1
        #     self.stride = -1
        
        # self.do_aug = do_aug
        # self.do_shift = do_shift
        # self.to_chw = to_chw
        # self.load_img_type = load_img_type

    def __getitem__(self, idx):
        return -1

    def __len__(self):
        return -1  

################################################################
# HARVARD data
class HARVARD(torch.utils.data.Dataset):
    def __init__(self, 
                image_dir, 
                spectral_dir, 
                do_discard = False, 
                  quality_log_dir = "",
                  quality_thres = -1, 
                do_crop = False,
                  stride = 8, 
                  crop_size = 128, 
                do_aug = False, 
                do_shuffle = False,
                do_shift = False,
                to_chw = False,
                load_img_type = 'rgb'):

        # attributes of  data
        self.bands= np.arange(400, 710, 10)
        self.bgr_bands = np.array([450, 550, 700])
        self.height = 482
        self.width = 512

        # # files location
        # self.image_dir = image_dir
        # self.spectral_dir = spectral_dir
        # self.img_files = sorted(glob.glob(self.image_dir + "/*.jpg"))
        # self.cube_files = sorted(glob.glob(self.spectral_dir + "/*.mat"))

        # # discard data based on quality threshold (do it for training data only)    
        # if do_discard:
        #     self.discardData(quality_log_dir, quality_thres)

        # # total data
        # self.img_files = np.asarray(self.img_files)
        # self.cube_files = np.asarray(self.cube_files)
        # self.total_files = len(self.img_files)

        # ####################
        # if do_shuffle:
        #     id = np.random.permutation(self.total_files)
        #     self.img_files = self.img_files[id]
        #     self.cube_files = self.cube_files[id]

        # if do_crop:
        #     self.crop_size, self.stride, \
        #       self.patch_per_line, self.patch_per_colume, self.patch_per_img \
        #         = defineCrop(stride, crop_size, self.height, self.width)
        # else:
        #     self.patch_per_img = 1
        #     self.crop_size = -1
        #     self.stride = -1
        
        # self.do_aug = do_aug
        # self.do_shift = do_shift
        # self.to_chw = to_chw
        # self.load_img_type = load_img_type

    def __getitem__(self, idx):
        return -1

    def __len__(self):
        return -1  




################################################################
# some general functions      
def loadRGB(img_path):
    '''
    return rgb in (h, w, c)
    range: (0, 255)
    '''
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb)
    return rgb 

def loadGray(img_path):
    bgr = cv2.imread(img_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    return gray[..., np.newaxis]

def loadHSV(img_path):
    bgr = cv2.imread(img_path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv = np.float32(hsv)
    return hsv

def shift(img, means, varis):
    h, w, c = img.shape
    x_mean = np.mean(img, axis = (0, 1))
    x_std = np.std(img, axis = (0, 1))

    a = np.sqrt(varis/x_std**2)
    b = means - a * x_mean

    # standardize
    img_n = img * np.broadcast_to(a, (h,w,c)) + np.broadcast_to(b, (h,w,c))

    # normalize
    img_n = (img_n - img_n.min()) / (img_n.max() - img_n.min())
    return img_n

def augumentation(img, cube):
    todo = np.random.choice(4)
    if todo == 0:
        img = np.rot90(img.copy(), axes=(0, 1))
        cube = np.rot90(cube.copy(), axes=(0, 1))
    elif todo == 1:
        img = img[:, ::-1, :].copy()
        cube = cube[:, ::-1, :].copy()
    elif todo == 2:
        img = img[:, ::-1, :].copy()
        cube = cube[:, ::-1, :].copy()
    else:
        return img, cube
        
    return img, cube

def defineCrop(stride, crop_size, height, width):
    if stride < 0 or crop_size > np.minimum(height, width) or crop_size < 0:
        print('stride and crop size must be valid to perform crop operation')
        crop_size = -1
        return -1, -1, -1, -1, -1
    else:
        patch_per_line = (width - crop_size) // stride + 1
        patch_per_colume = (height - crop_size) // stride + 1
        patch_per_img = patch_per_line * patch_per_colume
    return crop_size, stride, patch_per_line, patch_per_colume, patch_per_img

def crop(idx, img, cube, patch_per_img, patch_per_line, stride, crop_size):
    patch_idx = idx % patch_per_img
    h_idx = patch_idx // patch_per_line
    w_idx = patch_idx % patch_per_line

    img = img[
        h_idx * stride:h_idx * stride + crop_size, \
        w_idx * stride:w_idx * stride + crop_size, \
        :]
    cube = cube[
        h_idx * stride:h_idx * stride + crop_size, \
        w_idx * stride:w_idx * stride + crop_size, \
        :]

    return img, cube    

# collate function with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)