"""
Test dataloader
"""
from data.base_dataset import BaseDataset
import os
import glob
import pandas as pd 
import numpy as np
import nibabel as nib
import torch

class TestDataset(BaseDataset):
    """A Test dataset class for running inference datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.isTumor = opt.isTumor
        self.rootdir = opt.dataroot
        self.files_paths = sorted(glob.glob(self.rootdir + '/imagesTs/*.nii.gz'))  # get image paths

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk with nibabel, clip between [-1000:1000] and normalize [-1:1]
        Step 3: convert your data to a PyTorch tensor.
        Optional: load tumor file to be warped as well
        Step 4: return a data point as a dictionary.
        """    
        # Image A ------------------------------------------------
        # read a image given a random integer index
        filepath_A = self.files_paths[idx]
        
        # Read input image
        nifti_A = nib.load(filepath_A)
        spacing = nifti_A.header.get_zooms()
        img_arr_A = nifti_A.get_fdata()
        img_arr_A = img_arr_A.clip(-1000,1000)
        img_arr_A = (img_arr_A-img_arr_A.min())/(img_arr_A.max()-img_arr_A.min())*2-1
        
        # Convert to torch
        img_A = torch.from_numpy(img_arr_A.astype(np.float32))  
        tum_A = torch.zeros_like(img_A)

        # Warp tumor too is needed
        if self.isTumor:
            file_A = os.path.basename(file_A_path)
            tum_A_file = os.path.join(self.rootdir, 'tumor', file_A)
            tum_A_arr = nib.load(tum_A_file).get_fdata()
            tum_A = torch.from_numpy(tum_A_arr.astype(np.float32))

        
        return {'image_A': img_A, 'fpath_A': filepath_A, 'tum_A': tum_A}
    
    def __len__(self):
        """Return the total number of images."""
        return len(self.files_paths)
