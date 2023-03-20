"""
Train dataloader
"""
from data.base_dataset import BaseDataset
import os
import glob
import pandas as pd 
import numpy as np
import nibabel as nib
import torch
from util.util import seg_body, mkdir

class TrainDataset(BaseDataset):
    """A Train dataset class for training model."""
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
        
        self.rootdir = opt.dataroot
        self.isTrain = opt.isTrain
                    
        self.input_paths = sorted(glob.glob(self.rootdir + '/imagesTr/input/*.nii.gz'))  # get image paths
        self.target_paths = sorted(glob.glob(self.rootdir + '/imagesTr/target/*.nii.gz'))  # get image paths
        
    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk with nibabel, clip between [-1000:1000] and normalize [-1:1]
        Step 3: convert your data to a PyTorch tensor.
        Step 4: return a data point as a dictionary.
        """    
        # Image A ------------------------------------------------
        # read a image given a random integer index
        filepath_A = self.input_paths[idx]
        name_A = os.path.basename(filepath_A)
        # print("filepath_A", filepath_A)
        
        # Read input image
        nifti_A = nib.load(filepath_A)
        spacing = nifti_A.header.get_zooms()
        img_A = nifti_A.get_fdata()
        img_arr_A = img_A.clip(-1000,1000)
        img_arr_A = (img_arr_A-img_arr_A.min())/(img_arr_A.max()-img_arr_A.min())*2-1
        
        if self.isTrain:
            # Get body of the input
            maskpath_A = self.rootdir + '/imagesTr/input/body/' + name_A
            if not os.path.isfile(maskpath_A):
                # Body Segmentation of input
                mask_arr_A = seg_body(img_A)
                
                # Save segmentation to avoid resegmentation and reduce time
                if not os.path.isdir(self.rootdir + '/imagesTr/input/body/'):
                    mkdir(self.rootdir + '/imagesTr/input/body/')
                nib.save(nib.Nifti1Image(mask_arr_A, nifti_A.affine, nifti_A.header), maskpath_A)
            else:
                mask_arr_A = nib.load(maskpath_A).get_fdata()
            
            # Image B ------------------------------------------------
            tmp = name_A.rsplit('_',1)[0] # Get the image case number
            ftmp = [s for s in self.target_paths if tmp in s and name_A not in s]

            # Get the target path
            idx_rd = np.random.randint(0,len(ftmp))
            filepath_B = ftmp[idx_rd] # Get a random index value from the same case
            name_B = os.path.basename(filepath_B)
            maskpath_B = self.rootdir + '/imagesTr/target/body/' + name_B
            # print("filepath_B", filepath_B)

            # Read target image
            nifti_B = nib.load(filepath_B)
            img_B = nifti_B.get_fdata()
            img_arr_B = img_B.clip(-1000, 1000)
            img_arr_B = (img_arr_B-img_arr_B.min())/(img_arr_B.max()-img_arr_B.min())*2-1

            # Get body of the target
            if not os.path.isfile(maskpath_B):
                # Body Segmentation of input
                mask_arr_B = seg_body(img_B)
                
                # Save segmentation to avoid resegmentation and reduce time
                if not os.path.isdir(self.rootdir + '/imagesTr/target/body/'):
                    mkdir(self.rootdir + '/imagesTr/target/body/')
                nib.save(nib.Nifti1Image(mask_arr_B, nifti_B.affine, nifti_B.header), maskpath_B)
            else:
                mask_arr_B = nib.load(maskpath_B).get_fdata()

            # Dvf A to B ------------------------------------
            # NOTES: DVF are obtained with the ptvreg algorithm
            dvf_nii = nib.load(self.rootdir + '/imagesTr/target/dvf/' + name_B)
            dvf_arr = dvf_nii.get_fdata().transpose(3,0,1,2) # change 4th channels to C for Tensor
            dvf_arr[0, ...] = dvf_arr[0, ...] / spacing[0]
            dvf_arr[1, ...] = dvf_arr[1, ...] / spacing[1]
            dvf_arr[2, ...] = dvf_arr[2, ...] / spacing[2]
            dvf = torch.from_numpy(dvf_arr.astype(np.float32))
            
        # Convert to torch
        img_A = torch.from_numpy(img_arr_A.astype(np.float32))
        img_B = torch.from_numpy(img_arr_B.astype(np.float32))
        
        # Calculate alpha
        dif = np.where(mask_arr_A == mask_arr_B, 0, 1)
        dif[:,dif.shape[1]//2:,:] = 0 # Keep only the anterior part (the back doesn't matter)
        
        # Distance      
        alpha = np.array([np.mean(np.sum(dif, axis = 1))*spacing[1]])
        # print('alpha', alpha)
        
        return {'image_A': img_A, 'fpath_A': filepath_A,
                'image_B': img_B, 'fpath_B': filepath_B,  
                'alpha': alpha,
                'dvf' : dvf
               }
    

    def __len__(self):
        """Return the total number of images."""
        return len(self.target_paths)
