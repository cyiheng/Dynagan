"""
Dynagan base dataloader
"""
from data.base_dataset import BaseDataset
import os
import glob
import pandas as pd 
import numpy as np
import nibabel as nib
import torch

class DynaganDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--isTumor', default=False, help='Does it need to warp the tumor as well') # NEW ADDED
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        
        '''
        :param csv_path: path of csv file with the format : name,att1,att2, etc...
        where name is the file name, att = 1 for yes and -1 for no
        '''
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.isTrain = opt.isTrain
        self.isTumor = opt.isTumor
        self.rootdir = opt.dataroot
        if self.isTrain:
            folder = '/imagesTr/'
        else :
            folder = '/imagesTs/'
        self.files_paths = sorted(glob.glob(self.rootdir + folder + '*.nii.gz'))  # get image paths

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """    
        # Image A ------------------------------------------------
        # read a image given a random integer index
        filepath_A = self.files_paths[index]
        if self.isTrain:
            # Use the first phase as training base
            filepath_A = filepath_A.rsplit('_',1)[0] + '_0000.nii.gz'
        
        # get file name
        file_A = os.path.basename(file_A_path)
        
        if self.isTumor:
            tum_A_file = os.path.join(self.rootdir, 'tumor', file_A.rsplit('_',1)[0] + '_0000-tumor.nii.gz')
            tum_A_arr = nib.load(tum_A_file).get_fdata()
            
        # Read input image
        nifti_A = nib.load(filepath_A)
        spacing = nifti_A.header.get_zooms()
        img_arr_A = nifti_A.get_fdata()
        img_arr_A = img_arr_A.clip(-1000,1000)
        img_arr_A = (img_arr_A-img_arr_A.min())/(img_arr_A.max()-img_arr_A.min())*2-1

        # TODO: measure surface here, no need to read
        # Phase
        case_A = self.frame[self.frame['name'] == file_A]
        phase_A = np.array([case_A["volume"].values[0]]) # get the label value of the file
        
        
        
        # Image B -------------------------------------------------
        tmp = file_A.split('_')[0] # Get the image case number
        ftmp = self.frame[(self.frame['name'].str.match(tmp))] # Find the same case with different phase
        
        # Phase
        idx_rd = np.random.randint(0,len(ftmp)) # 5 classes (only chru dataset) 
        file_B = ftmp['name'].values[idx_rd] # Get a random index value from the same case
        #print("fileB", os.path.basename(file_B))

        case_B = self.frame[self.frame['name'] == file_B]
        phase_B = np.array([case_B["volume"].values[0]])        

        filepath_B = os.path.join(self.rootdir, file_B)
        nifti_B = nib.load(filepath_B)
        img_arr_B = nifti_B.get_fdata()
        img_arr_B = img_arr_B.clip(-1000,1000)
        img_arr_B = (img_arr_B-img_arr_B.min())/(img_arr_B.max()-img_arr_B.min())*2-1
                
        # Dvf A to B ------------------------------------
        if self.isTrain:
            dvf_nii = nib.load(os.path.join(self.rootdir, 'dvf', file_B))
            dvf_arr = dvf_nii.get_fdata().transpose(3,0,1,2)
            dvf_arr[0, ...] = dvf_arr[0, ...] / spacing[0]
            dvf_arr[1, ...] = dvf_arr[1, ...] / spacing[1]
            dvf_arr[2, ...] = dvf_arr[2, ...] / spacing[2]
            dvf = torch.from_numpy(dvf_arr.astype(np.float32))
        else:
            dvf = torch.zeros(2,3)
            tum_A = torch.from_numpy(tum_A_arr.astype(np.float32))
            
        # Different value as input : img_A, img_B, tumor_A, tumor_B, pvd
        img_A = torch.from_numpy(img_arr_A.astype(np.float32))  
        img_B = torch.from_numpy(img_arr_B.astype(np.float32))
        # dvf_x = torch.from_numpy(dvf_x_arr.astype(np.float32))  
        # dvf_y = torch.from_numpy(dvf_y_arr.astype(np.float32))  
        # dvf_z = torch.from_numpy(dvf_z_arr.astype(np.float32))  

        if self.mode == 'pvd':
            alpha = (phase_B-phase_A)/phase_A*100
        elif self.mode == 'surface':
            alpha = phase_B-phase_A
        
        if self.isTrain:
            return {'image_A': img_A, 'phase_A': phase_A, 'fpath_A': filepath_A,
                    'image_B': img_B, 'phase_B': phase_B, 'fpath_B': filepath_B,  
                    'alpha': alpha, 'spacing_y': spacing[1],
                    'dvf' : dvf,
                    #'dvf_x': dvf_x, 'dvf_y': dvf_y, 'dvf_z': dvf_z,
                   }
        else :
            return {'image_A': img_A, 'phase_A': phase_A, 'fpath_A': filepath_A, 'tum_A': tum_A,
                    'image_B': img_B, 'phase_B': phase_B, 'fpath_B': filepath_B,  
                    'alpha': alpha, 'spacing_y': spacing[1],
                    'dvf' : dvf,
                    #'dvf_x': dvf_x, 'dvf_y': dvf_y, 'dvf_z': dvf_z,
                   }
            
    
    
    def __len__(self):
        """Return the total number of images."""
        return len(self.files)
