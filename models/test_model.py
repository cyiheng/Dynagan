from .base_model import BaseModel
from . import networks

import torch
import itertools
import os
import pandas as pd
import numpy as np

from util.util import mkdir, save_as_3D, save_as_4D, create_4DCT
from util.spatialTransform import SpatialTransformer

class TestModel(BaseModel):
    """ 
    Test model
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='dynagan')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        parser.add_argument('--loop', type=int, default=0, help='Looping mode for generating the final 4DCT [ 0 | 1 | 2 ] ; default : 0')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.alphas = torch.linspace(opt.alpha_min, opt.alpha_max, steps=opt.alpha_step)
        self.isTumor = opt.isTumor
        self.loop = opt.loop
        self.transform = SpatialTransformer([128,128,128]).to(self.device)
        self.transform_tum = SpatialTransformer([128,128,128], 'nearest').to(self.device)
        self.output_dir = './{}/{}/'.format(opt.results_dir, opt.name)
        
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        
        self.real_A = input['image_A'].unsqueeze(1).to(self.device)
        self.image_paths = input['fpath_A']
        if self.isTumor:
            self.tum_input = input['tum_A'].unsqueeze(1).to(self.device).float()


    def forward(self):
        """Run forward pass."""
        # Scan number
        case = os.path.basename(self.image_paths[0]).split('_')[1]
        
        # Create folders for output file
        self.dvf_dir    = os.path.join(self.output_dir, case, 'dvf/')
        self.warped_dir = os.path.join(self.output_dir, case, 'warped/')
        mkdir(self.dvf_dir)
        mkdir(self.warped_dir)
        
        if self.isTumor:
            self.tumor_dir = os.path.join(self.output_dir, case, 'tumor/')
            mkdir(self.tumor_dir)
        
        print("Ready to generate 4DCT of case", case)
        
        for i, alpha in enumerate(self.alphas):
            self.fake_B_dvf = self.netG(self.real_A, torch.tensor([[alpha]])) # generate dvf 
            self.fake_B = self.transform(self.real_A, self.fake_B_dvf) # warp to image
            if self.isTumor:
                self.fake_B_tumor = self.transform_tum(self.tum_input, self.fake_B_dvf) # warp tumor
            
            # Save generated images
            self.save_samples(self.output_dir, i)
        
        # Stack and generate the 4DCT image
        output_file = os.path.join(self.output_dir, 'LungCT_{}_4DCT.nii.gz'.format(case))
        create_4DCT(self.image_paths, self.warped_dir, output_file, loop=self.loop)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
        
    def save_samples(self, image_folder, phase):
        """Saves a generated sample from the validation set
        :param image_folder: folder where images will be saved
        :param phase: for name purpose, give which i the sample are generated
        """
        
        # Output filenames
        fake_dvf_file    = os.path.join(self.dvf_dir, 'Fake_{:04d}-dvf.nii.gz'.format(phase))
        fake_warped_file = os.path.join(self.warped_dir, 'Fake_{:04d}-warped.nii.gz'.format(phase))
        
        # Save warped and dvf file
        save_as_3D(self.fake_B, self.image_paths, fake_warped_file)
        save_as_4D(self.fake_B_dvf, self.image_paths, fake_dvf_file)
        
        # If need to warp tumor
        if self.isTumor:
            fake_tum_file= os.path.join(self.tumor_dir, 'Fake_{:04d}-tumor.nii.gz'.format(phase))
            save_as_3D(self.fake_B_tumor, self.image_paths, fake_tum_file)

        
