from .base_model import BaseModel
from . import networks

import torch
import os
import pandas as pd
import numpy as np

from util.util import mkdir, save_as_3D, save_as_4D, create_4DCT
from util.spatialTransform import SpatialTransformer

class TrainModel(BaseModel):
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
        parser.set_defaults(netG='dvf', netD='dvf', dataset_mode='train')
        # parser.add_argument('--version', type=str, default='label', help='version of AdaIN (label | conv) or LabelGenerator (linear | repeat)')
        if is_train:
            parser.add_argument('--lambda_1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_2', type=float, default=1.0, help='weight for GAN loss')

        return parser

    def __init__(self, opt):
        """Initialize the train class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1', 'G_GAN', 'D_fake', 'D_real']
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        
        self.transform = SpatialTransformer([128,128,128]).cuda()
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G','D']
        else:  # during test time, only load G
            self.model_names = ['G']
        
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:  # only defined during training time
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netG_tumor.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(), self.netD_tumor.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """

        self.real_A = input['image_A'].unsqueeze(1).to(self.device)
        self.real_B = input['image_B'].unsqueeze(1).to(self.device)
        self.alpha = input['alpha'].to(self.device).float()
        self.dvf = input['dvf'].to(self.device)
        self.dvf_magn = torch.linalg.norm(self.dvf, dim = 1).unsqueeze(0)
        
        # self.dvf_x = input['dvf_x'].unsqueeze(1).to(self.device)
        # self.dvf_y = input['dvf_y'].unsqueeze(1).to(self.device)
        # self.dvf_z = input['dvf_z'].unsqueeze(1).to(self.device)  
        
        self.image_paths = input['fpath_A']
            
    def forward(self):
        """Run forward pass."""
        self.fake_B_dvf = self.netG(self.real_A, self.alpha) # DVF
        self.fake_B = self.transform(self.real_A, self.fake_B_dvf) # warp to image
        self.fake_magn = torch.linalg.norm(self.fake_B_dvf, dim = 1).unsqueeze(0) # calculate magnitude

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Image Discriminator
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.real_A.detach(), self.fake_B.detach(), self.fake_magn.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)   
        
        # Real
        pred_real = self.netD(self.real_A, self.real_B, self.dvf_magn)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        
        self.loss_D.backward()
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.real_A, self.fake_B, self.fake_magn)
        self.loss_G_L1 = self.criterionL1(self.fake_B_dvf, self.dvf) * self.opt.lambda_1
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_2
        
        self.loss_G = self.loss_G_GAN  + self.loss_G_L1
        
        self.loss_G.backward()
        
    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()
        self.optimizer_G.step()             # udpate G's weights
        
    def sample_voxel_volumes(self, epoch, image_folder):
        """Saves a generated sample
        :param image_folder: folder where images will be saved
        """

        print("Generates training samples")
        mkdir(image_folder)
        save_as_3D(self.real_A, self.image_paths, image_folder + 'epoch_%s_realA.nii.gz' % epoch)
        save_as_3D(self.real_B, self.image_paths, image_folder + 'epoch_%s_realB.nii.gz' % epoch)
        save_as_3D(self.fake_B, self.image_paths, image_folder + 'epoch_%s_fakeB.nii.gz' % epoch)            
        save_as_4D(self.fake_B_dvf, self.image_paths, image_folder + 'epoch_%s_fakeB_dvf.nii.gz' % epoch)