"""
Dynagan basic model
"""
import torch
import itertools
from .base_model import BaseModel
from . import networks
from util.util import mkdir, save_as_3D, save_as_4D
import os
import pandas as pd
import numpy as np
from util.spatialTransform import SpatialTransformer



class DynaganModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(netG='mask', netD='mask', dataset_mode='mask')
        # parser.add_argument('--version', type=str, default='label', help='version of AdaIN (label | conv) or LabelGenerator (linear | repeat)')
        if is_train:
            parser.add_argument('--lambda_L1_img', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L1_dvf', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for L1 loss')
            
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 
                           'G_L1_img',
                           'G_L1_dvf',
                           'D_fake',
                           'D_real',
                          ] 
        # , 'D_Dice_tumor', 'D_Dice_lobes' ]#, 'D_realmask', 'D_fakemask']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        # self.visual_names = ['image', 'phase', 'output']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.phase_names = ['phase_A', 'phase_B']
        self.ordered = 0
        self.mode= opt.scalar
        self.saveA = True
        self.testfile = opt.csv_path
        self.transform = SpatialTransformer([128,128,128]).cuda()
        self.transform_tum = SpatialTransformer([128,128,128], 'nearest').cuda()
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
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        self.real_A = input['image_A'].unsqueeze(1).to(self.device)
        self.phase_A = input['phase_A'].to(self.device).float()
        
        self.real_B = input['image_B'].unsqueeze(1).to(self.device)
        self.phase_B = input['phase_B'].to(self.device).float()
        
        self.alpha = input['alpha'].to(self.device).float()
        self.spacing_y = input['spacing_y'].to(self.device).float()
        self.dvf = input['dvf'].to(self.device)
        self.dvf_magn = torch.linalg.norm(self.dvf, dim = 1).unsqueeze(0)
        
        self.image_paths = input['fpath_A']
        if not self.isTrain:
            self.tum_input = input['tum_A'].unsqueeze(1).to(self.device).float()
        
    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
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
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
        
        self.loss_G_L1_img = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1_img
        self.loss_G_L1_dvf = self.criterionL1(self.fake_B_dvf, self.dvf) * self.opt.lambda_L1_dvf
        
        self.loss_G = self.loss_G_GAN  + self.loss_G_L1_dvf + self.loss_G_L1_img # + self.loss_G_GradSmooth
        
        self.loss_G.backward()
        
    def optimize_parameters(self, n_iter):
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
        
# -------- Others -------- 
    def sample_voxel_volumes(self, epoch, image_folder='./results/'):
        """Saves a generated sample from the validation set
        :param epoch: for name purpose, give at which epoch the sample are generated during training
        :param folder: folder where images will be saved
        """
        
        if self.isTrain:
            print("Generates training samples")
            # image_folder = "%s3D_train_%s/" % (folder)
            mkdir(image_folder)
            save_as_3D(self.real_A, self.image_paths, image_folder + 'epoch_%s_realA.nii.gz' % epoch)
            save_as_3D(self.real_B, self.image_paths, image_folder + 'epoch_%s_realB.nii.gz' % epoch)
            save_as_3D(self.fake_B, self.image_paths, image_folder + 'epoch_%s_fakeB.nii.gz' % epoch)            
            save_as_4D(self.fake_B_dvf, self.image_paths, image_folder + 'epoch_%s_fakeB_dvf.nii.gz' % epoch)
            
        else:
            # Create folders for output file
            mkdir(image_folder)
            mkdir(image_folder+'tumor/')
            mkdir(image_folder+'dvf/')
            mkdir(image_folder+'warped/')
            mkdir(image_folder+'input/')
            mkdir(image_folder+'mask/')
            cascade = False
            linear = True
            if "T50" in os.path.basename(self.image_paths[0]) or  "_0_in" in os.path.basename(self.image_paths[0]) or "eBHCT" in os.path.basename(self.image_paths[0])  or "in_000" in os.path.basename(self.image_paths[0]) :
                # Get input phase
                case = os.path.basename(self.image_paths[0]).split('_')[0]
                p = os.path.basename(self.image_paths[0])
                save_as_3D(img_in = self.real_A, real_path = self.image_paths, img_out = image_folder + 'input/Input_%s' % (p) )
                
                # Read testing file
                info = pd.read_csv(self.testfile)
                # ftmp = info[(info['name'].str.match(case))&(info['name'].str.contains("in"))] # Find the same case with different phase
                ftmp = info[(info['name'].str.match(case))] # Find the same case with different phase
                for idx, row in ftmp[:].iterrows():
                    # Get phase and name & prepare output file name
                    volume = row['volume']
                    phase = row['name']
                    
                    fake_output_file = image_folder + 'Fake_%s' % (phase)
                    fake_mask_file = image_folder + 'mask/Fake_%s' % (phase.replace('.nii','-mask.nii'))
                    fake_tum_file = image_folder + 'tumor/Fake_%s' % (phase.replace('.nii','-tumor.nii'))
                    fake_warped_file = image_folder + 'warped/Fake_%s' % (phase.replace('.nii','-warped.nii'))
                    fake_dvf_file = image_folder + 'dvf/Fake_%s' % (phase.replace('.nii','-dvf.nii'))
                    
                    
                    if "T50" in os.path.basename(self.image_paths[0]) :
                        if self.mode == "pvd":
                            alpha = (torch.tensor([[volume]]).to(self.device).float()-self.phase_A)/self.phase_A*100

                        elif self.mode == "surface":
                            if case in ['001','002','003']:
                                correction = 1.5
                            elif case in ['004']:
                                correction = 4
                            elif case in ['006','007','008','009','010']:
                                correction = 1/2
                            else :
                                correction = 1
                            alpha = (torch.tensor([[volume]]).to(self.device).float()-self.phase_A)*correction
                            print("alpha:", alpha)

                        fake_B_dvf = self.netG(self.real_A, alpha)
                        fake_B = self.transform(self.real_A, fake_B_dvf) # warp to image
                        
                        save_as_3D(fake_B, self.image_paths, fake_warped_file)
                        print("save : ", fake_warped_file)
                        save_as_4D(fake_B_dvf, self.image_paths, fake_dvf_file)
                        print("save : ", fake_dvf_file)
                        continue
                        
                    # Classic method : generate with corresponding phase
                    if linear == False:
                        volumes = np.linspace(0, 1, 11)
                        for volume in volumes:
                            if self.mode == "pvd":
                                alpha = (torch.tensor([[volume]]).to(self.device).float()-self.phase_A)/self.phase_A*100

                            elif self.mode == "surface":
                                alpha = (torch.tensor([[volume]]).to(self.device).float()-self.phase_A)
                                print("alpha:", alpha)

                            
                            fake_B_dvf = self.netG(self.real_A, alpha)
                            fake_B = self.transform(self.real_A, fake_B_dvf) # warp to image

                            save_as_3D(fake_B, self.image_paths, fake_warped_file.replace('.nii', '{:.03f}.nii'.format(volume)))
                            save_as_4D(fake_B_dvf, self.image_paths, fake_dvf_file.replace('.nii', '{:.03f}.nii'.format(volume)))
                            # Clear memory cache, otherwise not enough gpu memory to generate new image at next iteration
                            del fake_B
                            del fake_B_dvf
                            torch.cuda.empty_cache()
                        continue
                    if cascade == False:
                        if self.mode == "pvd":
                            alpha = (torch.tensor([[volume]]).to(self.device).float()-self.phase_A)/self.phase_A*100

                        elif self.mode == "surface":
                            alpha = (torch.tensor([[volume]]).to(self.device).float()-self.phase_A)
                            print("alpha:", alpha)

                        fake_B_dvf = self.netG(self.real_A, alpha)
                        fake_B = self.transform(self.real_A, fake_B_dvf) # warp to image
                        
                        save_as_3D(fake_B, self.image_paths, fake_warped_file)
                        print("save : ", fake_warped_file)
                        save_as_4D(fake_B_dvf, self.image_paths, fake_dvf_file)
                        print("save : ", fake_dvf_file)
                    
                    # Clear memory cache, otherwise not enough gpu memory to generate new image at next iteration
                    del fake_B
                    del fake_B_dvf
                    torch.cuda.empty_cache()
                    
