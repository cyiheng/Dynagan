"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import itertools
from .base_model import BaseModel
from . import networks
from util.util import mkdir, save_as_3D, save_as_4D, save_segmentation, run_reg, nii_arr, measure_surface, mask_volume
import os
import pandas as pd
import numpy as np
import util.pytorch_ssim
from util.spatialTransform import SpatialTransformer


class GanModel(BaseModel):
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
            parser.add_argument('--lambda_GAN_img', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN_dvf', type=float, default=1.0, help='weight for L1 loss')
            
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
        self.loss_names = ['G_GAN_img', 
                           'G_L1_img',
                           'G_L1_dvf',
                           'G_PS',
                           'D_fake_img',
                           'D_real_img',
                          ] 
        # , 'D_Dice_tumor', 'D_Dice_lobes' ]#, 'D_realmask', 'D_fakemask']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        # self.visual_names = ['image', 'phase', 'output']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.phase_names = ['phase_A', 'phase_B']
        self.max_correction_niter = 10
        self.updated_dvf = None
        self.mode= opt.scalar
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
            # self.netD_dvf = networks.define_D(3, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:  # only defined during training time
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSSIM = util.pytorch_ssim.SSIM3D(window_size = 11)
            
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
        """
        self.real_A = input['image_A'].unsqueeze(1).to(self.device)
        self.phase_A = input['phase_A'].to(self.device).float()
        
        self.real_B = input['image_B'].unsqueeze(1).to(self.device)
        self.phase_B = input['phase_B'].to(self.device).float()
        
        self.alpha = input['alpha'].to(self.device).float()
        self.spacing_y = input['spacing_y'].to(self.device).float()
        self.dvf = input['dvf'].to(self.device)
        
        # self.dvf_x = input['dvf_x'].unsqueeze(1).to(self.device)
        # self.dvf_y = input['dvf_y'].unsqueeze(1).to(self.device)
        # self.dvf_z = input['dvf_z'].unsqueeze(1).to(self.device)
        
        self.image_paths = input['fpath_A']
        
        if not self.isTrain:
            self.tum_input = input['tum_A'].unsqueeze(1).to(self.device).float()
        
    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_B_dvf = self.netG(self.real_A, self.alpha) # DVF
        self.fake_B = self.transform(self.real_A, self.fake_B_dvf) # warp to image
        # self.fake_B_dvf, self.fake_B = self.netG(self.real_A, self.pvd)
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Image Discriminator
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.real_A.detach(), self.fake_B.detach())
        self.loss_D_fake_img = self.criterionGAN(pred_fake, False)   
        
        # Real
        pred_real = self.netD(self.real_A, self.real_B)
        self.loss_D_real_img = self.criterionGAN(pred_real, True)
        
        self.loss_D = (self.loss_D_fake_img + self.loss_D_real_img) * 0.5
        
        self.loss_D.backward()
        
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.real_A, self.fake_B)
        self.loss_G_GAN_img = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN_img
        self.loss_G_L1_img = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1_img
        self.loss_G_L1_dvf = self.criterionL1(self.fake_B_dvf, self.dvf) * self.opt.lambda_L1_dvf
        self.loss_G_PS = self.calculate_edge_loss_dvf(self.dvf, self.fake_B_dvf) * 10
        
        self.loss_G = self.loss_G_GAN_img  + self.loss_G_L1_dvf + self.loss_G_L1_img # + self.loss_G_GradSmooth
        
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
            alphas = pd.read_csv(image_folder + 'alpha_correction.csv')
            cascade = False
            linear = False
            if "in_000" in os.path.basename(self.image_paths[0]) :
                # Get input phase
                case = os.path.basename(self.image_paths[0]).split('_')[0]
                p = os.path.basename(self.image_paths[0])
                save_as_3D(img_in = self.real_A, real_path = self.image_paths, img_out = image_folder + 'input/Input_%s' % (p) )
                
                # Read testing file
                info = pd.read_csv(self.testfile)
                # ftmp = info[(info['name'].str.match(case))&(info['name'].str.contains("in"))] # Find the same case with different phase
                ftmp = info[(info['name'].str.match(case))] # Find the same case with different phase
                start = 0
                for idx, row in ftmp[:].iterrows():
                    # Get phase and name & prepare output file name
                    alpha = row['volume']
                    phase = row['name']
                    
                    fake_output_file = image_folder + 'Fake_%s' % (phase)
                    fake_mask_file = image_folder + 'mask/Fake_%s' % (phase.replace('.nii','-mask.nii'))
                    fake_tum_file = image_folder + 'tumor/Fake_%s' % (phase.replace('.nii','-tumor.nii'))
                    fake_warped_file = image_folder + 'warped/Fake_%s' % (phase.replace('.nii','-warped.nii'))
                    fake_dvf_file = image_folder + 'dvf/Fake_%s' % (phase.replace('.nii','-dvf.nii'))
                        
                    # Classic method : generate with corresponding phase
                    if linear:
                        alphas = np.linspace(0.1, 3, 21)
                        for i, alpha in enumerate(alphas):
                            if self.mode == "pvd":
                                alpha = (torch.tensor([[alpha]]).to(self.device).float()-self.phase_A)/self.phase_A*100

                            elif self.mode == "surface":
                                alpha = (torch.tensor([[alpha]]).to(self.device).float()-self.phase_A)
                                # print("alpha:", alpha)

                            
                            fake_B_dvf = self.netG(self.real_A, alpha)
                            fake_B_tum = self.transform_tum(self.tum_input, fake_B_dvf) # warp tumor
                            fake_B = self.transform(self.real_A, fake_B_dvf) # warp to image

                            fake_warped_file_new = fake_warped_file.replace('000-warped', '{:02d}-warped'.format(i))
                            save_as_3D(fake_B, self.image_paths, fake_warped_file_new)
                            save_as_3D(fake_B_tum, self.image_paths, fake_tum_file.replace('000-tumor', '{:02d}-tumor'.format(i)), out_max = 1, out_min =-1)
                            mask_volume(fake_warped_file_new, image_folder + 'mask/')
                            
                            save_as_4D(fake_B_dvf, self.image_paths, fake_dvf_file.replace('000-dvf', '{:02d}-dvf'.format(i)))
                            # Clear memory cache, otherwise not enough gpu memory to generate new image at next iteration
                            del fake_B
                            del fake_B_dvf
                            del fake_B_tum
                            torch.cuda.empty_cache()
                        continue
                    if cascade:
                        if start == 0:
                            print("not need to generate itself now")
                            alpha_pred = torch.tensor([[alpha]]).to(self.device).float()
                            phase_pred = phase         
                            start = start + 1
                            continue
                        else:
                            print()
                            print("      Phase to treat:", phase)
                            print(" Pred phase to treat:", phase_pred)
                            print(" Init Alpha to treat:", alpha)
                            print(" Pred Alpha to treat:", alpha_pred)

                            if self.mode == "pvd":
                                alpha_used = (torch.tensor([[alpha]]).to(self.device).float()-alpha_pred)/alpha_pred*100
                                alpha_used = correction*new_pvd
                            elif self.mode == "surface":
                                alpha_used = (torch.tensor([[alpha]]).to(self.device).float()-alpha_pred)
                                # alpha_used = alpha_used + (alpha_used * 0.5)
                            print("          Alpha used:", alpha_used)

                            # Stock alpha and phase for the next one
                            alpha_pred = torch.tensor([[alpha]]).to(self.device).float()
                            phase_pred = phase

                            if start == 1:
                                # The first case, we use input file (already "warped")
                                print('Use image:', self.image_paths[0])
                                fake_B_dvf, fake_B = self.generate_fake(self.real_A, alpha_used, correction=False)
                                # fake_B = self.netG(self.real_A, new_pvd)
                                moving_path = self.image_paths[0]
                            else:
                                # Use previous warped generated image for generate the next phase
                                print('Use image:', pred_fake_file)
                                moving_path = pred_fake_file
                                # moving_path = self.image_paths[0]
                                # Small preprocess
                                img_arr = nii_arr(pred_fake_file)
                                img_arr = img_arr.clip(-1000,1000)
                                img_arr = (img_arr-img_arr.min())/(img_arr.max()-img_arr.min())*2-1
                                # fake_B = self.netG(reg_fake_B, new_pvd)
                                pred_fake_B = torch.from_numpy(img_arr.astype(np.float32)).unsqueeze(0).unsqueeze(1).to(self.device)
                                
                                fake_B_dvf, fake_B = self.generate_fake(pred_fake_B, alpha_used, correction=False)

                            fake_surface = measure_surface(self.real_A, fake_B, self.spacing_y)
                            print('      Alpha measured:', fake_surface)
                            new_row = pd.DataFrame({'name':[os.path.basename(fake_output_file)], 
                                                    'alpha_initial':[self.alpha.cpu().numpy()[0][0]], 
                                                    'alpha_used':[alpha_used.cpu().numpy()[0][0]], 
                                                    'alpha_measured':[fake_surface]})
                            alphas = pd.concat([alphas, new_row], ignore_index=True)


                            save_as_3D(fake_B, self.image_paths, fake_warped_file)
                            # print("save : ", fake_warped_file)
                            save_as_4D(fake_B_dvf, self.image_paths, fake_dvf_file)
                            # print("save : ", fake_dvf_file)
                            # Stock previous warped file for next iteration
                            pred_fake_file = fake_warped_file
                            start = start + 1
                        
                    else:
                        if 'in_000' in phase:
                            continue
                        if self.mode == "pvd":
                            self.alpha = (torch.tensor([[alpha]]).to(self.device).float()-self.phase_A)/self.phase_A*100

                        elif self.mode == "surface":
                            self.alpha = (torch.tensor([[alpha]]).to(self.device).float()-self.phase_A)
                            print("alpha initial:", self.alpha)
                            alpha_used = self.alpha * 2

                        fake_B_dvf, fake_B = self.generate_fake(self.real_A, alpha_used, correction=False)
                        fake_B_tum = self.transform_tum(self.tum_input, fake_B_dvf) # warp tumor
                        
                        # Measure surface amplitude
                        fake_surface = measure_surface(self.real_A, fake_B, self.spacing_y)
                        print('alpha measured', fake_surface)
                        new_row = pd.DataFrame({'name':[os.path.basename(fake_output_file)],'alpha_initial':[self.alpha.cpu().numpy()[0][0]],'alpha_used':[alpha_used.cpu().numpy()[0][0]],'alpha_measured':[fake_surface]})
                        alphas = pd.concat([alphas, new_row], ignore_index=True)
                        
                        save_as_3D(fake_B, self.image_paths, fake_warped_file)
                        save_as_3D(fake_B_tum, self.image_paths, fake_tum_file.replace('000-tumor', '{}-tumor'.format(phase.split('_')[2])), out_max = 1, out_min =-1)
                        mask_volume(fake_warped_file, image_folder + 'mask/')
                        # print("save : ", fake_warped_file)
                        save_as_4D(fake_B_dvf, self.image_paths, fake_dvf_file)
                        # print("save : ", fake_dvf_file)
   
                    
                    # Clear memory cache, otherwise not enough gpu memory to generate new image at next iteration
                    del fake_B
                    del fake_B_dvf
                    del fake_B_tum
                    torch.cuda.empty_cache()
                    alphas.to_csv(image_folder + 'alpha_correction.csv', index=False)
                start = 0
    def calculate_edge_loss_dvf(self, src_dvf, tgt_dvf):
        # grad_src = SpatialGradient3d()(src)
        # grad_tgt = SpatialGradient3d()(tgt)
        src_x = src_dvf[:,0,:,:,:]
        src_y = src_dvf[:,1,:,:,:]
        src_z = src_dvf[:,2,:,:,:]
        tgt_x = tgt_dvf[:,0,:,:,:]
        tgt_y = tgt_dvf[:,1,:,:,:]
        tgt_z = tgt_dvf[:,2,:,:,:]

        NGF = 3-(src_x*tgt_x + src_y*tgt_y + src_z*tgt_z)

        NGFM = torch.mean(NGF)
        return NGFM

    def generate_fake(self, input_img, scalar, correction=False, scalar_updated=False, iter_n=0, threshold=0.25):
        """
        Generate fake ddf and fake warped image. If necessary, update alpha value

        :param input_img: Tensor of input image
        :param scalar: Alpha value
        :param correction: if True, the alpha will be updated if necessary (default=False)
        :param threshold: error amplitude allowed
        :param scalar_update: if True, the scalar will be updated depending of the previous iteration (default=False)
        :return fake_dvf, fake_img: DDF and warped image

        """
        # Generate image
        fake_dvf = self.netG(input_img, scalar)
        fake_img = self.transform(input_img, fake_dvf)
        
        if correction:
            # Measure surface amplitude
            fake_surface = measure_surface(input_img, fake_img, self.spacing_y)

            # Calculate error
            error = self.alpha - fake_surface
            # print('Error between input and fake surface measure : ', error)

            # Correct if necessary
            # if not (error >= -0.1 and error <= 0):

            if torch.abs(error) >= threshold and iter_n < self.max_correction_niter:
                if not scalar_updated:
                    update_scalar = self.alpha + error
                else: # If already corrected, use corrected value to update
                    update_scalar = scalar + error

                iter_n = iter_n + 1
                    
                # Clear cache
                del fake_img
                del fake_dvf
                torch.cuda.empty_cache()

                # print('Updating scalar value from {} to {}'.format(scalar.cpu().data.numpy(), update_scalar.cpu().data.numpy()))
                return self.generate_fake(input_img, update_scalar, correction=True, scalar_updated=True, iter_n=iter_n)
            else:
                print('Alpha asked: ', self.alpha) 
                print('Alpha used : ', scalar)
                print('n_iter: ', iter_n)

                return fake_dvf, fake_img
        else:
            return fake_dvf, fake_img