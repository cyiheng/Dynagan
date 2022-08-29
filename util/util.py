"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import nibabel as nib

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

        
def mkdir(dirpath):
    """
    Create directory if no directory with the given path
    :param dirpath: directory path
    :return: nothing but create directory (if needed)
    """
    if not os.path.exists(dirpath):
        print("Creating directory at:", dirpath)
        os.makedirs(dirpath)
        
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
        
def inv_normalize(x, output_min = -1000, output_max = 1000):
    """
    Simple denormalization
    :param x: array to denormalize
    :param output_min: minimum image intensity
    :param output_max: maximum image intensity
    :return: denormalized image
    """
    output_range = output_max - output_min
    return output_range*(x + 1) / 2 + output_min

def save_as_3D(img_in, real_path, img_out, range_intensity=[-1000,1000]):
    """
    Save Tensor image to 3D nii.gz image.
    :param img_in: image to save (Tensor)
    :param real_path: input image path used as reference for header
    :param img_out: output image (nii.gz)
    :param range_intensity: range of image intensity to invert normalization ; default [-1000;1000]
    """
    # convert Tensor to numpy arrays
    img_arr = img_in.cpu().data.numpy()
    img_arr = img_arr[0,0,:,:,:]
    # img_arr = img_arr[:,:,:,-1].transpose(2, 1, 0) # Back to the good orientation
    img_arr = inv_normalize(img_arr, *range_intensity) # Denormalize

    # Get input affine and header 
    ref = nib.load(real_path[0])
    hdr, aff = ref.header, ref.affine
    
    # Save image
    nib.save(nib.Nifti1Image(img_arr, aff, hdr), img_out)

def save_as_4D(img_in, real_path, img_out):
    """
    Save Tensor image to 4D nii.gz image (for deformation vector field).
    :param img_in: image to save (Tensor)
    :param real_path: input image path used as reference for header
    :param img_out: output image (nii.gz)
    """
    # convert Tensor to numpy arrays
    img_arr = img_in.cpu().data.numpy()
    img_arr = img_arr[0,:,:,:,:].transpose(1,2,3,0) # Back to the good orientation

    # Copy Information to original image
    real = nib.load(real_path[0])
    real.header.set_data_dtype(np.float32) 
    hdr, aff = real.header, real.affine
    nib.save(nib.Nifti1Image(img_arr,aff,hdr), img_out)

def create_4DCT(real_path, fake_path, output_file, loop=0):
    """
    Pack images to generate 4DCT image
    :param real_path: input image path used as reference for header and first of 4DCT image
    :param fake_path: directory where the warped image are located
    :param output_file: output filename
    :param loop: mode of loop if needed ; default = 0 (no loop)
    """
    # Use the input image as the first of the 4DCT
    ref_img = nib.load(real_path[0])
    hdr, aff = ref_img.header, ref_img.affine
    res_img = ref_img.get_fdata()[..., np.newaxis] # add axis to stack
    
    # Get all warped image in the directory for one case
    img_list = sorted(glob.glob(fake_path + '*-warped.nii.gz'))
    
    if loop == 0:
        pass
    elif loop == 1:
        # EOI then IOE with full fake images
        reversed_list = img_list[::-1]
        img_list = img_list + reversed_list
    elif loop == 2:
        # EOI then IOE with every two images (no repetition)
        reversed_list = img_list[::-1]
        if len(reversed_list)%2:
        	start_idx = 1
        else:
            start_idx = 2
        img_list = img_list[::2] + reversed_list[start_idx::2]
    
    for img_file in img_list:
        fake_img = nib.load(img_file).get_fdata()[..., np.newaxis]
        res_img = np.concatenate((res_img,fake_img), axis=-1)
        
    # Save 4DCT image
    print('4DCT created as', output_file)
    nib.save(nib.Nifti1Image(res_img, aff, hdr), output_file)
