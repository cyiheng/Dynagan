<br>
<p align="center">
  <img src="imgs/real_fake.gif" width="600" />
</p>

# Patient specific 4D CT respiratory motion synthesis using generative adversarial networks

A general objective of radiotherapy treatment planning is to deliver the lowest radiation dose induced by imaging protocols. Four-dimensional computed tomography (4D CT) imaging is used routinely for respiratory motion synchronization in radiotherapy treatment planning. However, those images require a longer acquisition time leading to a higher radiation exposure, up to six times a standard 3D CT acquisition. There is therefore a clear need for alternative planning methods that would reduce the dose impact.

This study proposes a new deep learning architecture architecture to generate realistic respiratory motion from static 3D CT images tailored to the actual patient breathing dynamics. An image-to-image 3D generative adversarial network conditioned with a 3D CT image and an amplitude-based scalar value using a scalar injection mechanism based on an Adaptive Instance Normalization layer is proposed in this study. 

This repository shares source code to run inference and thus generate motion from a 3D CT image.

<p align="center">
  <img src="imgs/generation.png" width="800" />
</p>

If you use this code for your research, please cite our papers.
```
TODO
```

# Table of Contents
- [Intro](#Patient-specific-4D-CT-respiratory-motion-synthesis-using-generative-adversarial-networks)
- [Table of Contents](#table-of-contents)
- [Ready-to-go](#ready-to-go)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Download pretrained weight](#download-pretrained-weight)
  - [Datasets](#datasets)
  - [Run inference](#run-inference)
- [Acknowledgements](#acknowledgements)


# Ready-to-go

We made a simple application to try out the model.

> For [Windows user](https://ubocloud.univ-brest.fr/s/tqjfEe39Q3J8qyD) (~700MB)

> For [Linux user](https://ubocloud.univ-brest.fr/s/praTqmtddTdS6jH) (~2.4GB)

<p align="center">
  <img src="imgs/screenshot_interface.png" width="600" />
</p>

**NOTES:** 
- It can takes only one file at time (unlike the version in [Usage](#usage)
- The preprocessing is included in the application but might take some time to run depending of the image size, CPU and GPU.
- It doesn't need GPU, but it will take a longer time for preprocessing & generating phases.
- It takes some times to be launched, some warning might appear but nothing wrong. 

For more details about the alphas values and loop modes, please check [Run inference](#run-inference).

# Usage

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation

- Clone this repo:
```bash
git clone https://github.com/cyiheng/InProgress
cd InProgress
```

- Install dependencies:
```bash
pip install -r requirements.txt
```

## Download pretrained weight
Download a pre-trained model with `./scripts/download_pretrained_model.sh`.

```bash
bash ./scripts/download_pretrained_model.sh
```

## Datasets

As example, you can use the 4D-Lung dataset from The Cancer Imaging Archive.
```
Hugo, Geoffrey D., Weiss, Elisabeth, Sleeman, William C., Balik, Salim, Keall, Paul J., Lu, Jun, & Williamson, Jeffrey F. (2016). Data from 4D Lung Imaging of NSCLC Patients. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE
```
<p align="center">
  <img src="imgs/preprocessing.png" width="800" />
</p>

A Jupyter notebook is available to preprocess data into the input image format.
Before running the notebook, please check the following information:
- Fileformat supported: NifTI
- Filename: `LungCT_patient_phase.nii.gz (i.e: LungCT_0100_0005.nii.gz)`
- Initial files location: `./datasets/001_original/`
- Initial image orientation: RAI

- **NOTE:** We assume that the files are already convert from DICOM to NifTI format

The notenook needs the following tools:
- Lung segmentation : [lungmask](https://github.com/JoHof/lungmask)
- Several operation is based on SimpleITK : [SimpleITK](https://github.com/SimpleITK/SimpleITKPythonPackage)

After running the notebook, the dataset directories should be like following if all 4D-lung dataset is used:
```text
./datasets
├── 000_csv
│   ├── body_bb.csv
│   ├── lung_bb.csv
│   └── final_bb.csv
├── 001_original
│   ├── body
│   │   └── ...
│   ├── lung
│   │   └── ...
│   ├── tumor
│   │   └── ...
│   ├── LungCT_0100_0000.nii.gz
│   ├── ...
│   └── LungCT_0119_0009.nii.gz
├── 002_bounding_box
├── 003_128x128x128
└── imagesTs
```
The final images are in a shape of 128 x 128 x 128.
Please select the images you want to use as input for the model to the directory `imagesTs`


## Run Inference

- Test the model after download pretrained weight:
```bash
python ./test_3D.py --dataroot ./datasets/ --name pretrained_model --model test --dataset_mode test --num_test 1
```
- The test results will be saved by default in the directory : `./results/pretrained_model/`
- You can change the range of alpha with the following options: 
```bash
--alpha_min : the minimum value of the generated images (default: 0.0)
--alpha_max : the maximum value of the generated images (default: 2.0)
--alpha_step: the number of intermediate images between the range [alpha_min, alpha_max] (default: 5)
```
- **NOTE:** The minimum and maximum alphas values are 0.0 and 3.5 respectively. 

- You can select the loop mode for the final generated 4DCT: 
```bash
--loop : change how the 4DCT is stacked from the generated images
	0 : Only Source phase to alpha-inhale phase - [0,1,...,alpha]
	1 : Source to alpha-inhale phase and then add reversed images (alpha-inhale phase to source) - [0,1,...,alpha,...,2,1]
	2 : Same as 1, but with a step of 2, avoiding re-using a same image twice - [0,2,4,...,alpha,...,5,3,1]
```

- **NOTE:** For no GPU user, please add the option `--gpu_ids -1`, it will run on CPU instead.

# Acknowledgments
Our code is inspired by :
- [CycleGAN and pix2pix in Pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- [vox2vox](https://github.com/enochkan/vox2vox)
