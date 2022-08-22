
<p float="left">
  <img src="imgs/Fake_TCIA_103.gif" width="300" />
  <img src="imgs/Real_TCIA_103.gif" width="300" /> 
</p>

<br><br><br>

# Patient Specific 4DCT

Work In progress

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/cyiheng/InProgress
cd InProgress
```

- Install dependencies:
```bash
pip install -r requirements.txt
```

### Download pretrained weight
Download a pre-trained model with `./scripts/download_pretrained_model.sh`.

```bash
bash ./scripts/download_pretrained_model.sh
```

### Run Inference
- Test the model after download pretrained weight:
```bash
python ./test_3D.py --dataroot ./datasets/ --name pretrained_model --model test --dataset_mode test --num_test 1
```
- The test results will be saved by default in the directory : `./results/pretrained_model/`
- You can change the range of alpha with the following options: 
```bash
--alpha_min : the minimum value of the generated images (default: 0.0)
--alpha_max : the maximum value of the generated images (default: 1.0)
--alpha_step: the number of intermediate images between the range [alpha_min, alpha_max] (default: 5)
```
- **NOTE:** The minimum and maximum alphas values are 0.0 and 3.0 respectively. 

- You can select the loop mode for the final generated 4DCT: 
```bash
--loop : change how the 4DCT is stacked from the generated images
	0 : Only Source phase to alpha-inhale phase - [0,1,...,alpha]
	1 : Source to alpha-inhale phase and then add reversed images (alpha-inhale phase to source) - [0,1,...,alpha,...,2,1]
	2 : Same as 1, but with a step of 2, avoiding re-using a same image twice - [0,2,4,...,alpha,...,5,3,1]
```

### Datasets

(TODO) Ref: DIRLAB & TCIA for using their data as example

(TODO) Preprocess : 
- Filename : `LungCT_0000_0000.nii.gz`
- Need to have the same orientation (RAI)
- Image bounding box (need to keep the surface)
- Isotropic (plastimatch command)
- Resample to 128x128x128

## Citation
If you use this code for your research, please cite our papers.
```
TODO
```

## Acknowledgments
Our code is inspired by :
- [CycleGAN and pix2pix in Pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
- [vox2vox](https://github.com/enochkan/vox2vox)

