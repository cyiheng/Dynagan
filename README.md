
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

