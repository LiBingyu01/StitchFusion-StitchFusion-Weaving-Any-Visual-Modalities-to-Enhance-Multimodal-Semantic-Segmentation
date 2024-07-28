<div align="center"> 

## StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation

</div>


</div>

## Introduction

Multimodal semantic segmentation shows significant potential for enhancing segmentation accuracy in complex scenes. However, current methods often incorporate specialized feature fusion modules tailored to specific modalities, thereby restricting input flexibility and increasing the number of training parameters. To address these challenges, we propose \textit{StitchFusion}, a straightforward yet effective modal fusion framework that integrates large-scale pre-trained models directly as encoders and feature fusers. This approach facilitates comprehensive multi-modal and multi-scale feature fusion, accommodating any visual modal inputs.
Specifically, Our framework achieves modal integration during encoding by sharing multi-modal visual information. To enhance information exchange across modalities, we introduce a multi-directional adapter module (\textit{MultiAdapter}) to enable cross-modal information transfer during encoding. By leveraging MultiAdapter to propagate multi-scale information across pre-trained encoders during the encoding process, StitchFusion achieves multi-modal visual information integration during encoding. Extensive comparative experiments demonstrate that our model achieves state-of-the-art performance on four multi-modal segmentation datasets with minimal additional parameters. Furthermore, the experimental integration of MultiAdapter with existing Feature Fusion Modules (FFMs) highlights their complementary nature.

## Updates
- [x] 07/2024: init repository.
- [x] 07/2024: release the code for StitchFusion.

## StitchFusion model

<div align="center"> 

![StitchFusion](./figs/StitchFusion_1.png)
**Figure:** Comparison of different model fusion paradigms.

![StitchFusion](./figs/StitchFusion_2.png)
**Figure:** MultiAdapter Module For StitchFusion Framwork At Different Density Levels.

</div>

## Environment

First, create and activate the environment using the following commands: 
```bash
conda env create -f environment.yaml
conda activate StitchFusion
```

## Data preparation
Download the dataset:
- [MCubeS](https://github.com/kyotovision-public/multimodal-material-segmentation), for multimodal material segmentation with RGB-A-D-N modalities.
- [FMB](https://github.com/JinyuanLiu-CV/SegMiF), for FMB dataset with RGB-Infrared modalities.
- [PST](https://github.com/ShreyasSkandanS/pst900_thermal_rgb), for PST900 dataset with RGB-Thermal modalities.
- [DeLiver](https://github.com/jamycheung/DELIVER), for DeLiVER dataset with RGB-D-E-L modalities.
- [MFNet](https://github.com/haqishen/MFNet-pytorch), for MFNet dataset with RGB-T modalities.
Then, put the dataset under `data` directory as follows:

```
data/
├── MCubeS
│   ├── polL_color
│   ├── polL_aolp_sin
│   ├── polL_aolp_cos
│   ├── polL_dolp
│   ├── NIR_warped
│   ├── NIR_warped_mask
│   ├── GT
│   ├── SSGT4MS
│   ├── list_folder
│   └── SS
├── FMB
│   ├── test
│   │   ├── color
│   │   ├── Infrared
│   │   ├── Label
│   │   └── Visible
│   ├── train
│   │   ├── color
│   │   ├── Infrared
│   │   ├── Label
│   │   └── Visible
├── PST
│   ├── test
│   │   ├── rgb
│   │   ├── thermal
│   │   └── labels
│   ├── train
│   │   ├── rgb
│   │   ├── thermal
│   │   └── labels
├── DELIVER
|   ├── depth
│       ├── cloud
│       │   ├── test
│       │   │   ├── MAP_10_point102
│       │   │   │   ├── 045050_depth_front.png
│       │   │   │   ├── ...
│       │   ├── train
│       │   └── val
│       ├── fog
│       ├── night
│       ├── rain
│       └── sun
│   ├── event
│   ├── hha
│   ├── img
│   ├── lidar
│   └── semantic
├── MFNet
|   ├── img
|   └── ther
```

## Model Zoo
All .pth will release later.
### MCubeS

### FMB

### PST900

### DeLiver

### MFNet

## Training

Before training, please download [pre-trained SegFormer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5), and put it in the correct directory following this structure:

```text
checkpoints/pretrained/segformer
├── mit_b0.pth
├── mit_b1.pth
├── mit_b2.pth
├── mit_b3.pth
└── mit_b4.pth
```

To train StitchFusion model, please update the appropriate configuration file in `configs/` with appropriate paths and hyper-parameters. Then run as follows:

```bash
cd path/to/StitchFusion
conda activate StitchFusion

python -m tools.train_mm --cfg configs/mcubes_rgbadn.yaml

python -m tools.train_mm --cfg configs/fmb_rgbt.yaml

python -m tools.train_mm --cfg configs/pst_rgbt.yaml
```


## Evaluation
To evaluate StitchFusion models, please download respective model weights ([**GoogleDrive**](https://drive.google.com/drive/folders/1OPr7PUrL7hkBXogmHFzHuTJweHuJmlP-?usp=sharing)) and save them under any folder you like.


Then, update the `EVAL` section of the appropriate configuration file in `configs/` and run:

```bash
cd path/to/StitchFusion
conda activate StitchFusion

python -m tools.val_mm --cfg configs/mcubes_rgbadn.yaml

python -m tools.val_mm --cfg configs/fmb_rgbt.yaml

python -m tools.val_mm --cfg configs/pst_rgbt.yaml

python -m tools.val_mm --cfg configs/deliver.yaml

python -m tools.val_mm --cfg configs/mfnet_rgbt.yaml
```

## License
This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.


## Citations

## Acknowledgements
Our codebase is based on the following Github repositories. Thanks to the following public repositories:
- [DELIVER](https://github.com/jamycheung/DELIVER)
- [MMSFormer](https://github.com/csiplab/MMSFormer)
- [Semantic-segmentation](https://github.com/sithu31296/semantic-segmentation)

**Note:** This is a research level repository and might contain issues/bugs. Please contact the authors for any query.
