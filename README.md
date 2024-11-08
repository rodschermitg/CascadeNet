# CascadeNet: Convolutional Neural Networks for Longitudinal MRI-based Glioma Growth Modeling

![Cascade Net overview](https://github.com/rodschermitg/CascadeNet/blob/media/cascade_net_overview.png?raw=true)

## Description

**TODO**: Update description with reference to conference paper

## Getting Started

### Installation

To install the required packages, run

```
git clone https://github.com/rodschermitg/CascadeNet
pip install -r requirements.txt
```

### Dataset

To replicate our experiments, download the [LUMIERE dataset](https://doi.org/10.1038/s41597-022-01881-7) and place each patient directory in the corresponding `data/processed/patients/train` and `data/processed/patients/test` directories (refer to the [`dataset.json`](data/processed/patients/dataset.json) file to verify the train-test split used in our experiments).

If you want to use your own dataset (or your own train-test split), make sure that the directory structure and naming format follows that of the LUMIERE dataset, place each patient directory in the corresponding directories mentioned above and run

```
python3 -m scripts.create_dataset_json
```

to generate your own corresponding `dataset.json` file. 

### Run experiments

The experiment hyperparameters can be adjusted in the [`config.py`](src/config.py) file. Changes to the image and segmentation augmentation pipeline can be made in the [`transforms.py`](src/transforms.py) file.

To train the Cascade Net, run 

```
python3 src/train.py
```

and to evaluate the Cascade Net, you can run one of the following scripts:

```
python3 src/test.py
python3 src/predict.py
```

See the [`scripts`](scripts) directory for additional helper scripts for model and data evaluation.
