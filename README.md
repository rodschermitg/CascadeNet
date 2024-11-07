# CascadeNet: Convolutional Neural Networks for Longitudinal MRI-based Glioma Growth Modeling

![Cascade Net architecture](https://private-user-images.githubusercontent.com/71263770/384088476-cb0614b5-1883-404e-bf28-64d7f0407ac6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzEwMDM4NjEsIm5iZiI6MTczMTAwMzU2MSwicGF0aCI6Ii83MTI2Mzc3MC8zODQwODg0NzYtY2IwNjE0YjUtMTg4My00MDRlLWJmMjgtNjRkN2YwNDA3YWM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMDclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTA3VDE4MTkyMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRjYjdhMmEzZWY2OTlkNWE0ZDY4ZjQ2ZjQyYTgyNGVjNzk1MTI2NTdjZjE5YmNiOTY4NmZiNTE4YzE1YzZhNDMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.eb5-LR__T6DXdnsOdILFb2my_gx26rqkfYEB9bAWyZE)

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
