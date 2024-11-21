# MGCN-for-Traffic-Prediction

The repo is the official implementation for the paper: [MGCN: Mamba-Integrated Spatiotemporal Graph Convolutional Network for Long-Term Traffic Forecasting]

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```
2. Train and Test

The settings for each experiments are given in the "configurations" folder.

We take the commands on PEMS04 for example.

- Step 1: set up configurations:

MGCN_/configurations/PEMS04_mgcn.conf

Parameters to be set:

num_of_vertices=307(The number of nodes in the dataset)

num_for_predict=12(The Predict step size)

len_input=96(The input step size)

points_per_hour=96(The input step size)

You can keep the default Settings for other parameters


- Step 2: train and test the model:

```
python - run.py
```

## Contact

If you have any questions or want to use the code, feel free to contact:220233409@seu.edu.cn
* 

If you find the repository is useful, please cite our paper. Thank you!
