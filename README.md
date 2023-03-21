# GFLower

Graph Convolutional assisted Federated Learning with Flower.

Product developed at the University of Cambridge based on: 
https://github.com/dawenzi098/SFL-Structural-Federated-Learning

## First time running

```
conda create --name gflower python=3.9

sh install_dependencies.sh 

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

pip install -e gflower/
```

Examples available in gflower/examples

