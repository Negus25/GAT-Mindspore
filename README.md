# Mindspore Graph Attention Network(GAT)

This is a MindSpore implementation of the Graph Attention Network (GAT) model, originally proposed by Veličković and colleagues in 2017 (https://arxiv.org/abs/1710.10903).
## Citation

If you use this implementation in your research, please cite the original paper:  

@article{velickovic2018graph,  
title="{Graph Attention Networks}",  
author={Veli{\v{c}}kovi{'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{'{o}}, Pietro and Bengio, Yoshua},  
journal={International Conference on Learning Representations},  
year={2018},  
url={https://openreview.net/forum?id=rJXMpikCZ},  
note={accepted as poster},  
}

## Regarding the Cora dataset:

Cora is a dataset containing 2708 scientific papers, grouped into seven distinct categories. The citation network comprises 10556 connections. Each paper is represented by a binary word vector, which indicates whether a particular word from the 1433-word dictionary is present or absent.

## Cora_v2 Data:

  - Total Nodes: 2708
  - Total Edges: 10556
  - Number of Class: 7

## Label split:

  - Training: 140
  - Validation: 500
  - Testing: 1000
# Performances
Transductive Learning  
The final accuracy is between 84 % and 85 % for epochs = 1000.

# Environment
- GPU
- MindSpore version:  2.0.0rc1.dev20230416
