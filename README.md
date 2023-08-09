# HiSIF-DTA
Code and Dataset for "HiSIF-DTA: A Universal Hierarchical Semantic Information Fusion Framework for Drug-Target Affinity Prediction".
# Framework
![HiSIF-DTA architecture](https://github.com/bixiangpeng/HiSIF-DTA/blob/main/Framework.png)
## Requirement
Pytorch == 1.8.0

rdkit == 2022.03.2

torch-geometric == 2.0.4

autograd == 1.4

autograd-gamma == 0.5.0

biopython == 1.79

## Data Preparation

1. Pre-trained models and relevant results can be available at the [Link](https://pan.baidu.com/s/1kejvhktBZ6e0QcmbYWus5w?pwd=er8j).

2. The relevant data files can be available at the [Link](https://pan.baidu.com/s/1YC2n9G2oldSATik5fxLUDg?pwd=8dgu).
   
## Run
  1.Train
  
    1) For DTA task
    
      python train.py --model{0 for BUNet or 1 for TDNet } --dataset{davis or kiba}
      
    2) For CPI task
    
      python train_for_CPI.py --model{0 for BUNet or 1 for TDNet } --dataset{Human}

  2.Test
  
    python test.py --model{0 for BUNet or 1 for TDNet } --dataset{davis or kiba}
  
 
# Contact
We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.
