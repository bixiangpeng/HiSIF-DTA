# HiSIF-DTA
Code and Dataset for "HiSIF-DTA: A Universal Hierarchical Semantic Information Fusion Framework for Drug-Target Affinity Prediction".
# Framework
![HiSIF-DTA architecture](https://github.com/gu-yaowen/REDDA/blob/main/model_structure.png)
## Requirement
Pytorch == 1.8.0

rdkit == 2022.03.2

torch-geometric == 2.0.4

autograd == 1.4

autograd-gamma == 0.5.0

biopython == 1.79
## Run
  1.Train
  
    1) For DTA task
    
      python train.py --model{0 for BU_model or 1 for TD_model } --dataset{davis or kiba}
      
    2) For CPI task
    
      python train_for_CPI.py --model{0 for BU_model or 1 for TD_model } --dataset{Human}

  2.Test
  
    python test.py --model{0 for BU_model or 1 for TD_model } --dataset{davis or kiba}
  
 
# Contact
We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.
