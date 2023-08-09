# HiSIF-DTA
Code and Dataset for "HiSIF-DTA: A Universal Hierarchical Semantic Information Fusion Framework for Drug-Target Affinity Prediction".

Install
---

* __download project__

  Downloading this project to your local server by running `git clone https://github.com/bixiangpeng/HiSIF-DTA/`.


* __environment configuration__

  :key: Create and activate the conda virtual environment: `conda create -n HiSIF python==3.8` and `conda activate HiSIF`.
  
  :key: Install needed python packages: ` pip install -r requirements.txt`.

Data preparation
---

There are three benchmark datasets adopted in our paper, including `Davis`, `Kiba` and `Human`. All pre-processed data for these datasets can be downloaded from this link.  The `data.zip` file is organized as follows:

```text
├── data
    ├── davis / kiba / Human                   - dataset name
        ├── PPI                                
            ├── ppi_data.pkl                   - PPI graph data ( A tuple inluding adjacency matrix, feature matrix, and id_maping. )
        ├── mol_data.pkl                       - Molecular graph data ( A tuple including node numbers, feature matrix, and adjacency matrix )
        ├── pro_data.pkl                       - Protein graph data ( A tuple including node numbers, feature matrix, and adjacency matrix )
        ├── train.csv                          - Train set
        └── test.csv                           - Test / Validation set
```
:bulb: Noting that the splitting methods of `Davis` and `Kiba` datasets are consistent with DGraphDTA (Jiang et al.) . As for `Human` dataset, a typical method of random five-fold cross-validation was utilized for data splitting. 

1. Davis

   Use `esm` to load the main ES module and export it as CommonJS.

    __index.js__
    ```js
    // Set options as a parameter, environment variable, or rc file.
    require = require("esm")(module/*, options*/)
    module.exports = require("./main.js")
    ```
    __main.js__
    ```js
    // ESM syntax is supported.
    export {}
    ```
    :bulb: These files are automagically created with `npm init esm` or `yarn create esm`.

2. Enable `esm` for local runs:

    ```shell
    node -r esm main.js
    ```
    :bulb: Omit the filename to enable `esm` in the REPL.

Features
---







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
