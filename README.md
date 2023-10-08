
# Under Construction...

# HiSIF-DTA
A repo for "HiSIF-DTA: A Hierarchical Semantic Information Fusion Framework for Drug-Target Affinity Prediction".

Exploring appropriate protein representation methods and improving protein information abundance is a critical step in enhancing the accuracy of DTA prediction. Recently, numerous deep learning-based models have been proposed to utilize **sequential** or **structural** features of target proteins.

However, these models capture only **_low-order semantics_** that exists in a single protein, while the **_high-order semantics_** abundant in biological networks are largely ignored. In this article, we propose **HiSIF-DTA—a hierarchical semantic information fusion framework for DTA prediction**. 

In this framework, a hierarchical protein graph is constructed that includes not only contact map as **_low-order structural semantics_** but also protein-protein interaction network (PPI) as **_high-order functional semantics_**. Particularly, two distinct hierarchical fusion strategies (i.e., **_Top-down_** and **_Bottom-Up_**) are designed to integrate the different protein semantics, therefore contributing to a richer protein representation. **Comprehensive experimental results demonstrate that HiSIF-DTA outperforms current state-of-the-art methods for prediction on the benchmark datasets of DTA task**.

![HiSIF-DTA architecture](https://github.com/bixiangpeng/HiSIF-DTA/blob/main/Framework.jpg)
## Requirements
---

* ### Download projects

   Download the GitHub repo of this project onto your local server: `git clone https://github.com/bixiangpeng/HiSIF-DTA`


* ### Configure the environment manually

   Create and activate virtual env: `conda create -n HiSIF python=3.7 ` and `conda activate HiSIF`
   
   Install specified version of pytorch: ` conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge`
   
   Install other python packages:
   ```shell
   pip install -r requirements.txt \
   && pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html \
   && pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html \
   && pip install torch-spline-conv==1.2.1 -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
   ```
   
   :bulb: Note that the operating system we used is `ubuntu 22.04` and the version of Anaconda is `23.3.1`.


* ### Docker Image

    We also provide the Dockerfile to build the environment, please refer to the Dockerfile for more details. Make sure you have Docker installed locally, and simply run following command:
   ```shell
   # Build the Docker image
   sudo docker build --build-arg env_name=HiSIF -t hisif-image:v1 .
   # Create and start the docker container
   sudo docker run --name hisif-con --gpus all -it hisif-image:v1 /bin/bash
   # Check whether the environment deployment is successful
   conda list 
   ```
  
##  Usage
---
* ### Contents page

   ```text
       >  HiSIF-DTA
          ├── baselines                       - Baseline models directory. All the baseline models we re-trained can be found in this directory.
          ├── data                           - Data directory. The detailed information can be found in next section.
          ├── models                          
          │   ├── HGCN.py                    - Original model file, which includes both Top-Down (TDNet) and Bottom-Up(BUNet) semantic fusion models.
          │   ├── HGCN_for_CPI.py            - A model modified for datasets (Human) with large numbers of proteins.
          │   └── HGCN_for_Ablation.py       - Three ablation variants we used in this study.
          ├── results                         - The reslut directory storing the experimental results and pre-trained models.
          │   └── davis / kiba / Human      
          │       ├── pretrained_BUNet.csv   - A CSV file recording the optimal predicting results of BUNet on davis/kiba/Human. 
          │       ├── pretrained_BUNet.model - A file recording the optimal model parameters of BUNet on davis/kiba/Human.
          │       ├── pretrained_TDNet.csv
          │       └── pretrained_TDNet.model
          ├── generate_contact_map.py         - A Python script used to generate the contact map based on PDB files.
          ├── create_data.py                  - A python script used to convert original data to the input data that model needed.
          ├── utils.py                        - A python script recording the various tools needed for training.
          ├── training_for_DTA.py             - A python script used to train the model on DTA dataset (davis or kiba).
          ├── training_for_CPI.py             - A python script used to train the model on CPI dataset (Human).
          ├── test_for_DTA.py                 - A python script that reproduces the DTA prediction results using the pre-trained models.
          ├── test_for_CPI.py                 - A python script that reproduces the CPI prediction results using the pre-trained models.
          ├── test_for_Ablation.py            - A python script that reproduces the ablation results using the pre-trained models. 
          ├── grad_pre.py                     - A python script using backpropagation gradients to predict protein binding pockets.
          ├── requirements.txt                - A txt file recording the python packages that model depend on to run.
          ├── Dockerfile                      - A file used to build the environment image via Docker.
          └── experimental_results.ipynb      - A notebook indicating the prediction results of our models and other baseline models.
   ```
* ### Data preparation
  There are three benchmark datasets were adopted in this project, including two DTA datasets (`davis and kiba`) and a CPI dataset (`Human`).

   1. __Download processed data__
   
      The data file (`data.zip`) of these three datasets can be downloaded from this link. Uncompress this file to get a 'data' folder containing all the original data and processed data. 
      Replace the original 'data' folder by this new folder.
      
      For clarity, the content architecture of `data` directory is described as follows:
      
      ```text
       >  data
          ├── davis / kiba                          - DTA dataset directory.
          │   ├── ligands_can.txt                   - A txt file recording ligands information (Original)
          │   ├── proteins.txt                      - A txt file recording proteins information (Original)
          │   ├── Y                                 - A file recording binding affinity score (Original)
          │   ├── folds                         
          │   │   ├── test_fold_setting1.txt        - A txt file recording test set entry (Original)
          │   │   └── train_fold_setting1.txt       - A txt file recording training set entry (Original)
          │   ├── (davis/kiba)_dict.txt             - A txt file recording the corresponding Uniprot ID for every protein in datasets (processed)
          │   ├── contact_map
          │   │   └── (Uniprot ID).npy              - A npy file recording the corresponding contact map for every protein in datasets (processed)
          │   ├── PPI
          │   │   └── ppi_data.pkl                  - A pkl file recording the related PPI network data including adjacency matrix (dense),
          │   │                                       feature matrix and the protein index in PPI (processed)
          │   ├── train.csv                         - Training set data in CSV format (processed)
          │   ├── test.csv                          - Test set data in CSV format (processed)
          │   ├── mol_data.pkl                      - A pkl file recording drug graph data for all drugs in dataset (processed)
          │   └── pro_data.pkl                      - A pkl file recording protein graph data for all proteins in dataset (processed)
          └── Human                                 - CPI dataset directory.
              ├── Human.txt                         - A txt file recording the information of drugs and proteins that interact (Original)
              ├── contact_map
              │   └── (XXXXX).npy
              ├── PPI
              │   └── ppi_data.pkl                   
              ├── Human_dict.txt
              ├── train(fold).csv                   - 5-fold training set data in CSV format (processed)
              ├── test(fold).csv                    - 5-fold test set data in CSV format (processed)
              ├── mol_data.pkl
              └── pro_data.pkl
      ```
   2. __Customize your data__

      You might like to test the model on more DTA or CPI datasets. If this is the case, please add your data in the folder 'data' and process them to be suitable for our model. We provide a detailed processing script for converting original data to the input data that our model needed, i.e., `create_data.py`. The processing steps are as follows:
     
      1. Split the raw dataset into training and test sets, and convert them into CSV format respectively（i.e., `train.csv` and `test.csv`）.
         The content of the csv file can be organized as follows:
         ```text
                   compound_iso_smiles                                 target_sequence                                       affinity
         C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1          MAAVILESIFLKRSQQKKKTSPLNFKKRLFLLTVHKLSY                        5.568636236
                                                             YEYDFERGRRGSKKGSIDVEKITCVETVVPEKNPPPERQ
                                                             IPRRGEESSEMEQISIIERFPYPFQVVYDEGP
         ```
      2. Collect the Uniprot ID of all proteins in dataset from Uniprot DB(https://www.uniprot.org/) and record it as a txt file, such as `davis_dict.txt`:
         ```text
         >MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILM...	Q2M2I8
         >PFWKILNPLLERGTYYYFMGQQPGKVLGDQRRPSLPALHFIKGAGKKESSRHGGPHCNVFVEHEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLR...	P00519
         ```
      3. Download the corresponding protein structure file from the PDB（https://www.rcsb.org/） or Alphafold2(https://alphafold.com/) DB according to the Uniprot ID. Then you can get the contact map file by runing the following scripts:
         ```python
         python generate_contact_map.py --input_path '...data/your_dataset_name/your_pdb_dir/'  --output_path '...data/your_dataset_name/your_contact_map_dir/'  --chain_id 'A'
         ``` 
      4. Construct the graph data for drugs and proteins. Assume that you already have aboving files (1.2.3) in your `data/your_dataset_name/` folder, you can simply run following scripts:
         ```python
         python created_data.py --path '..data/'  --dataset 'your_dataset_name'  --output_path '..data/'
         ```
      5. Finally, Upload the Uniprot IDs of all proteins in your dataset to the String DB(https://string-db.org/) for PPI networks data, and the feature descriptor of protein in PPI network we used can be available from Interpro (https://www.ebi.ac.uk/interpro/).
      
   :bulb: Note that the above is just a description of the general steps, and you may need to make some modification to the original script for different datasets.
     
   :blush: Therefore，We have provided detailed comments on the functionality of each function in the script, hoping that it could be helpful for you.

* ### Training
  After processing the data, you can retrain the model from scratch with the following command:
  ```text
  
  python training_for_DTA.py --model TDNet --epochs 2000 --batch 512 --LR 0.0005 --log_interval 20 --device 0 --dataset davis --num_workers 6 
  or
  python training_for_CPI.py --model BUNet --epochs 2000 --batch 512 --LR 0.0005 --log_interval 20 --device 0 --dataset kiba --num_workers 6 
  ```
   Here is the detailed introduction of the optional parameters when running `training_for_DTA/CPI.py`:
     ```text
      --model: The model name, specifying the name of the model to be used.There are two optional backbones, BUNet and TDNet.
      --epochs: The number of epochs, specifying the number of iterations for training the model on the entire dataset.
      --batch: The batch size, specifying the number of samples in each training batch.
      --LR: The learning rate, controlling the rate at which model parameters are updated.
      --log_interval: The log interval, specifying the time interval for printing logs during training.
      --device: The device, specifying the GPU device number used for training.
      --dataset: The dataset name, specifying the dataset used for model training.
      --num_workers: This parameter is an optional value in the Dataloader, and when its value is greater than 0, it enables 
       multiprocessing for data processing.
   ```
   :bulb: We provided an additional training file (`training_for_CPI.py`) specifically for conducting five-fold cross-training on the Human dataset.
  
   :bulb: Additionally, due to the larger scale of proteins in the Human dataset, we have made modifications to the original architecture to alleviate the memory requirements. For detailed changes, please refer to the file  `HGCN_for_CPI.py`.

* ### Predicting
   If you don't want to re-train the model, we provide pre-trained model parameters as shown below. You can download these model parameter files and place them in the "results/dataset_name/" directory.
<a name="pretrained-models"></a>

   | Datasets | Pre-trained models          | Description |
   |:-----------:|:-----------------------------:|:--------------|
   | davis    | [HiSIF<sub>BUNet</sub>](https://) &nbsp; , &nbsp; [HiSIF<sub>TDNet</sub>](https://)       | SOTA general-purpose protein language model. |
   | kiba     | [HiSIF<sub>BUNet</sub>](https://) &nbsp; , &nbsp; [HiSIF<sub>TDNet</sub>](https://)          | End-to-end single sequence 3D structure predictor (Nov 2022 update). |
   | Human    | [HiSIF<sub>BUNet</sub>](https://) &nbsp; , &nbsp; [HiSIF<sub>TDNet</sub>](https://)          | End-to-end single sequence 3D structure predictor (Nov 2022 update). |
  
   After that, you can perform DTA predictions by running the following command:
   ```text 
   python test_for_DTA.py --model TDNet --dataset davis  or
   python test_for_CPI.py --model BUNet --dataset Human
   ```
   :bulb: Please note that before making predictions, in addition to placing the pre-trained model parameter files in the correct location, it is also necessary to place the required data files mentioned in the previous section in the appropriate location.
## Results
---
* ### Experimental results

  We have designed a protein semantic information fusion framework based on the concept of hierarchical graph to enhance the richness of protein representation. Meanwhile, we propose two different strategies for semantic information fusion (_Top-Down_ and _Bottom-Up_) and evaluate their performance on different datasets. The performance of two different strategies on different datasets is as follows:

  1. __Performance on the Davis dataset__
     <a name="Experimental results on davis dataset"></a>
  
      | Backbone | MSE          | CI |
      |:--------:|:---------:|:--------------:|
      | __TDNet__ (Top-Down)    |  0.193 | 0.907  |
      | __BUNet__ (Bottom-Up)   |  0.191 | 0.906 |

  2. __Performance on the KIBA dataset__
      <a name="Experimental results on kiba dataset"></a>
        
      | Backbone | MSE          | CI |
      |:--------:|:---------:|:--------------:|
      | __TDNet__ (Top-Down) |  0.120 | 0.904 |
      | __BUNet__ (Bottom-Up)|  0.121 | 0.904 |

  3. __Performance on the Human dataset__
      <a name="Experimental results on kiba dataset"></a>
     
      | Backbone | AUROC     | Precision |  Recall |
      |:--------:|:---------:|:--------------:|:-------:|
      | __TDNet__ (Top-Down) |  0.988 | 0.945 | 0.952 |
      | __BUNet__ (Bottom-Up)|  0.986 | 0.947 | 0.947|

   
* ### Reproduce the results with singal command
   To facilitate the reproducibility of our experimental results, we have provided a Docker Image-based solution that allows for reproducing our experimental results on multiple datasets with just a single command. You can easily experience this function with the following simple command：
  ```text
  sudo docker run --name hisif-con --gpus all --shm-size=2g -v /your/local/path/HiSIF-DTA/:/media/HiSIF-DTA -it hisif-image:v1

  # docker run ：Create and start a new container based on the specified image.
  # --name : It specifies the name ("hisif-con") for the container being created. You can use this name to reference and manage the container later.
  # --gpus : It enables GPU support within the container and assigns all available GPUs to it. This allows the container to utilize the GPU resources for computation.
  # -v : This is a parameter used to map local files to the container,and it is used in the following format: `-v /your/local/path/HiSIF-DTA:/mapped/container/path/HiSIF-DTA`
  # -it : These options are combined and used to allocate a pseudo-TTY and enable interactive mode, allowing the user to interact with the container's command-line interface.
  # hisif-image:v1 : It is a doker image, builded from Dockerfile. For detailed build instructions, please refer to the `Requirements` section.
  ```
  :bulb: Please note that the above one-click run is only applicable for the inference process and requires you to pre-place all the necessary processed data and pretrained models in the correct locations on your local machine. If you want to train the model in the created Docker container, please follow the instructions below:
   ```text
   1. sudo docker run --name hisif-con --gpus all --shm-size=16g -v /your/local/path/HiSIF-DTA/:/media/HiSIF-DTA -it hisif-image:v1 /bin/bash
   2. cd /media/HiSIF-DTA
   3. python training_for_DTA.py --dataset davis --model TDNet
   ```
   
## Contact
---
We welcome you to contact us (email: bixiangpeng@stu.ouc.edu.cn) for any questions and cooperations.
