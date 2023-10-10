## Baseline models
To demonstrate the superiority of the proposed model, we conduct experiments to compare our approach with the following state-of-the-art (SOTA) models:

**For DTA task:**
- **KronRLS**: Molecular descriptors of drug compounds and protein targets are encoded as kernels, and used for binding affinity prediction with a regularized least squares regression model KronRLS.
- **SimBoost**：Based on the similarities among drugs and among targets, SimBoost constructs features for drugs, targets, and drug–target pairs, and uses gradient boosting machines to predict the binding affinity.
- **DeepDTA\***: DeepDTA utilizes the 1D-CNN to automatically learn representations from drug smiles and target protein sequences.
- **AttentionDTA\***: AttentionDTA also employs the 1D-CNN to learn sequence representations from drugs and targets, and an attention mechanism is introduced to focus on the most relevant regions for interaction.
- **GraphDTA\***: GraphDTA represents drugs as graphs, where nodes represent atoms and edges represent chemical bonds. By utilizing GCN in drug graph and 1D-CNN in target sequence, GraphDTA demonstrates food performance in DTA prediction.
- **MGraphDTA\***: Similar to GraphDTA, MGraphDTA also models drugs as graph and target protein as sequence, and further propose a deep GNN and multiscale CNN to extract the representation of drugs and proteins, respectively.
- **DGraphDTA\***: On the basic of GraphDTA, DGraphDTA further represents proteins as molecular graphs by calculating the contact distance among residues. It utilizes graph neural networks (GNNs) to learn structural information of drugs and proteins on these topology-aware graphs.

**For CPI task:**
- **KNN, L2, RF**: Three machine learning methods, K nearest neighbors (KNN), L2-logistic (L2) and random forest (RF) are adopted to evaluate the performance of our model.
- **DrugVQA (seq)\***: DrugVQA represents proteins as 2D distance map from monomer structures (Image), and drugs as molecular linear notation (String), and predicts drug protein interaction following the Visual Question Answering mode. Note that DrugVQA (seq) is a simplified version of DrugVQA by replacing protein distance map with protein sequence, and a self-attentive BiLSTM is used to learn the sequence representation of drugs and proteins.
- **GraphDTA\***: GraphDTA is originally designed for regression task, here, we tailor its last layer to binary classification task.
- **CPI-GNN\***: CPI-GNN represents drugs as graphs and proteins as sequences, and utilizes GNN and 1D-CNN to learn molecular representations. It adopts an n-gram amino acid approach to map protein sequences into word embeddings in the protein branch.
- **TransformerCPI\***: TransformerCPI regards protein sequences or drug SMILES strings as sentences and considers residues or atoms as words. It proposes a CPI prediction method based on the Transformer architecture, which encodes and decodes interactions between compounds and proteins, achieving accurate predictions.

## Re-training
To avoid any potential inconsistence when comparing these methods, we re-trained all baseline models with strictly the identical experimental setting to our model. Besides, we adopted the optimal parameters as reported in their papers. In this directory, we have provided the relevant code used for retraining these models, as well as the saved model parameters. Please refer to the `CPI` and `DTA` directories for more information.

💡 Due to restrictions on upload file size, some of the data could not be uploaded. The complete `baselines` file can be obtained via the following link:.........

💡 In addition, due to the unavailability of the relevant training codes for some traditional baseline methods, such as KronLS and SimBoost, we did not re-train them and kept the original results reported in their source papers.
## Results
The model performance of these models can also be found in `experimental_results.ipynb` in the upper-level directory.

**DTA:**
<table border= '1'  height='300px'>
    <tr>
        <th rowspan='2'>Method</th> <th colspan='2'>Davis</th> <th colspan='2'>KIBA</th>
    </tr>
    <tr>
        <th>MSE</th><th>CI</th><th>MSE</th><th>CI</th>
    </tr>
    <tr>
        <th>KronRLS</th><td>0.379</td><td>0.871</td><td>0.411</td><td>0.782</td>
    </tr>
    <tr>
        <th>SimBoost</th><td>0.282</td><td>0.872</td><td>0.222</td><td>0.836</td>
    </tr>
    <tr>
        <th>DeepDTA*</th><td>0.253</td><td>0.879</td><td>0.187</td><td>0.854</td>
    </tr>
    <tr>
        <th>AttentionDTA*</th><td>0.244</td><td>0.885</td><td>0.175</td><td>0.867</td>
    </tr>
    <tr>
        <th>GraphDTA*</th><td>0.230</td><td>0.885</td><td>0.146</td><td>0.885</td>
    </tr>
    <tr>
        <th>MGraphDTA*</th><td>0.207</td><td>0.895</td><td>0.128</td><td>0.902</td>
    </tr>
    <tr>
        <th>DGraphDTA*</th><td>0.202</td><td>0.905</td><td>0.127</td><td>0.902</td>
    </tr>
    <tr>
        <th>Ours(Top-Down)</th><th>0.193</th><td>0.907</td><th>0.120</th><td>0.904</td>
    </tr>
    <tr>
        <th>Ours(Bottom-Up)</th><td>0.191</td><th>0.906</th><td>0.121</td><th>0.904</th>
    </tr>
</table>

**CPI:**
<table border= '1'  height='300px'>
    <tr>
        <th rowspan='2'>Method</th> <th colspan='3'>Human</th>
    </tr>
    <tr>
        <th>AUROC</th><th>Precision</th><th>Recall</th>
    </tr>
    <tr>
        <th>K-NN</th><td>0.860</td><td>0.927</td><td>0.798</td>
    </tr>
    <tr>
        <th>L2</th><td>0.911</td><td>0.913</td><td>0.867</td>
    </tr>
    <tr>
        <th>RF</th><td>0.940</td><td>0.897</td><td>0.861</td>
    </tr>
    <tr>
        <th>GCN</th><td>0.956</td><td>0.862</td><td>0.928</td>
    </tr>
    <tr>
        <th>CPI-GNN*</th><td>0.965</td><td>0.919</td><td>0.912</td>
    </tr>
    <tr>
        <th>DrugVQA (seq)*</th><td>0.966</td><td>0.921</td><td>0.914</td>
    </tr>
    <tr>
        <th>TransformerCPI*</th><td>0.974</td><td>0.914</td><td>0.923</td>
    </tr>
    <tr>
        <th>GraphDTA*</th><td>0.975</td><td>0.930</td><td>0.917</td>
    </tr>
    <tr>
        <th>Ours(Top-Down)</th><th>0.988</th><td>0.945</td><th>0.952</th>
    </tr>
    <tr>
        <th>Ours(Bottom-Up)</th><td>0.986</td><th>0.947</th><td>0.947</td>
    </tr>
</table>

💡 Methods with an asterisk (*) indicate that they have been re-trained.


## Forked repos

Below are the repository links for the relevant baseline models. Our training code is based on these source codes with slight modifications.

**DTA:**
- **DeepDTA** : [Repo Link ](https://github.com/hkmztrk/DeepDTA)
- **AttentionDTA** : [Repo Link ](https://github.com/zhaoqichang/AttentionDTA_BIBM)
- **GraphDTA** : [Repo Link ](https://github.com/thinng/GraphDTA)
- **MGraphDTA** : [Repo Link ](https://github.com/guaguabujianle/MGraphDTA)
- **DGraphDTA** : [Repo Link ](https://github.com/595693085/DGraphDTA)

**CPI:**
- **DrugVQA (seq)** : [Repo Link ](https://github.com/prokia/drugVQA)
- **GraphDTA** : [Repo Link ](https://github.com/thinng/GraphDTA)
- **CPI-GNN** : [Repo Link ](https://github.com/masashitsubaki/CPI_prediction)
- **TransformerCPI** : [Repo Link ](https://github.com/lifanchen-simm/transformerCPI)

Please click on the links to access the corresponding source codes and detailed information.
