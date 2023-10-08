# Baseline models
To demonstrate the superiority of the proposed model, we conduct experiments to compare our approach with the following state-of-the-art (SOTA) models:

**For DTA task:**
- **KronRLS**: Molecular descriptors of drug compounds and protein targets are encoded as kernels, and used for binding affinity prediction with a regularized least squares regression model KronRLS.
- **SimBoost**：Based on the similarities among drugs and among targets, SimBoost constructs features for drugs, targets, and drug–target pairs, and uses gradient boosting machines to predict the binding affinity.
- **DeepDTA**: DeepDTA utilizes the 1D-CNN to automatically learn representations from drug smiles and target protein sequences.
- **AttentionDTA**: AttentionDTA also employs the 1D-CNN to learn sequence representations from drugs and targets, and an attention mechanism is introduced to focus on the most relevant regions for interaction.
- **GraphDTA**: GraphDTA represents drugs as graphs, where nodes represent atoms and edges represent chemical bonds. By utilizing GCN in drug graph and 1D-CNN in target sequence, GraphDTA demonstrates food performance in DTA prediction.
- **MGraphDTA**: Similar to GraphDTA, MGraphDTA also models drugs as graph and target protein as sequence, and further propose a deep GNN and multiscale CNN to extract the representation of drugs and proteins, respectively.
- **DGraphDTA**: On the basic of GraphDTA, DGraphDTA further represents proteins as molecular graphs by calculating the contact distance among residues. It utilizes graph neural networks (GNNs) to learn structural information of drugs and proteins on these topology-aware graphs.

**For CPI task:**
- **KNN, L2, RF**: Three machine learning methods, K nearest neighbors (KNN), L2-logistic (L2) and random forest (RF) are adopted to evaluate the performance of our model.
- **DrugVQA (seq)**: DrugVQA represents proteins as 2D distance map from monomer structures (Image), and drugs as molecular linear notation (String), and predicts drug protein interaction following the Visual Question Answering mode. Note that DrugVQA (seq) is a simplified version of DrugVQA by replacing protein distance map with protein sequence, and a self-attentive BiLSTM is used to learn the sequence representation of drugs and proteins.
- **GraphDTA**: GraphDTA is originally designed for regression task, here, we tailor its last layer to binary classification task.
- **CPI-GNN**: CPI-GNN represents drugs as graphs and proteins as sequences, and utilizes GNN and 1D-CNN to learn molecular representations. It adopts an n-gram amino acid approach to map protein sequences into word embeddings in the protein branch.
- **TransformerCPI**: TransformerCPI regards protein sequences or drug SMILES strings as sentences and considers residues or atoms as words. It proposes a CPI prediction method based on the Transformer architecture, which encodes and decodes interactions between compounds and proteins, achieving accurate predictions.
  
<a name="Experimental results on davis dataset"></a>
  
  | DTA      | CPI          |
  |:--------:|:---------:|
  | KronRLS  |     k-NN   | 
  | SimBoost |     L2   | 
  | DeepDTA * |     RF   | 
  | AttentionDTA * |   GCN     |
  | GraphDTA * |    CPI-GNN  *  | 
  | MGraphDTA *|   DrugVQA (seq)  *   | 
  | DGraphDTA *|    TransfomerCPI *   | 
  |             |    GraphDTA *   | 

===
##
