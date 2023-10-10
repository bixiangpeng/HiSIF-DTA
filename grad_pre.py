# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月10日
"""
from models.HGCN import *
from utils import *
import numpy as np
from torch_geometric.loader import DataLoader
import argparse
import torch

def grad_pre(device,loader,ppi_adj,ppi_features,proGraph,gradcam,test_seq):
    '''
    :param device: Device for loading models and data.
    :param loader: Dataloader used to batch the input data.
    :param ppi_adj: The adjacency matrix of a Protein-Protein Interaction (PPI) graph.
    :param ppi_features: The feature matrix of a Protein-Protein Interaction (PPI) graph.
    :param proGraph: Graph data encapsulated by all proteins in the current dataset.
    :param gradcam: Class object of GradAAM.
    :param test_seq: Protein sequences used for testing pocket positions.
    :return: Predicted pocket.
    '''
    for i,data in enumerate(loader):
        mol_data = data[0].to(device)
        pro_data = data[1].to(device)
        _, atom_att = gradcam(mol_data, pro_data, ppi_adj, ppi_features, proGraph)
        index = np.argwhere(atom_att > 0.92).reshape(-1)
        print(index)
        temp_list = []
        for j in index:
            temp_list.append(test_seq[i][j])
        pocket_string = ''.join(temp_list)
        print(pocket_string)


def main(args):
    dataset = 'kiba'
    model_dict_ = {'BUNet': BUNet, 'TDNet': TDNet} # Two model architectures we proposed.
    modeling = model_dict_[args.model]
    model_st = modeling.__name__

    path = f'results/{dataset}/pretrained_{model_st}.model'
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
    check_point = torch.load(path,map_location=device) # Load the model parameters trained on the KIBA dataset.

    model = modeling()
    model.load_state_dict(check_point) # Loading pre-trained model parameters into the current model.
    model =model.to(device)

    test_smile = ['O=c1oc2c(O)c(O)cc3c(=O)oc4c(O)c(O)cc1c4c23','CNC(=O)c1ccccc1Sc1ccc2c(C=Cc3ccccn3)n[nH]c2c1']
    test_seq =['MSGPVPSRARVYTDVNTHRPREYWDYESHVVEWGNQDDYQLVRKLGRGKYSEVFEAINITNNEKVVVKILKPVKKKKIKREIKILENLRGGPNIITLADIVKDPVSRTPALVFEHVNNTDFKQLYQTLTDYDIRFYMYEILKALDYCHSMGIMHRDVKPHNVMIDHEHRKLRLIDWGLAEFYHPGQEYNVRVASRYFKGPELLVDYQMYDYSLDMWSLGCMLASMIFRKEPFFHGHDNYDQLVRIAKVLGTEDLYDYIDKYNIELDPRFNDILGRHSRKRWERFVHSENQHLVSPEALDFLDKLLRYDHQSRLTAREAMEHPYFYTVVKDQARMGSSSMPGGSTPVSSANMMSGISSVPTPSPLGPLAGSPVIAAANPLGMPVPAAAGAQQ','MQSKVLLAVALWLCVETRAASVGLPSVSLDLPRLSIQKDILTIKANTTLQITCRGQRDLDWLWPNNQSGSEQRVEVTECSDGLFCKTLTIPKVIGNDTGAYKCFYRETDLASVIYVYVQDYRSPFIASVSDQHGVVYITENKNKTVVIPCLGSISNLNVSLCARYPEKRFVPDGNRISWDSKKGFTIPSYMISYAGMVFCEAKINDESYQSIMYIVVVVGYRIYDVVLSPSHGIELSVGEKLVLNCTARTELNVGIDFNWEYPSSKHQHKKLVNRDLKTQSGSEMKKFLSTLTIDGVTRSDQGLYTCAASSGLMTKKNSTFVRVHEKPFVAFGSGMESLVEATVGERVRIPAKYLGYPPPEIKWYKNGIPLESNHTIKAGHVLTIMEVSERDTGNYTVILTNPISKEKQSHVVSLVVYVPPQIGEKSLISPVDSYQYGTTQTLTCTVYAIPPPHHIHWYWQLEEECANEPSQAVSVTNPYPCEEWRSVEDFQGGNKIEVNKNQFALIEGKNKTVSTLVIQAANVSALYKCEAVNKVGRGERVISFHVTRGPEITLQPDMQPTEQESVSLWCTADRSTFENLTWYKLGPQPLPIHVGELPTPVCKNLDTLWKLNATMFSNSTNDILIMELKNASLQDQGDYVCLAQDRKTKKRHCVVRQLTVLERVAPTITGNLENQTTSIGESIEVSCTASGNPPPQIMWFKDNETLVEDSGIVLKDGNRNLTIRRVRKEDEGLYTCQACSVLGCAKVEAFFIIEGAQEKTNLEIIILVGTAVIAMFFWLLLVIILRTVKRANGGELKTGYLSIVMDPDELPLDEHCERLPYDASKWEFPRDRLKLGKPLGRGAFGQVIEADAFGIDKTATCRTVAVKMLKEGATHSEHRALMSELKILIHIGHHLNVVNLLGACTKPGGPLMVIVEFCKFGNLSTYLRSKRNEFVPYKTKGARFRQGKDYVGAIPVDLKRRLDSITSSQSSASSGFVEEKSLSDVEEEEAPEDLYKDFLTLEHLICYSFQVAKGMEFLASRKCIHRDLAARNILLSEKNVVKICDFGLARDIYKDPDYVRKGDARLPLKWMAPETIFDRVYTIQSDVWSFGVLLWEIFSLGASPYPGVKIDEEFCRRLKEGTRMRAPDYTTPEMYQTMLDCWHGEPSQRPTFSELVEHLGNLLQANAQQDGKDYIVLPISETLSMEEDSGLSLPTSPVSCMEEEEVCDPKFHYDNTAGISQYLQNSKRKSRPVSVKTFEDIPLEEPEVKVIPDDNQTDSGMVLASEELKTLEDRTKLSPSFGGMVPSKSRESVASEGSNQTSGYQSGYHSDDTDTTVYSSEEAELLKLIEIGVQTGSTAQILQPDSGTTLSSPPV']
    test_label =[13.49794001,14.50011693]

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file) # Reading drug graph data from the serialized file.
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2) # Reading protein graph data from the serialized file.
    with open(f'data/{dataset}/PPI/ppi_data.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3) # Reading PPI graph data from the serialized file.
    # 'ppi_index' is a dictionary that records the order of protein nodes in PPI, where the keys are protein sequences and the values are node indices

    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device) # Tensorization and sparsification of the adjacency matrix of the PPI graph.
    ppi_features = torch.Tensor(ppi_features).to(device) # Tensorization of the feature matrix of the PPI graph.
    pro_graph = proGraph(pro_data, ppi_index, device) # A function that encapsulates all protein graphs in the dataset into a single graph.

    test_dataset = DTADataset(test_smile, test_seq, test_label, mol_data=mol_data, ppi_index=ppi_index)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    gradcam = GradAAM(model, module=model.proGconv2) # A utility class for calculating weighted gradients of specified hidden layer with respect to prediction values.
    grad_pre(device, test_loader, ppi_adj, ppi_features, pro_graph,gradcam,test_seq) # A function for computing residues that contribute significantly to prediction values.



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'BUNet' ,help = '0: BUNet 1:TDNet')
    parser.add_argument('--batch', type = int, default = 1)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--num_workers', type= int, default = 6)
    args = parser.parse_args()
    main(args)
