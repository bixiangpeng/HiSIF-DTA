# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年04月10日
"""
from models.HGCN import *
from utils import *
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from lifelines.utils import concordance_index
import argparse

import torch






def grad_pre(device,loader,ppi_adj,ppi_features,proGraph,gradcam,test_seq):
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
    dataset = args.dataset
    modeling = [BUNet, TDNet][args.model]
    model_st = modeling.__name__

    path = f'/home/sgzhang/xiangpeng/HGraphDTA/results/{dataset}/{model_st}.model'

    check_point = torch.load(path)

    model = modeling()
    model.load_state_dict(check_point)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
    model =model.to(device)

    test_smile = ['O=c1oc2c(O)c(O)cc3c(=O)oc4c(O)c(O)cc1c4c23','CNC(=O)c1ccccc1Sc1ccc2c(C=Cc3ccccn3)n[nH]c2c1']
    test_seq =['MSGPVPSRARVYTDVNTHRPREYWDYESHVVEWGNQDDYQLVRKLGRGKYSEVFEAINITNNEKVVVKILKPVKKKKIKREIKILENLRGGPNIITLADIVKDPVSRTPALVFEHVNNTDFKQLYQTLTDYDIRFYMYEILKALDYCHSMGIMHRDVKPHNVMIDHEHRKLRLIDWGLAEFYHPGQEYNVRVASRYFKGPELLVDYQMYDYSLDMWSLGCMLASMIFRKEPFFHGHDNYDQLVRIAKVLGTEDLYDYIDKYNIELDPRFNDILGRHSRKRWERFVHSENQHLVSPEALDFLDKLLRYDHQSRLTAREAMEHPYFYTVVKDQARMGSSSMPGGSTPVSSANMMSGISSVPTPSPLGPLAGSPVIAAANPLGMPVPAAAGAQQ','MQSKVLLAVALWLCVETRAASVGLPSVSLDLPRLSIQKDILTIKANTTLQITCRGQRDLDWLWPNNQSGSEQRVEVTECSDGLFCKTLTIPKVIGNDTGAYKCFYRETDLASVIYVYVQDYRSPFIASVSDQHGVVYITENKNKTVVIPCLGSISNLNVSLCARYPEKRFVPDGNRISWDSKKGFTIPSYMISYAGMVFCEAKINDESYQSIMYIVVVVGYRIYDVVLSPSHGIELSVGEKLVLNCTARTELNVGIDFNWEYPSSKHQHKKLVNRDLKTQSGSEMKKFLSTLTIDGVTRSDQGLYTCAASSGLMTKKNSTFVRVHEKPFVAFGSGMESLVEATVGERVRIPAKYLGYPPPEIKWYKNGIPLESNHTIKAGHVLTIMEVSERDTGNYTVILTNPISKEKQSHVVSLVVYVPPQIGEKSLISPVDSYQYGTTQTLTCTVYAIPPPHHIHWYWQLEEECANEPSQAVSVTNPYPCEEWRSVEDFQGGNKIEVNKNQFALIEGKNKTVSTLVIQAANVSALYKCEAVNKVGRGERVISFHVTRGPEITLQPDMQPTEQESVSLWCTADRSTFENLTWYKLGPQPLPIHVGELPTPVCKNLDTLWKLNATMFSNSTNDILIMELKNASLQDQGDYVCLAQDRKTKKRHCVVRQLTVLERVAPTITGNLENQTTSIGESIEVSCTASGNPPPQIMWFKDNETLVEDSGIVLKDGNRNLTIRRVRKEDEGLYTCQACSVLGCAKVEAFFIIEGAQEKTNLEIIILVGTAVIAMFFWLLLVIILRTVKRANGGELKTGYLSIVMDPDELPLDEHCERLPYDASKWEFPRDRLKLGKPLGRGAFGQVIEADAFGIDKTATCRTVAVKMLKEGATHSEHRALMSELKILIHIGHHLNVVNLLGACTKPGGPLMVIVEFCKFGNLSTYLRSKRNEFVPYKTKGARFRQGKDYVGAIPVDLKRRLDSITSSQSSASSGFVEEKSLSDVEEEEAPEDLYKDFLTLEHLICYSFQVAKGMEFLASRKCIHRDLAARNILLSEKNVVKICDFGLARDIYKDPDYVRKGDARLPLKWMAPETIFDRVYTIQSDVWSFGVLLWEIFSLGASPYPGVKIDEEFCRRLKEGTRMRAPDYTTPEMYQTMLDCWHGEPSQRPTFSELVEHLGNLLQANAQQDGKDYIVLPISETLSMEEDSGLSLPTSPVSCMEEEEVCDPKFHYDNTAGISQYLQNSKRKSRPVSVKTFEDIPLEEPEVKVIPDDNQTDSGMVLASEELKTLEDRTKLSPSFGGMVPSKSRESVASEGSNQTSGYQSGYHSDDTDTTVYSSEEAELLKLIEIGVQTGSTAQILQPDSGTTLSSPPV']
    test_label =[13.49794001,14.50011693]

    with open(f"data/{dataset}/mol_data.pkl", 'rb') as file:
        mol_data = pickle.load(file)
    with open(f'data/{dataset}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)
    with open(f'data/{dataset}/PPI/ppi_data_only_domain.pkl', 'rb') as file3:
        ppi_adj, ppi_features, ppi_index = pickle.load(file3)
    ppi_adj = torch.LongTensor(np.argwhere(ppi_adj == 1).transpose(1, 0)).to(device)
    ppi_features = torch.Tensor(ppi_features).to(device)

    pro_graph = proGraph(pro_data, ppi_index, device)
    test_dataset = DtaDataset(test_smile, test_seq, test_label, mol_data=mol_data, ppi_index=ppi_index)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate)#,num_workers=args.num_workers)

    gradcam = GradAAM(model, module=model.proGconv2)
    grad_pre(device, test_loader, ppi_adj, ppi_features, pro_graph,gradcam,test_seq)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = int, default = 0 ,help = '0: BUNet 1:TDNet')
    parser.add_argument('--batch', type = int, default = 1)
    parser.add_argument('--device', type = int, default = 0)
    parser.add_argument('--dataset', type = str, default = 'kiba')
    parser.add_argument('--num_workers', type= int, default = 6)
    args = parser.parse_args()
    main(args)
