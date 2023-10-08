import torch
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
from model import *


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')


    with open('./data/Human/smile_dict.pkl','rb') as file:
        smile_dict = pkl.load(file)
    with open('./data/Human/seq_dict.pkl','rb') as file:
        seq_dict = pkl.load(file)

    results = []
    for fold in range(1,6):
        df_test = pd.read_csv(f'./data/Human/test{fold}.csv')
        dataset_test = []
        test_smile = []
        test_seq = []
        for index, row in df_test.iterrows():
            smile, sequence, interaction = row['0'], row['1'], row['2']
            compounds = torch.FloatTensor(smile_dict[smile][0]).to(device)
            adjacencies = torch.FloatTensor(smile_dict[smile][1]).to(device)
            proteins = torch.FloatTensor(seq_dict[sequence]).to(device)
            interactions = torch.LongTensor([interaction]).to(device)
            dataset_test.append((compounds, adjacencies, proteins, interactions))
            test_smile.append(smile)
            test_seq.append(sequence)
        encoder = Encoder(args.protein_dim, args.hid_dim, 3, args.kernel_size, args.dropout, device)
        decoder = Decoder(args.atom_dim, args.hid_dim, args.n_layers, args.n_heads, args.pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, args.dropout, device)
        model = Predictor(encoder, decoder, device)
        model.to(device)
        tester = Tester(model)

        file_model = f'results/model/model_fold{fold}.pt'
        model.load_state_dict(torch.load(file_model))
        G, P_value, P_label, AUC_test, precision_test, recall_test = tester.test(dataset_test)
        predicted_data = {
            'smile': test_smile,
            'sequence': test_seq,
            'label': G,
            'predicted value': P_value,
            'predicted label': P_label
        }
        df_pre = pd.DataFrame(predicted_data)
        df_pre.to_csv(f'./results/predicted_value_of_TransformerCPI_on_Human_test{fold} .csv')

        print(f'Fold-{fold}:  | auc: {AUC_test} | precision: {precision_test} | recall: {recall_test}')
        results.append([AUC_test, precision_test, recall_test])

    valid_results = np.array(results)
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]
    print("5-fold: auc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1], valid_results[0][2],valid_results[1][2]))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--protein_dim', type=int, default=100)
    parser.add_argument('--atom_dim', type=int, default=34)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--pf_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--decay_interval', type=int, default=5)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--iteration', type=int, default=40)
    parser.add_argument('--kernel_size', type=int, default=5)
    args = parser.parse_args()

    main(args)