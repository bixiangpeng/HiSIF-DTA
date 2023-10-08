from utils import *
from dataPre import *
from sklearn import metrics
from tqdm import tqdm
import argparse
from model import *
import warnings
warnings.filterwarnings("ignore")

def test(args,model,test_loader,criterion, smiles_letters,sequence_letters):
    model.eval()
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    with torch.no_grad():
        for batch_idx, (smiles, seqs, properties) in enumerate(test_loader):
            smile_input, smile_lengths, y = make_variables(smiles, properties, smiles_letters)
            seq_input, seq_lengths = make_variables_seq(seqs, sequence_letters)
            y_pred = model(smile_input, seq_input)
            correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
            all_pred=np.concatenate((all_pred,y_pred.data.cpu().squeeze(1).numpy()),axis = 0)
            all_target = np.concatenate((all_target,y.data.cpu().numpy()),axis = 0)
            loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
            total_loss+=loss.data
            n_batches+=1

    testAcc = round(correct.numpy()/(n_batches*test_loader.batch_size),3)
    testRecall = round(metrics.recall_score(all_target,np.round(all_pred)),3)
    testPrecision = round(metrics.precision_score(all_target,np.round(all_pred)),3)
    testAuc = round(metrics.roc_auc_score(all_target, all_pred),3)
    # print("AUPR = ",metrics.average_precision_score(all_target, all_pred))
    testLoss = round(total_loss.item()/n_batches,5)

    return testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target


def main(args):
    smileLettersPath = './data/Human/voc/smile.voc'
    seqLettersPath = './data/Human/voc/sequence.voc'
    print('get letters....')
    smiles_letters = getLetters(smileLettersPath)
    sequence_letters = getLetters(seqLettersPath)

    n_chars_smi = len(smiles_letters)
    n_chars_seq = len(sequence_letters)

    device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'

    results = []
    for fold in range(1, 6):
        testFoldPath = f'./data/Human/test{fold}.csv'
        testDataSet = getDataSet(testFoldPath)
        test_dataset = ProDataset(dataSet=testDataSet)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
        model = DrugVQA(args, n_chars_smi, n_chars_seq).to(device)
        criterion = torch.nn.BCELoss()
        model_file_name = f'./results/model/fold_{fold}.model'
        model.load_state_dict(torch.load(model_file_name))
        testAcc, testRecall, testPrecision, testRoc, testLoss, G , P_value = test(args, model,test_loader, criterion, smiles_letters, sequence_letters)

        df = pd.read_csv(testFoldPath)
        test_smile = list(df['0'])
        test_seq = list(df['1'])
        P_label = np.round(P_value)
        G_list = G.tolist()
        P_value_list = P_value.tolist()
        P_label_list = P_label.tolist()
        predicted_data = {
            'smile': test_smile,
            'sequence': test_seq,
            'label': G_list,
            'predicted value': P_value_list,
            'predicted label': P_label_list
        }
        df_pre = pd.DataFrame(predicted_data)
        df_pre.to_csv(f'./results/predicted_value_of_DrugVQA_on_Human_test{fold} .csv')

        print(f'Fold-{fold}:  acc: {testAcc} | auc: {testRoc} | precision: {testPrecision} | recall: {testRecall}')
        results.append([testAcc, testRoc, testPrecision, testRecall])

    valid_results = np.array(results)
    valid_results = [np.mean(valid_results, axis=0), np.std(valid_results, axis=0)]
    print("5-fold : " "acc:{:.3f}±{:.4f} | auc:{:.3f}±{:.4f} | precision:{:.3f}±{:.4f} | recall:{:.3f}±{:.4f}".format(valid_results[0][0], valid_results[1][0], valid_results[0][1], valid_results[1][1],valid_results[0][2], valid_results[1][2], valid_results[0][3], valid_results[1][3]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lstm_hid_dim', type=int, default=64)
    parser.add_argument('--d_a', type=int, default=32)
    parser.add_argument('--r', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--in_channels', type=int, default=8)
    parser.add_argument('--cnn_channels', type=int, default=32)
    parser.add_argument('--cnn_layers', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=30)
    parser.add_argument('--dense_hid', type=int, default=64)
    parser.add_argument('--task_type', type=int, default=0)
    parser.add_argument('--doTest', type=bool, default=True)
    parser.add_argument('--use_regularizer', type=bool, default=False)
    parser.add_argument('--penal_coeff', type=float, default=0.03)
    parser.add_argument('--clip', type=bool, default=True)
    parser.add_argument('--doSave', type=bool, default=False)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=1)
    args = parser.parse_args()
    main(args)