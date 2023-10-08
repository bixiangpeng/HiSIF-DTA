import tensorflow as tf
import pandas as pd
import numpy as np
import DTA_model as model
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score  # R square
from lifelines.utils import concordance_index

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




def calculateMSE(X, Y):
    in_bracket = []
    for i in range(len(X)):
        num = Y[i] - X[i]
        num = pow(num, 2)
        in_bracket.append(num)
    all_sum = sum(in_bracket)
    MSE = all_sum / len(X)
    return MSE


def parser(record):
    read_features = {
        'drug': tf.FixedLenFeature([MAX_SMI_LEN], dtype=tf.int64),
        'protein': tf.FixedLenFeature([MAX_SEQ_LEN], dtype=tf.int64),
        'affinity': tf.FixedLenFeature([1], dtype=tf.float32)
    }

    read_data = tf.parse_single_example(
        serialized=record, features=read_features)
    # read_data = tf.parse_example(serialized=record, features=read_features)

    drug = tf.cast(read_data['drug'], tf.int32)
    protein = tf.cast(read_data['protein'], tf.int32)
    affinit_y = read_data['affinity']

    return drug, protein, affinit_y


def test( test_path):
    with tf.Graph().as_default() as g:
        dataset = tf.data.TFRecordDataset(test_path)
        dataset = dataset.map(parser)
        dataset = dataset.batch(batch_size=batch_size)
        iterator = dataset.make_initializable_iterator()

        drug_to_embeding, proteins_to_embeding, labels_batch = iterator.get_next()
        _, _, test_label = model.inference(drug_to_embeding,proteins_to_embeding,regularizer=None,keep_prob=1,trainlabel=0)
        mean_squared_eror = tf.losses.mean_squared_error(test_label, labels_batch)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            ckpt = tf.train.get_checkpoint_state("./results/" + dataname + "/model/" )
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                predictions_eval = []
                labels_eval = []
                test_MSElist = []
                try:
                    while True:
                        df, pf, p, l, MSE = sess.run([drug_to_embeding, proteins_to_embeding, test_label, labels_batch, mean_squared_eror])
                        predictions_eval.append(p)
                        labels_eval.append(l)
                        test_MSElist.append(MSE)
                except tf.errors.OutOfRangeError:
                    pass

                predictions_eval = np.concatenate(predictions_eval)
                labels_eval = np.concatenate(labels_eval)
                labels_eval.resize([labels_eval.shape[0], 1])
                pre = [label[0] for label in predictions_eval]
                df = pd.read_csv(f'./tfrecord/{dataname}/test.csv')
                test_smile = list(df['compound_iso_smiles'])
                test_seq = list(df['target_sequence'])
                test_label = list(df['affinity'])
                predicted_data = {
                    'smile': test_smile,
                    'sequence': test_seq,
                    'label': test_label,
                    'predicted value': pre
                }
                df_pre = pd.DataFrame(predicted_data)
                df_pre.to_csv(f'./results/{dataname}/predicted_value_of_AttentionDTA_on_{dataname} .csv')
                test_MSE = sum(test_MSElist) / len(test_MSElist)
                test_MAE = mean_absolute_error(labels_eval, predictions_eval)
                test_R2 = r2_score(labels_eval, predictions_eval)
                test_CI = concordance_index(labels_eval, predictions_eval)
                print("MSE:", test_MSE, "MAE:", test_MAE, "R2:", test_R2,"CI:",test_CI)
                return test_MSE,test_MAE,test_R2,test_CI



MAX_SEQ_LEN = 1200
MAX_SMI_LEN = 100

dataname = "davis"
# dataname = "kiba"
if dataname == "kiba":
    batch_size = 100
else:
    batch_size = 64

test_path = "./tfrecord/" + dataname + "/test.tfrecord"
mse,mae,r2, ci = test(test_path)

