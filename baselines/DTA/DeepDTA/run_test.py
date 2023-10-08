# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年09月09日
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from keras import backend as K
from datahelper import *
from arguments import argparser, logging
import keras
from keras.layers import Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Input, Embedding,Dense
from keras.models import Model
from emetrics import get_aupr, get_cindex, get_rm2
import pandas as pd

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'
device_id = "6"
config = tf.ConfigProto()
config.gpu_options.visible_device_list = device_id
sess = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(sess)

def build_combined_categorical(FLAGS, NUM_FILTERS, FILTER_LENGTH1, FILTER_LENGTH2):
    XDinput = Input(shape=(FLAGS.max_smi_len,), dtype='int32')  ### Buralar flagdan gelmeliii
    XTinput = Input(shape=(FLAGS.max_seq_len,), dtype='int32')

    ### SMI_EMB_DINMS  FLAGS GELMELII
    encode_smiles = Embedding(input_dim=FLAGS.charsmiset_size + 1, output_dim=128, input_length=FLAGS.max_smi_len)(
        XDinput)
    encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
                           strides=1)(encode_smiles)
    encode_smiles = GlobalMaxPooling1D()(encode_smiles)

    encode_protein = Embedding(input_dim=FLAGS.charseqset_size + 1, output_dim=128, input_length=FLAGS.max_seq_len)(
        XTinput)
    encode_protein = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH2, activation='relu', padding='valid',
                            strides=1)(encode_protein)
    encode_protein = GlobalMaxPooling1D()(encode_protein)

    encode_interaction = keras.layers.concatenate([encode_smiles, encode_protein],
                                                  axis=-1)  # merge.Add()([encode_smiles, encode_protein])

    # Fully connected
    FC1 = Dense(1024, activation='relu')(encode_interaction)
    FC2 = Dropout(0.1)(FC1)
    FC2 = Dense(1024, activation='relu')(FC2)
    FC2 = Dropout(0.1)(FC2)
    FC2 = Dense(512, activation='relu')(FC2)

    # And add a logistic regression on top
    predictions = Dense(1, kernel_initializer='normal')(FC2)  # OR no activation, rght now it's between 0-1, do I want this??? activation='sigmoid'

    interactionModel = Model(inputs=[XDinput, XTinput], outputs=[predictions])

    interactionModel.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=[cindex_score, mse_score])  # , metrics=['cindex_score']

    return interactionModel


def nfold_1_2_3_setting_sample(te_XD, te_XT, te_Y, measure, runmethod, FLAGS, dataset):
    test_set, outer_train_sets = dataset.read_sets(FLAGS)
    train_set = [item for sublist in outer_train_sets for item in sublist]
    all_predictions, all_losses = general_nfold_cv(te_XD, te_XT, te_Y, measure, runmethod, FLAGS,
                                                   train_set, test_set)


def general_nfold_cv(te_XD, te_XT, te_Y, prfmeasure, runmethod, FLAGS, labeled_sets,val_sets):

    valinds = val_sets
    te_label_row_inds, te_label_col_inds = np.where(
        np.isnan(te_Y) == False)  # basically finds the point address of affinity [x,y]
    terows = te_label_row_inds[valinds]
    tecols = te_label_col_inds[valinds]
    val_drugs, val_prots, val_Y = prepare_interaction_pairs(FLAGS, te_XD, te_XT, te_Y, terows, tecols)

    gridmodel = runmethod(FLAGS, FLAGS.num_windows, FLAGS.smi_window_lengths, FLAGS.seq_window_lengths)
    gridmodel.load_weights(filepath=FLAGS.output_dir + f'/{FLAGS.dataset}/best_model.ckpt')
    predicted_labels = gridmodel.predict([np.array(val_drugs), np.array(val_prots)])
    pre = [label[0] for label in predicted_labels]
    df = pd.read_csv(f'./data/{FLAGS.dataset}/test.csv')
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
    df_pre.to_csv(f'./results/{FLAGS.dataset}/predicted_value_of_DeepDTA_on_{FLAGS.dataset} .csv')
    loss, _, MSE = gridmodel.evaluate(([np.array(val_drugs), np.array(val_prots)]), np.array(val_Y), verbose=0)
    CI1 = prfmeasure(val_Y, predicted_labels)
    print(f'CI={CI1[0]},  MSE={loss}')

    return CI1, loss


def mse_score(y, f):
    mse = tf.reduce_mean(((y - f) ** 2), axis=0)
    return mse


def cindex_score(y_true, y_pred):
    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g / f)  # select


def prepare_interaction_pairs(FlAGS, XD, XT, Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity = []

    for pair_ind in range(len(rows)):

        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target = XT[cols[pair_ind]]
        targets.append(target)
        if FLAGS.dataset == 'davis':
            affinity.append(-np.log10(Y[rows[pair_ind], cols[pair_ind]] / 1e9))
        else:
            affinity.append(Y[rows[pair_ind], cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data, target_data, affinity


def experiment(FLAGS, perfmeasure, deepmethod):
    dataset = DataSet(setting_no=FLAGS.problem_type,
                      seqlen=FLAGS.max_seq_len,
                      smilen=FLAGS.max_smi_len,
                      need_shuffle=False)
    FLAGS.charseqset_size = dataset.charseqset_size
    FLAGS.charsmiset_size = dataset.charsmiset_size
    _, _, _, te_XD, te_XT, te_Y = dataset.parse_train_test_data(FLAGS)
    te_XD = np.asarray(te_XD)
    te_XT = np.asarray(te_XT)
    te_Y = np.asarray(te_Y)
    nfold_1_2_3_setting_sample( te_XD, te_XT, te_Y, perfmeasure, deepmethod, FLAGS, dataset)


def run_regression(FLAGS):
    perfmeasure = get_cindex
    deepmethod = build_combined_categorical

    experiment(FLAGS, perfmeasure, deepmethod)


if __name__ == "__main__":
    FLAGS = argparser()
    run_regression(FLAGS)
