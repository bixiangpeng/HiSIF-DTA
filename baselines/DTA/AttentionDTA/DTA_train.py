import tensorflow as tf
import DTA_model as model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dataname = "davis"
# dataname = "kiba"

LEARNING_RATE_BASE = 0.0001
EPOCH = 80
#
if dataname == "kiba":
    batch_size = 100
else:
    batch_size = 64

    
MAX_SEQ_LEN = 1200
MAX_SMI_LEN = 100

Train_path = "./tfrecord/" + dataname + "/train.tfrecord"
Test_path = "./tfrecord/" + dataname + "/test.tfrecord"
MODEL_SAVE_PATH = "./results/" + dataname + "/model/"
MODEL_NAME = "model.ckpt"


def parser(record):
    read_features = {
        'drug': tf.FixedLenFeature([MAX_SMI_LEN], dtype=tf.int64),
        'protein': tf.FixedLenFeature([MAX_SEQ_LEN], dtype=tf.int64),
        'affinity': tf.FixedLenFeature([1], dtype=tf.float32)
    }
    read_data = tf.parse_single_example(serialized=record, features=read_features)
    drug = tf.cast(read_data['drug'], tf.int32)
    protein = tf.cast(read_data['protein'], tf.int32)
    affinit_y = read_data['affinity']
    return drug, protein, affinit_y

def train(train_path, test_path):
    with tf.variable_scope("input"):
        train_dataset = tf.data.TFRecordDataset(train_path)
        train_dataset = train_dataset.map(parser)
        train_dataset = train_dataset.shuffle(500).batch(batch_size=batch_size)
        train_iterator = train_dataset.make_initializable_iterator()
        train_drug, train_proteins_to_embeding, train_labels_batch = train_iterator.get_next()

        test_dataset = tf.data.TFRecordDataset(test_path)
        test_dataset = test_dataset.map(parser)
        test_dataset = test_dataset.batch(batch_size=batch_size)
        test_iterator = test_dataset.make_initializable_iterator()
        test_drug, test_proteins_to_embeding, test_labels_batch = test_iterator.get_next()

    _, _, train_label = model.inference(train_drug, train_proteins_to_embeding, regularizer=None, keep_prob=0.9, trainlabel=1)
    _, _, test_label = model.inference(test_drug, test_proteins_to_embeding, regularizer=None, keep_prob=1.0, trainlabel=0)

    test_mean_squared_error = tf.losses.mean_squared_error(test_label, test_labels_batch)
    mean_squared_error = tf.losses.mean_squared_error(train_label, train_labels_batch)

    global_step = tf.Variable(0, trainable=False)


    learning_rate = LEARNING_RATE_BASE
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(mean_squared_error, global_step=global_step)
        with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')

    var_list = [var for var in tf.global_variables() if "moving" in var.name]
    var_list += tf.trainable_variables()
    saver = tf.train.Saver(var_list=var_list, max_to_keep=20)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess, open("./results/" + dataname + "/log.txt", "w") as f:
        print("Beginning training and validation")
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        maxloss = 100
        for epoch in range(EPOCH):
            sess.run(train_iterator.initializer)
            train_MSE_list = []
            try:
                while True:
                    _, train_MSE= sess.run([train_op, mean_squared_error])
                    train_MSE_list.append(train_MSE)
            except tf.errors.OutOfRangeError:
                pass
            sess.run(test_iterator.initializer)
            test_MSE_list = []
            try:
                while True:
                    test_MSE = sess.run(test_mean_squared_error)
                    test_MSE_list.append(test_MSE)
            except tf.errors.OutOfRangeError:
                pass

            test_MSE_avg = sum(test_MSE_list) / len(test_MSE_list)
            print( "%s-model-epoch:%d;test_MSE:%g;" % (dataname, epoch, test_MSE_avg))
            str = "%s-model-epoch:%d;test_MSE:%g;" % (dataname, epoch, test_MSE_avg)
            f.write(str + "\n")
            if test_MSE_avg < maxloss:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
                maxloss = test_MSE_avg
                print("Save model")


def main(argv=None):
    tf.reset_default_graph()
    if os.path.exists(MODEL_SAVE_PATH) is False:
        os.makedirs(MODEL_SAVE_PATH )
    train_path = Train_path
    test_path = Test_path
    train(train_path,test_path)


if __name__ == '__main__':
    tf.app.run()
