import numpy as np
import tensorflow as tf
from tensorflow import keras
from queue import Queue
import csv
import math
import sys

strategy = 'guided'
def mutate_corpus(q, x):
    mut1 = 0.0001 * tf.ones_like(x)
    mut2 = 0.000001 * tf.ones_like(x)
    mut3 = 0.00000001 * tf.ones_like(x)
    q.put(x + mut1)
    q.put(x + mut2)
    q.put(x + mut3)


def tf_execute(input, opname):
    if opname == "softplus":
        return tf.raw_ops.Softplus(features = input)
    elif opname == 'leakyrelu':
        return tf.raw_ops.LeakyRelu(features = input)
    elif opname == 'selu':
        return tf.raw_ops.Selu(features = input)
    elif opname == 'elu':
        return tf.raw_ops.Elu(features = input)


def get_diff(input, csv_writer, opname, strategy, index):
    res = []
    res.append(index)
    x_16 = tf.cast(input, tf.bfloat16)
    x_32 = tf.cast(input, tf.float32)
    x_64 = tf.cast(input, tf.float64)
  

    out_16 = tf_execute(x_16, opname).numpy().astype(np.float64)
    out_32 = tf_execute(x_32, opname).numpy().astype(np.float64)
    out_64 = tf_execute(x_64, opname)

    if strategy == "max":
        diff1 = np.max(np.abs(out_16 - out_32))
        # diff2 = np.max(np.abs(out_32 - out_64))
    else:
        diff1 = np.mean(np.abs(out_16 - out_32))
        # diff2 = np.mean(np.abs(out_32 - out_64))

    res.append(diff1)
    # res.append(diff2)

    for n in out_64.numpy().ravel():
        if math.isnan(n):
            res.append("NAN")
            break

    csv_writer.writerow(res)
    # return max(res[1:3])
    return diff1

def createCorpus(size, shape, dtype):
    q = Queue()
    for i in range(size):
        # 使用PyTorch的随机函数生成数据
        x = tf.random.normal(shape=shape,dtype=dtype) * 10
        q.put(x)
    return q
def test_tf(opname):
    seed_num = 20000
    # corpus = createCorpus(seed_num, shape)
    rounds = 1
    out1 = open(file=f"data/{opname}_diffdata_tf.csv", mode="w", newline="")
    out2 = open(file=f"data/{opname}_res_tf.csv", mode="w", newline="")
    csv_writer1 = csv.writer(out1)
    csv_writer2 = csv.writer(out2)
    csv_writer1.writerow(["No.", "16-32", "isNaN"])
    csv_writer2.writerow(
        [
            "No.",
            "全局最大误差(同输入)",
            "全局累积误差",
        ]
    )
    h_error = 0
    for i in range(rounds):
        acc_error = 0
        info = []
        info.append(i)
        index = 0
        pre_max = 0
        max_index = 0
        corpus = createCorpus(seed_num, [1], tf.dtypes.bfloat16)
        for j in range(seed_num):
            input_array = corpus.get()
            max_diff = get_diff(input_array, csv_writer1, opname, "mean", index)
            acc_error += max_diff
            if max_diff > pre_max:
                if strategy == "guided" and pre_max != 0:
                    mutate_corpus(corpus, input_array)
                print('features',input_array)
                pre_max = max_diff
                max_index = index
            index += 1
        h_error = max(h_error, pre_max)
        # info.append(pre_max)
        info.append(h_error)
        info.append(acc_error)
        # info.append(max_index)

        csv_writer2.writerow(info)

    out1.close()
    out2.close()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置TensorFlow使用的GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs available: {gpus}")
        except RuntimeError as e:
            # 异常处理
            print(e)
    else:
        print("No GPUs found. Running on CPU.")
        exit()
    if sys.argv[1] == "softplus":
        test_tf('softplus')
    elif sys.argv[1] == 'leakyRelu':
        test_tf('leakyrelu')
    elif sys.argv[1] == 'selu':
        test_tf('selu')
    elif sys.argv[1] == 'elu':
        test_tf('elu')