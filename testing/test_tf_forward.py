import numpy as np
import tensorflow as tf
from tensorflow import keras
from queue import Queue
import csv
import math
import sys
import os
# os.environ['TF_USE_CUDNN'] = '0'
np.random.seed(42)
def mutate_corpus(q, x):
    mut1 = 0.001 * tf.ones_like(x)
    mut2 = 0.0001 * tf.ones_like(x)
    mut3 = 0.00001 * tf.ones_like(x)
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
    elif opname == 'logcumsumexp':
        return tf.raw_ops.CumulativeLogsumexp(x=input,axis=0)
    elif opname == 'logsoftmax':
        return tf.raw_ops.LogSoftmax(logits=input)
    elif opname == 'floordiv':
        return tf.raw_ops.FloorDiv(x=input[0],y=input[1])

def get_diff(input, csv_writer, opname, strategy, index):
    res = []
    res.append(index)
    x_16 = tf.cast(input, tf.bfloat16)
    x_32 = tf.cast(input, tf.float32)
    x_64 = tf.cast(input, tf.float64)
  

    out_16 = tf_execute(x_16, opname).numpy().astype(np.float64)
    out_32 = tf_execute(x_32, opname).numpy().astype(np.float64)
    out_64 = tf_execute(x_64, opname)

    # if strategy == "mean":
    #     diff1 = np.mean(np.abs(out_16 - out_64))
    #     # diff2 = np.mean(np.abs(out_32 - out_64))
    # else:
    absolute_error = np.max(np.abs(out_16 - out_64))
    relative_error = np.max(np.abs((out_16 - out_64)/out_64))
        # diff2 = np.max(np.abs(out_32 - out_64))


    res.append(relative_error)
    # res.append(diff2)

    # for n in out_64.numpy().ravel():
    #     if math.isnan(n):
    #         res.append("NAN")
    #         break

    csv_writer.writerow(res)
    # return max(res[1:3])
    return absolute_error,relative_error

def createCorpus(size, shape, dtype):
    q = Queue()
    for i in range(size):
        # 使用PyTorch的随机函数生成数据
        np_x = np.random.randn(shape)
        x = tf.convert_to_tensor(np_x,dtype=tf.bfloat16)*10
        # x = tf.random.normal(shape=shape,dtype=dtype) * 10
        q.put(x)
    return q
def test_tf(opname):
    # corpus = createCorpus(seed_num, shape)
    rounds = 10
    out1 = open(file=f"data/{opname}_diffdata_tf.csv", mode="w", newline="")
    out2 = open(file=f"data/{opname}_res_tf.csv", mode="w", newline="")
    csv_writer1 = csv.writer(out1)
    csv_writer2 = csv.writer(out2)
    csv_writer1.writerow(["No.", "16-32", "isNaN"])
    csv_writer2.writerow(
        [
            "No.",
            "全局最大绝对误差",
            "全局最大相对误差",
        ]
    )
    global_abs_error_max = 0
    global_rel_error_max = 0
    acc_error = 0
    for i in range(rounds):
        
        info = []
        info.append(i)
        index = 0
        pre_abs_max = 0
        pre_rel_max = 0
        max_index = 0
        corpus = createCorpus(seed_num, tested_shape, tf.dtypes.bfloat16)
        while not corpus.empty() and index < termination_num:
            input_array = corpus.get()
            abs_error, rel_error = get_diff(input_array, csv_writer1, opname, "mean", index)
            # acc_error += max_diff
            if abs_error > pre_abs_max:
                if strategy == "guided" and pre_abs_max != 0:
                    mutate_corpus(corpus, input_array)
                pre_abs_max = abs_error
                max_index = index
            if rel_error > pre_rel_max:
                pre_rel_max = rel_error
            index += 1
        global_abs_error_max = max(global_abs_error_max, pre_abs_max)
        global_rel_error_max = max(global_rel_error_max, pre_rel_max)
        # info.append(pre_max)
        info.append(global_abs_error_max)
        info.append(global_rel_error_max)
        # info.append(max_index)

        csv_writer2.writerow(info)

    out1.close()
    out2.close()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    # if gpus:
    #     try:
    #         # 设置TensorFlow使用的GPU
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         print(f"GPUs available: {gpus}")
    #     except RuntimeError as e:
    #         # 异常处理
    #         print(e)
    # else:
    #     print("No GPUs found. Running on CPU.")
    #     exit()
    strategy = 'random'
    seed_num = 2000
    termination_num = 2000
    tested_shape = (1)
    if sys.argv[1] == "softplus":
        test_tf('softplus')
    elif sys.argv[1] == 'leakyRelu':
        test_tf('leakyrelu')
    elif sys.argv[1] == 'selu':
        test_tf('selu')
    elif sys.argv[1] == 'elu':
        test_tf('elu')
    elif sys.argv[1] == 'logcumsumexp':
        tested_shape = (5)
        test_tf('logcumsumexp')
    elif sys.argv[1] == 'logsoftmax':
        tested_shape = (5)
        test_tf('logsoftmax')
    elif sys.argv[1] == 'floordiv':
        tested_shape = (2)
        test_tf('floordiv')