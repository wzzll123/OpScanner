import numpy as np
import tensorflow as tf
from tensorflow import keras
from queue import Queue
import csv
import math
import sys
np.random.seed(42)

def tf_execute(input_kargs, opname):
    if opname == "softplus_backward":
        return tf.raw_ops.SoftplusGrad(**input_kargs)
    elif opname == 'leakyrelu_backward':
        return tf.raw_ops.LeakyReluGrad(**input_kargs)
    elif opname == 'selu_backward':
        return tf.raw_ops.SeluGrad(**input_kargs)
    elif opname == 'elu_backward':
        return tf.raw_ops.EluGrad(**input_kargs)


def get_diff(input_kargs, csv_writer, opname, strategy, index):
    res = []
    res.append(index)
    x_16 = {}
    x_32 = {}
    x_64 = {}

    for k,v in input_kargs.items():
        x_16[k] = tf.cast(v, dtype=tf.bfloat16)
        x_32[k] = tf.cast(v, dtype=tf.float32)
        x_64[k] = tf.cast(v, dtype=tf.float64)

    if opname == 'selu_backward' or opname == 'elu_backward':
        if opname == 'selu_backward':
            forward_ops = tf.raw_ops.Selu
        else:
            forward_ops = tf.raw_ops.Elu
        x_16['outputs'] = forward_ops(features=x_16['features'])
        del x_16['features']
        x_32['outputs'] = forward_ops(features=x_32['features'])
        del x_32['features']
        x_64['outputs'] = forward_ops(features=x_64['features'])
        del x_64['features']

    out_16 = tf_execute(x_16, opname).numpy().astype(np.float64)
    out_32 = tf_execute(x_32, opname).numpy().astype(np.float64)
    out_64 = tf_execute(x_64, opname)

    absolute_error = np.max(np.abs(out_16 - out_64))
    relative_error = np.max(np.abs((out_16 - out_64)/out_64))

    res.append(relative_error)
    # res.append(diff2)

    # for n in out_64.numpy().ravel():
    #     if math.isnan(n):
    #         res.append("NAN")
    #         break

    csv_writer.writerow(res)
    # return max(res[1:3])
    return absolute_error,relative_error


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
            "全局最大绝对误差",
            "全局最大相对误差",
        ]
    )
    global_abs_error_max = 0
    global_rel_error_max = 0
    for i in range(rounds):
        acc_error = 0
        info = []
        info.append(i)
        index = 0
        pre_abs_max = 0
        pre_rel_max = 0
        max_index = 0
        for j in range(seed_num):
            dtype = tf.dtypes.bfloat16
            shape = (1)
            features = generate_one_input(shape,dtype)
            gradients = generate_one_input(shape,dtype)
            # features = tf.random.normal(shape=[1],dtype=dtype) * 10
            # gradients = tf.random.normal(shape=[1],dtype=dtype) * 10
            input_kargs = {'features': features, 'gradients': gradients}
            abs_error, rel_error = get_diff(input_kargs, csv_writer1, opname, "mean", index)
            # acc_error += max_diff
            if abs_error > pre_abs_max:
                print('features',features)
                print('gradients',gradients)
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
def generate_one_input(shape,dtype):
    np_x = np.random.randn(shape)
    x = tf.convert_to_tensor(np_x,dtype=dtype)*10
    return x
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
        test_tf('softplus_backward')
    elif sys.argv[1] == 'leakyRelu':
        test_tf('leakyrelu_backward')
    elif sys.argv[1] == 'selu':
        test_tf('selu_backward')
    elif sys.argv[1] == 'elu':
        test_tf('elu_backward')