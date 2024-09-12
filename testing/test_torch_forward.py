import torch
import numpy as np
from queue import Queue
import csv
import math
import sys
import test_torch_backward as ttb

torch.backends.cudnn.enabled = False
np.random.seed(42)
def mutate_corpus(q, x):
    device = torch.device("cuda")
    mut1 = 0.001 * torch.ones(x.shape, dtype=tested_dtype).to(device)
    mut2 = 0.0001 * torch.ones(x.shape, dtype=tested_dtype).to(device)
    mut3 = 0.00001 * torch.ones(x.shape, dtype=tested_dtype).to(device)
    q.put(x + mut1)
    q.put(x + mut2)
    q.put(x + mut3)

def createCorpus(size, shape, dtype):
    device = torch.device("cuda")
    q = Queue()
    for i in range(size):
        # 使用PyTorch的随机函数生成数据
        np_x = np.random.randn(shape)
        x = (torch.from_numpy(np_x).to(torch.bfloat16)*10).to(device)
        q.put(x)
    return q


def torch_execute(input_tensor, opname):
    device = torch.device("cuda")
    if opname == 'selu':
        op = torch.nn.SELU().to(device)
    elif opname == 'softplus':
        op = torch.nn.Softplus(threshold=70).to(device)
    elif opname == 'leakyrelu':
        op = torch.nn.LeakyReLU(0.2).to(device)
    elif opname == 'elu':
        op = torch.nn.ELU().to(device)
    elif opname == 'logcumsumexp':
        return torch.logcumsumexp(input_tensor,0)
    elif opname == 'logsoftmax':
        op = torch.nn.LogSoftmax(dim=0).to(device)
    elif opname == 'floordiv':
        return torch.floor_divide(input_tensor[0],input_tensor[1]).to(device)
    return op(input_tensor)


def get_diff(input_tensor, csv_writer, opname, strategy, index):
    res = []
    res.append(index)
    x_16 = input_tensor.to(torch.bfloat16)
    x_32 = input_tensor.to(torch.float32)
    x_64 = input_tensor.to(torch.float64)

    out_16 = torch_execute(x_16, opname).to(torch.float64).detach().cpu().numpy()
    # out_16_2 = torch_execute(x_16, opname).detach().cpu().numpy().astype(np.float64)
    out_32 = torch_execute(x_32, opname).to(torch.float64).detach().cpu().numpy()
    out_64 = torch_execute(x_64, opname).to(torch.float64).detach().cpu().numpy()

    absolute_error = np.max(np.abs(out_16 - out_64))
    relative_error = np.max(np.abs((out_16 - out_64)/out_64))

    # diff2 = np.max(np.abs(out_16 - out_32))

    res.append(relative_error)
    # res.append(diff1)

    # for n in out_64.ravel():
    #     if math.isnan(n):
    #         res.append("NAN")
    #         break

    csv_writer.writerow(res)
    # return max(res[1:3])
    return absolute_error,relative_error


def test_torch(opname):
    rounds = 10
    out1 = open(file=f"data/{opname}_diffdata_torch.csv", mode="w", newline="")
    out2 = open(file=f"data/{opname}_res_torch.csv", mode="w", newline="")
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
        # here using bfloat16 dtype to prevent the collection of errors in input conversion
        corpus = createCorpus(seed_num, tested_shape, tested_dtype)
        while not corpus.empty() and index < termination_num:
            input_tensor = corpus.get()
            abs_error, rel_error = get_diff(input_tensor, csv_writer1, opname, "mean", index)
            # acc_error += max_diff
            if abs_error > pre_abs_max:
                if strategy == "guided" and pre_abs_max != 0:
                    mutate_corpus(corpus, input_tensor)
                pre_abs_max = abs_error
                max_index = index
            if rel_error > pre_rel_max:
                pre_rel_max = rel_error
                if rel_error > global_rel_error_max:
                    print('triggered tensor: ', input_tensor)
            index += 1
        global_abs_error_max = max(global_abs_error_max, pre_abs_max)
        global_rel_error_max = max(global_rel_error_max, pre_rel_max)
        # "当前最大误差(同输入)",
        # info.append(pre_max)
        info.append(global_abs_error_max)
        # info.append(acc_error)
        # "引起最大误差的输入编号",
        info.append(global_rel_error_max)

        csv_writer2.writerow(info)

    out1.close()
    out2.close()


if __name__ == "__main__":
    # PyTorch通常会自动使用可用的GPU，但你可以通过以下方式进行确认或配置
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    strategy = 'random'
    tested_dtype = torch.bfloat16
    tested_shape = (1)
    seed_num = 2000
    termination_num = 2000
    if sys.argv[1] == "softplus":
        test_torch('softplus')
    elif sys.argv[1] == 'leakyRelu':
        test_torch('leakyrelu')
    elif sys.argv[1] == 'selu':
        test_torch('selu')
    elif sys.argv[1] == 'elu':
        test_torch('elu')
    elif sys.argv[1] == 'logcumsumexp':
        tested_shape = (5)
        test_torch('logcumsumexp')
    elif sys.argv[1] == 'logsoftmax':
        tested_shape = (5)
        test_torch('logsoftmax')
    elif sys.argv[1] == 'floordiv':
        tested_shape = (2)
        test_torch('floordiv')
