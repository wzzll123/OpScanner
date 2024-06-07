import torch
import numpy as np
from queue import Queue
import csv
import math
import sys
import test_torch_backward as ttb

strategy = 'guided'

def mutate_corpus(q, x):
    device = torch.device("cuda")
    mut1 = 0.0001 * torch.ones(x.shape).to(device)
    mut2 = 0.000001 * torch.ones(x.shape).to(device)
    mut3 = 0.00000001 * torch.ones(x.shape).to(device)
    q.put(x + mut1)
    q.put(x + mut2)
    q.put(x + mut3)

def createCorpus(size, shape, dtype):
    device = torch.device("cuda")
    q = Queue()
    for i in range(size):
        # 使用PyTorch的随机函数生成数据
        x = (torch.randn(shape, dtype=dtype)*10).to(device)
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

    if strategy == "max":
        diff1 = np.max(np.abs(out_16 - out_32))
        # diff2 = np.max(np.abs(out_16 - out_16_2))
    else:
        diff1 = np.mean(np.abs(out_16 - out_32))
        # diff2 = np.mean(np.abs(out_16 - out_16_2))

    res.append(diff1)
    # res.append(diff2)

    for n in out_64.ravel():
        if math.isnan(n):
            res.append("NAN")
            break

    csv_writer.writerow(res)
    # return max(res[1:3])
    return diff1


def test_torch(opname):
    seed_num = 20000
    rounds = 1
    out1 = open(file=f"data/{opname}_diffdata_torch.csv", mode="w", newline="")
    out2 = open(file=f"data/{opname}_res_torch.csv", mode="w", newline="")
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
    acc_error = 0
    for i in range(rounds):
        info = []
        info.append(i)
        index = 0
        pre_max = 0
        max_index = 0
        corpus = createCorpus(seed_num, [1], torch.bfloat16)
        while not corpus.empty():
            input_tensor = corpus.get()
            max_diff = get_diff(input_tensor, csv_writer1, opname, "mean", index)
            acc_error += max_diff
            if max_diff > pre_max:
                if strategy == "guided" and pre_max != 0:
                    mutate_corpus(corpus, input_tensor)
                pre_max = max_diff
                max_index = index
            index += 1
        h_error = max(h_error, pre_max)
        # "当前最大误差(同输入)",
        # info.append(pre_max)
        info.append(h_error)
        info.append(acc_error)
        # "引起最大误差的输入编号",
        # info.append(max_index)

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

    if sys.argv[1] == "softplus":
        test_torch('softplus')
    elif sys.argv[1] == 'leakyRelu':
        test_torch('leakyrelu')
    elif sys.argv[1] == 'selu':
        test_torch('selu')
    elif sys.argv[1] == 'elu':
        test_torch('elu')
