import numpy as np
import torch
from queue import Queue
import csv
import math
import sys

def get_diff(input1, grad_input1, csv_writer, opname, strategy, index):
    device = torch.device("cuda")
    res = []
    res.append(index)

    if opname == 'selu_backward':
        op = torch.nn.SELU().to(device)
    elif opname == 'softplus_backward':
        op = torch.nn.Softplus(threshold=70).to(device)
    elif opname == 'leakyrelu_backward':
        op = torch.nn.LeakyReLU(0.2).to(device)
    elif opname == 'elu_backward':
        op = torch.nn.ELU().to(device)
    
    input32 = input1.clone().detach().float().requires_grad_(True)
    grad_input32 = grad_input1.clone().detach().float().requires_grad_(True)
    # input32 = torch.tensor(input1, dtype=torch.float32, device=device, requires_grad=True)
    # grad_input32 = torch.tensor(grad_input1, dtype=torch.float32, device=device)
    out32 = op(input32)
    out32.backward(grad_input32)

    op_bfp16 = op.bfloat16()
    input16 = input32.detach().bfloat16().requires_grad_()
    grad_input16 = grad_input32.bfloat16()
    out16 = op_bfp16(input16)
    out16.backward(grad_input16)

    grad_output_16 = input16.grad.data.to(torch.float64).detach().cpu().numpy()
    grad_output_32 = input32.grad.data.to(torch.float64).detach().cpu().numpy()

    if strategy == "max":
        diff1 = np.max(np.abs(grad_output_16 - grad_output_32))
        # diff2 = np.max(np.abs(out_32 - out_64))
    else:
        diff1 = np.mean(np.abs(grad_output_16 - grad_output_32))
        # diff2 = np.mean(np.abs(out_32 - out_64))

    res.append(diff1)
    # res.append(diff2)

    # for n in out_64.numpy().ravel():
    #     if math.isnan(n):
    #         res.append("NAN")
    #         break

    csv_writer.writerow(res)
    # return max(res[1:3])
    return diff1


def test_torch(opname):
    device = torch.device("cuda")
    seed_num = 20000
    # corpus = createCorpus(seed_num, shape)
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
    for i in range(rounds):
        acc_error = 0
        info = []
        info.append(i)
        index = 0
        pre_max = 0
        max_index = 0
        for j in range(seed_num):
            shape = [1]
            dtype = torch.bfloat16
            input1 = (torch.randn(shape, dtype=dtype)*10).to(device)
            input2 = (torch.randn(shape, dtype=dtype)*10).to(device)
            max_diff = get_diff(input1, input2, csv_writer1, opname, "mean", index)
            acc_error += max_diff
            if max_diff > pre_max:
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
    if sys.argv[1] == "softplus":
        test_torch('softplus_backward')
    elif sys.argv[1] == 'leakyRelu':
        test_torch('leakyrelu_backward')
    elif sys.argv[1] == 'selu':
        test_torch('selu_backward')
    elif sys.argv[1] == 'elu':
        test_torch('elu_backward')