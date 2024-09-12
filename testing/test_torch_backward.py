import numpy as np
import torch
from queue import Queue
import csv
import math
import sys
np.random.seed(42)
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
    input64 = input1.clone().detach().to(torch.float64).requires_grad_(True)
    grad_input64 = grad_input1.clone().detach().to(torch.float64).requires_grad_(True)
    out64 = op(input64)
    out64.backward(grad_input64)
    
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
    grad_output_64 = input64.grad.data.to(torch.float64).detach().cpu().numpy()

    absolute_error = np.max(np.abs(grad_output_16 - grad_output_64))
    relative_error = np.max(np.abs((grad_output_16 - grad_output_64)/grad_output_64))

    res.append(relative_error)
    # res.append(diff2)

    # for n in out_64.numpy().ravel():
    #     if math.isnan(n):
    #         res.append("NAN")
    #         break

    csv_writer.writerow(res)
    # return max(res[1:3])
    return absolute_error,relative_error, grad_output_16, grad_output_64


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
            shape = (1)
            dtype = torch.bfloat16
            # input1 = (torch.randn(shape, dtype=dtype)*10).to(device)
            # input2 = (torch.randn(shape, dtype=dtype)*10).to(device)
            input1 = generate_one_input(shape,dtype,device)
            input2 = generate_one_input(shape,dtype,device)
            abs_error, rel_error, grad_output_16, grad_input64 = get_diff(input1, input2, csv_writer1, opname, "mean", index)
            # acc_error += max_diff
            if abs_error > pre_abs_max:
                pre_abs_max = abs_error
                max_index = index
            if rel_error > pre_rel_max:
                pre_rel_max = rel_error
                if rel_error > global_rel_error_max:
                    print('triggered tensor: ', input1, input2)
                    print('grad output: ', grad_output_16, grad_input64)
            index += 1
        global_abs_error_max = max(global_abs_error_max, pre_abs_max)
        global_rel_error_max = max(global_rel_error_max, pre_rel_max)
        # info.append(pre_max)
        info.append(global_abs_error_max)
        info.append(global_rel_error_max)
        # info.append(acc_error)
        # info.append(max_index)
        csv_writer2.writerow(info)

    out1.close()
    out2.close()
def generate_one_input(shape,dtype,device):
    np_x = np.random.randn(shape)
    x = (torch.from_numpy(np_x).to(dtype)*10).to(device)
    return x
if __name__ == '__main__':
    if sys.argv[1] == "softplus":
        test_torch('softplus_backward')
    elif sys.argv[1] == 'leakyRelu':
        test_torch('leakyrelu_backward')
    elif sys.argv[1] == 'selu':
        test_torch('selu_backward')
    elif sys.argv[1] == 'elu':
        test_torch('elu_backward')