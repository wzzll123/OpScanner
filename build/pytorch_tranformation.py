import sys
import os
import io
import subprocess
import json
failed = []
# generate bc files in current directory
if __name__ == '__main__':
    # compile_json = "pytorch-v1.11.0/build/compile_commands.json"
    compile_json = "pytorch-v2.1.0/build/compile_commands.json"
    # generate_root_path = '1.11.0-bc'
    # flags = "--cuda-host-only -emit-llvm -g -O0"
    # flags = "--cuda-gpu-arch=sm_70 -emit-llvm -g -O0"
    cxx = "/home/wzz/opt/llvm/llvm16/bin/clang++ --cuda-device-only "
    cc = "/home/wzz/opt/llvm/llvm16/bin/clang "
    black_options = ['-Xcompiler','-Xcudafe','-Xfatbin','-forward-unknown-to-host-compiler','-O1', '-O2', '-O3', '--diag_suppress=', '--expt-relaxed-constexpr', '--expt-extended-lambda', '-compress-all']
    with open(compile_json, 'r') as f:
        cmds = json.load(f)
        for c in cmds:
            
            compile_cmd = c['command'].split(' ')
            is_cuda_op = False
            is_cpu_op = False
            if 'nvcc' in compile_cmd[0] and 'aten/src/ATen/native/' in c['command']:
                is_cuda_op = True
            # if '-c /data2/wzz/pytorch/pytorch-v1.11.0/build/aten/src/ATen/native/cpu/' in c['command'] and 'DEFAULT' in c['command']:
            #     is_cpu_op = True
            if (not is_cuda_op) and (not is_cpu_op):
                continue
            if is_cuda_op:
                compile_cmd[0] = cxx + "-emit-llvm -g -O0"
                # compile_cmd[0] = cxx + "--cuda-gpu-arch=sm_70 " + "-emit-llvm -g -O0"
            elif is_cpu_op:
                compile_cmd[0] = cxx + "-emit-llvm -g -O0" 
            # if 'CumprodKernel.cu' not in c['command']:
            #    continue
            j = 0
            for op in compile_cmd:
                for black_option in black_options:
                    if op.startswith(black_option):
                        compile_cmd[j] = ''
                if op.startswith('-gencode'):
                    compile_cmd[j] = ''
                    compile_cmd[j+1] = ''
                # if op == '-isystem=/usr/local/cuda/include':
                #     compile_cmd[j] = '-isystem=/usr/local/cuda-11.0/include'
                if op == '-o':
                    compile_cmd[j] = ''
                    compile_cmd[j+1] = ''
                    break 
                j += 1
            j += 1
            # file_dir = os.path.join(generate_root_path, c['directory'][1:])
            # if not os.path.exists(file_dir):
            #     os.makedirs(file_dir)
            
            # bc_file_path = os.path.join(file_dir, os.path.basename(compile_cmd[j]).split('.')[0]) + '.bc'
            # compile_cmd[j] = str(bc_file_path)
            source_file = compile_cmd[-3]
            compile_cmd = " ".join(compile_cmd)
            print(c['command'])
            print("==>", compile_cmd)
            try:
                subprocess.run(compile_cmd, shell=True, check=True)
            except:
                failed.append(source_file)
    print(failed)
