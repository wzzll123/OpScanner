import sys
import os
import io
import subprocess
import json
import shlex
compile_json = "compile_commands.json"
nvcc_log = "nvcc.log"
generate_root_path = 'bc'
# flags = "--cuda-host-only -emit-llvm -g -O0"
flags = "--cuda-gpu-arch=sm_70 -emit-llvm -g -O0"
cxx = "/usr/bin/clang++-11 " + flags
black_options = ['-O1', '-O2', '-O3', '--diag_suppress=', '--expt-relaxed-constexpr', 
'--expt-extended-lambda', '-compress-all', '-nvcc_options', '--compiler-bindir', 
'--ftz', '--fatbin-options','-fno-canonical-system-headers', '-g0']
def handle_cmd(cmd: str) -> str:
    compile_cmd_options = shlex.split(cmd)
    # if '-x cuda' not in cmd or '-c tensorflow/core/kernels' not in cmd:
    #     return ''
    compile_cmd_options[0] = cxx
    j = 0
    for op in compile_cmd_options:
        for black_option in black_options:
            if op.startswith(black_option):
                compile_cmd_options[j] = ''
        if op.startswith('-gencode'):
            compile_cmd_options[j] = ''
            compile_cmd_options[j+1] = ''
        if op.startswith('--compiler-options'):
            compile_cmd_options[j] = ''
            compile_cmd_options[j+1] = compile_cmd_options[j+1].replace('-fno-canonical-system-headers','')
        if op.startswith('-fno-canonical-system-headers'):
            compile_cmd_options[j] = ''
        if op == '-o':
            compile_cmd_options[j] = ''
            compile_cmd_options[j+1] = ''
            break 
        j +=1
    j += 1
    # file_dir = os.path.join(generate_root_path, c['directory'][1:])
    # if not os.path.exists(file_dir):
    #     os.makedirs(file_dir)
    
    # bc_file_path = os.path.join(generate_root_path, os.path.basename(compile_cmd_options[j]).split('.')[0]) + '.bc'
    # compile_cmd_options[j] = str(bc_file_path)

    # put the debug option to the last
    compile_cmd_options.append('-g')
    compile_cmd_options.append('-O0')
    compile_cmd = " ".join(compile_cmd_options)
    print("==>", compile_cmd)
    return compile_cmd
if __name__ == '__main__':

    with open(nvcc_log, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if 'gpus/crosstool: PATH=/usr/bin:$PATH' not in line:
                continue
            begin_index = line.find('/usr/local/cuda-11.2/bin/nvcc')
            compile_cmd = handle_cmd(line[begin_index:])
            # break
            subprocess.run(compile_cmd, shell=True)
        