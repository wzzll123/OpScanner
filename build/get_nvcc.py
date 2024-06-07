import sys
import os
import io
import subprocess
import json

if __name__ == '__main__':
    compile_log = "command.log"
    nvcc_log = 'nvcc.log'
    # clear
    with open(nvcc_log, 'w'):
        pass
    with open(compile_log, 'r') as f:
        for line in f.readlines():
            if 'crosstool_wrapper_driver_is_not_gcc' not in line:
                continue
            if '-c tensorflow/core/kernels' not in line:
                continue
            command = line.strip()[:-1] # remove newline character and ')' 
            if '-x cuda' not in command:
                continue
            command_option = command.split(' ')
            command_option[0] = './crosstool_wrapper_driver_is_not_gcc'
            command = " ".join(command_option)
            command += ' --cuda_log'
            # print(command)
            # Use subprocess.Popen to execute the command and capture stdout
            with subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, text=True) as process:
                # Read the stdout
                stdout_content = process.stdout.read()
                # Append the stdout to the file
                with open(nvcc_log, 'a') as output_file:
                    output_file.write(stdout_content)