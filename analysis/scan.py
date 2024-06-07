import subprocess
import os


def analyze_one_file(file_name):
    # Command to run
    command = [
        "opt",
        "-load-pass-plugin=build/lib/libFunctionInfo.so",
        "-passes=scan-api",
        file_name,
        "-disable-output"
    ]
    print(file_name)
    # Run the command and capture output
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        str_result = output.decode('utf-8')  # Decode bytes to string
        print(str_result)
        return str_result
    except subprocess.CalledProcessError as e:
        # If the command fails, print the error
        print("Error:", e.output.decode('utf-8'))  # Decode bytes to string
        return None

increase_dict = {}
file_names = {}
user_type_count = {}

def process_result(result):
    if result is None:
        return
    lines = result.split('\n')
    for i in range(len(lines)):
        line = lines[i]
        if 'Variable at function' not in line:
            continue
        func_line = line
        func_name = func_line.split(': ')[1]

        file_line = lines[i+1]
        file_name = file_line.split(': ')[1]

        if 'Use API:' in lines[i+3]:
            api_line = lines[i+3]
            user_line = lines[i+4]
        else:
            api_line = lines[i+4]
            user_line = lines[i+5]
        api_name = api_line.split(': ')[1]

        # if 'fcmp' in user_line:
        #     user_name = 'fcmp'
        # elif 'fadd' in user_line or 'fmul' in user_line or 'fsub' in user_line or 'fdiv' in user_line or 'operator float' not in api_line:
        #     user_name = 'calculate'
        # elif 'llvm' in user_line or '_ZL3maxff' in user_line or '_ZN6thrust4pairIffEC1ERKfS3_' in user_line or '_ZL3minff' in user_line or '_ZL5rsqrtf' in user_line:
        #     user_name = 'call non calculate'
        # elif '':
        #     user_name = 'call calculate'
        # else:
        #     print('manual check', user_line)

        if file_name in file_names:
            continue

        if func_name not in increase_dict:
            increase_dict[func_name] = {}

        file_names[file_name] = 1
        increase_dict[func_name][file_name] = api_name

def analyze_pytorch():
    global increase_dict
    increase_dict = {}
    # bc_dir = '/data2/wzz/pytorch/cuda-bc'
    # bc_dir = '/data2/wzz/pytorch/pytorchv2.1-cuda-bc'
    bc_dir = '/home/wzz/tensorflow/tensorflow-2.11.0-new/tensorflow-2.11.0/cuda-bc'
    for bc_name in os.listdir(bc_dir):
        if not bc_name.endswith('.bc'):
            continue
        file_path = os.path.join(bc_dir, bc_name)
        process_result(analyze_one_file(file_path))
        # print(increase_dict)
    # import json
    # api_json = json.dumps(increase_dict, indent=4)
    # print(api_json)
analyze_pytorch()
    