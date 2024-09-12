with open('scan_pytorch.txt') as scan_file:
    lines = [line.rstrip() for line in scan_file]

result = {}
import re
def extract_n(input_str):
    match = re.search(r'The (\d+)st user', input_str)
    if match:
        return int(match.group(1))
    return None
def get_user_type(user_str: str):
    arithmetic_type = ['fadd', 'fsub', 'fmul', 'fdiv', 'fneg']
    for arith in arithmetic_type:
        if arith in user_str:
            return 'arithmetic'
    if 'fcmp' in user_str:
        return 'compare'
    non_computional = ['llvm.nvvm.fabs.f','llvm.nvvm.trunc.f','llvm.nvvm.floor.f','llvm.nearbyint.f32','llvm.nvvm.ceil.f','_ZL3maxff','_ZL3minff']
    if 'call' in user_str:
        for call in non_computional:
            if call in user_str:
                return 'call non-compute'
        return 'call normal'
    cast_type = ['fpext', 'bitcast', 'fptosi']
    for cast in cast_type:
        if cast in user_str:
            return 'cast' 
    if 'ret' in user_str:
        return 'return'
    if 'load' in user_str:
        return 'load'
    return None

def is_filtered(use_api:str, users:list) -> bool:
    if len(users) == 0:
        return False
    for arith in '+-*/':
        if arith in use_api:
            return False
    for user in users:
        if user not in ['call non-compute', 'compare']:
            return False
    return True
i=0
for i in range(len(lines)):
    line = lines[i]
    if line.startswith('Increse at:'):
        increase_location = line.split(': ')[1]
        use_api = lines[i+2]
        result[increase_location] = {}
        result[increase_location]['use_api'] = use_api
        result[increase_location]['users'] = []
        j = i
        while j < len(lines):
            if lines[j].startswith('Variable at function:'):
                break
            if extract_n(lines[j]) is not None:
                user = lines[j]
                result[increase_location]['users'].append(get_user_type(user))
            j=j+1
    i=i+1
print('before filter:', len(result.keys()))
print(result['/home/zzwen/pytorch-v2.1.0/aten/src/ATen/native/cuda/UnaryFractionKernels.cu:16'])
filtered_result = []
for increase_location in result:
    if not is_filtered(result[increase_location]['use_api'], result[increase_location]['users']):
        filtered_result.append(increase_location)
print('after filter', len(filtered_result))

module2num={
    'Normalization': 0,
    'Image Processing': 0,
    'Transformer': 0,
    'Math': 0,
    'CNN': 0,
    'Activation': 0,
    'RNN': 0,
    'Distribution': 0,
    'Loss': 0,
    'Optimizers': 0,
    'Others': 0,
}
module2files={
    'Math': ['Lerp','Unary','Log', 'Binary','Pointwise','Compare','CopysignKe']

}
for file_location in filtered_result:
    if 'attention' in file_location:
        module2num['Transformer'] += 1
    elif 'RNN' in file_location:
        module2num['RNN'] += 1
    elif 'Norm' in file_location or 'norm' in file_location:
        module2num['Normalization'] += 1
    elif 'Sample' in file_location:
        module2num['Image Processing'] += 1
    elif 'Activation' in file_location:
        module2num['Activation'] += 1
    elif 'Conv' in file_location or 'Pool' in file_location or 'im2col' in file_location or 'vol2col' in file_location:
        module2num['CNN'] += 1
    elif 'adam' in file_location:
        module2num['Optimizers'] += 1
    elif 'Distribution' in file_location or 'Multinomial' in file_location:
        module2num['Distribution'] += 1
    elif 'Loss' in file_location or 'MultiLabel' in file_location:
        module2num['Loss'] += 1
    elif 'Lerp' in file_location or 'Unary' in file_location or 'Binary' in file_location or 'Pointwise' in file_location \
        or 'Log' in file_location or 'Compare' in file_location or 'CopysignKe' in file_location or 'Foreach' in file_location:
        print(file_location)
        module2num['Math'] += 1
    else:
        module2num['Others'] += 1
print(module2num)

        
