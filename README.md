# OpScanner
OpScanner is a tool designed to perform program analysis on operator implementation. In this repository, we provide LLVM analysis passes for extracting precision casting operations critical to numerical stability. OpScanner can also be used for other purposes by add new passes.

## Directory Structure
The build/ directory contains scripts that transform compile commands and generate LLVM IR for operator implementations.

The analysis/ directory includes LLVM analysis passes for extracting precision casting operations critical to numerical stability.

The testing/ directory houses precision testing tools for activation operators. 
## Usage
#### Build and Compile
Firstly, to obtain the compilation database, you need to compile the DL frameworks from source code. For more details, refer to thehttps://github.com/pytorch/pytorch and https://www.tensorflow.org/install/source. Specifically, when compiling TensorFlow using Bazel, you should add the option --subcommands to print the full command line for each command. After compiling, the compilation database can be found in the build directory for PyTorch. For TensorFlow, you need to capture the compilation database by redirection.

Once you have obtained the compilation database, replace the database directory in the scripts located in the build/ directory. Then, run the scripts to generate the LLVM IR for the CUDA implementation of the operators.

#### Analysis
To build the LLVM passes, run the following command:
```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
Afterwards, running scan.py will display the scanning results in the command line.
#### Precision Testing
We refer to the tool Predoo (https://github.com/predoodl/predoo), reimplement and improve it. We have implemented both the random strategy and the guided strategy proposed by Predoo. You can run the tool by directly calling the Python script in the /testing directory. For example, to test the PyTorch ELU operator, you can run the following command:
```
python test_torch_forward.py elu
```
The results will be saved in the testing/data directory.