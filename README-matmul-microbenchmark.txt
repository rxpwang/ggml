FL: based on test/README_RX.md

------------------------------
CHANGELOG
updated by FL, 09/02/24

Readme writen by WRX for conv2d and mul-mat modification and measurement
This readme is written by Rongxiang documenting the modification and usage of test-conv2d.cpp and test-mulmat.cpp.

------------------------------

### To build
(works for both Windows & Linux)

Open a Windows dev console ... 

e: 
cd workspace-ggml/ggml/

--------
# CONFIG, one time 
# under ggml/
### older code 
cmake -B build-cuda -DGGML_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES="50;52;61;70;86" 
#### newer code 
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="50;52;61;70;86" 
### NB: 50 for quadro k2200, 86 for RTX3050

--------
# BUILD
# windows: do it from x64 Native tools comamnd prompot. no vs launch needed
cmake --build build-cuda -j --config Release

----------------------------------------------

### General measurement scheme
In both measurement, we do the computation multiple times to get more accurate results. 
for each measurement, we pre-heat the computation for 200 times, and measure the 1000 times execution latency.
The latency is shown after the informing message "1000 times compute finished."

-----------------------
### test-conv2d
usage:
build-cuda/bin/Release/test-conv2d input_dim1 input_dim2 kernel_len use_gpu thread_num

The input_channel and output_channel is pre-set to be all 64.
e.g. for our experiment, the command is as below for input size (1000, 1000), kernel size (3, 3), input channel 64, output channel 64 and cpu thread number 8, with GPU backend.

# Windows: (git bash, dev console, etc.)
build-cuda/bin/Release/test-conv2d 1000 1000 3 1 8
# Linux
build-cuda/bin/test-conv2d 1000 1000 3 1 8



By the above command, we get 1000 times computation latency. We can further get the average latency t_avg for one execution from that.

The corresponding computation op counts is 69.632 billion operations,
we can further get the ops/ms = 69.632 billion / t_avg

-----------------------
### test-mulmat
usage:
test-mulmat M N K batch_size use_gpu


e.g. for our experiments. 

In **matrix-matrix multiplication** case, for matrix (1024, 512) multiply (512 * 1024) with batch_size 1 and GPU backend, the command is as below.

# WIN   
build-cuda/bin/Release/test-mul-mat 1024 1024 512 1 1
# Linux 
build-cuda/bin/test-mul-mat 1024 1024 512 1 1


By the above command, we get 1000 times computation latency. We can further get the average latency t_avg for one execution from that.

The corresponding computation op counts is 1.073 billion operations,
we can further get the ops/ms = 1.073 billion / t_avg

-----------------------

### **matrix-vector multiplication** case, 

for matrix (4096, 2048) multiply (2048, 2) with batch_size 32 and GPU backend, the command is as below

# Win
build-cuda/bin/Release/test-mul-mat 4096 2 2048 32 1
# Linux 
build-cuda/bin/test-mul-mat 4096 2 2048 32 1

By the above command, we get 1000 times computation latency. We can further get the average latency t_avg for one execution from that.

The corresponding computation op counts is also 1.073 billion operations,
we can further get the ops/ms = 1.073 billion / t_avg
