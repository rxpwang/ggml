## Readme writen by WRX for conv2d and mul-mat modification and measurement
This readme is written by Rongxiang documenting the modification and usage of test-conv2d.cpp and test-mulmat.cpp.

### General measurement scheme
In both measurement, we do the computation multiple times to get more accurate results. for each measurement, we pre-heat the computation for 200 times, and measure the 1000 times execution latency.The latency is shown after the informing message "1000 times compute finished."

### test-conv2d
usage:
test-conv2d input_dim1 input_dim2 kernel_len use_gpu thread_num

The input_channel and output_channel is pre-set to be all 64.
e.g. for our experiment, the command is as below for input size (1000, 1000), kernel size (3, 3), input channel 64, output channel 64 and cpu thread number 8, with GPU backend.

test-conv2d 1000 1000 3 1 8

By the above command, we get 1000 times computation latency. We can further get the average latency t_avg for one execution from that.

The corresponding computation op counts is 69.632 billion operations,
we can further get the ops/ms = 69.632 billion / t_avg

### test-mulmat
usage:
test-mulmat M N K batch_size use_gpu


e.g. for our experiments. 

In **matrix-matrix multiplication** case, for matrix (1024, 512) multiply (512 * 1024) with batch_size 1 and GPU backend, the command is as below.

test-mulmat 1024 1024 512 1 1

By the above command, we get 1000 times computation latency. We can further get the average latency t_avg for one execution from that.

The corresponding computation op counts is 1.073 billion operations,
we can further get the ops/ms = 1.073 billion / t_avg

In **matrix-vector multiplication** case, for matrix (4096, 2048) multiply (2048, 2) with batch_size 32 and GPU backend, the command is as below

test-mulmat 4096 2 2048 32 1

By the above command, we get 1000 times computation latency. We can further get the average latency t_avg for one execution from that.

The corresponding computation op counts is also 1.073 billion operations,
we can further get the ops/ms = 1.073 billion / t_avg