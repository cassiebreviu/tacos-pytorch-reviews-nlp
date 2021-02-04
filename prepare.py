#import os
#import sys
#
#
#def convert(imgf, labelf, outf, n):
#    f = open(imgf, "rb")
#    # any data prep goes here
#
#
#mounted_input_path = sys.argv[1]
#mounted_output_path = sys.argv[2]
#os.makedirs(mounted_output_path, exist_ok=True)
#
#convert(os.path.join(mounted_input_path, 'mnist-fashion/train-images-idx3-ubyte'),
#        os.path.join(mounted_input_path, 'mnist-fashion/train-labels-idx1-ubyte'),
#        os.path.join(mounted_output_path, 'mnist_train.csv'), 60000)
#convert(os.path.join(mounted_input_path, 'mnist-fashion/t10k-images-idx3-ubyte'),
#        os.path.join(mounted_input_path, 'mnist-fashion/t10k-labels-idx1-ubyte'),
#        os.path.join(mounted_output_path, 'mnist_test.csv'), 10000)