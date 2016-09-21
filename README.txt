
1) 'sk_transform.py' contains implemetation for SW-Net to SK-Net transformation script.
2) This script requires modified 'caffe.proto.caffe_pb2' module which includes SK-Net specific 'kstride' parameter for PoolingParameter 
   and ConvolutionParameter message definitions. So use attached modified version of 'caffe.proto.caffe_pb2' module to excecute this script.
3) Detailed description of every class, method and algoritm steps are provided in sk_transform.py.
4) Detailed description about command line arguments has also been provided in sk_transform.py file header.
5) Customized initial input size for SK-Net can be provided as an optional argument to excution command (see command line description).

Available Execution Commands:
    
    Default Command:
        python sk_transform.py input.prototxt output.prototxt

    Optional commands:

    Excecute with SK-Net initial input size:
        python sk_transform.py --sk_dim 229 input.prototxt output.prototxt
    
    Execution with verbose:
        python sk_transform.py input.prototxt output.prototxt -v
        OR
        python sk_transform.py input.prototxt output.prototxt --verbose
    
    Display help:
        python sk_transform.py -h
        OR
        python sk_transform.py --help


Note:
    Provided test case (input: lenet.prototxt, output: fast_lenet.prototxt) is wrong because as per algorithm, in fast_lenet.prototxt file, value for 'kstride' param of "ip2" layer should be 1 insted of 4. However, implemented script 'sk_transform.py' works fine and generates correct SK-Net configuration for input lenet.prototxt.

    I have tested with several other test cases and all of them were succesfully passed.

    Kindly let me know if you have any question or confusion.
    
