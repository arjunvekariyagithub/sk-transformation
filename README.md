# SK-Transformation
Implementation of Sliding Window(SW) network to Strided Kernel(SK) caffe .prototext network conversion algorithm (SKNet Transformation Algorithm) presented in page 14 of paper available at http://arxiv.org/pdf/1509.03371v1.pdf.

#### 'sk_transform.py' contains implemetation for SW-Net to SK-Net transformation script.
#### This script requires modified 'caffe.proto.caffe_pb2' module which includes SK-Net specific 'kstride' parameter for      PoolingParameter and ConvolutionParameter message definitions. So use attached modified version of 'caffe.proto.caffe_pb2' module to excecute this script.
#### Detailed description of every class, method and algoritm steps are provided in sk_transform.py.
#### Detailed description about command line arguments has also been provided in sk_transform.py file header.
#### Customized initial input size for SK-Net can be provided as an optional argument to excution command (see command line description).

### Requirements:

        Attached modified Caffe protobuf layer definitions (https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)
        
        Google protobug package (https://github.com/google/protobuf)
        
        ProtoText package (PyPI) (https://github.com/XericZephyr/prototext)

### Commands:

    Default Command:
        python sk_transform.py <input.prototxt> <output.prototxt>

    Optional commands:

    Excecute with SK-Net initial input size:
        python sk_transform.py <--sk_dim> 229 <input.prototxt> <output.prototxt>
    
    Execution with verbose:
        python sk_transform.py <input.prototxt> <output.prototxt> -v
        OR
        python sk_transform.py <input.prototxt> <output.prototxt> --verbose
    
    Display help:
        python sk_transform.py -h
        OR
        python sk_transform.py --help
      
  
