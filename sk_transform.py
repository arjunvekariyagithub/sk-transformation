"""
    This script implements SW-Net to SK-Net conversion algorithm (SKNet Transformation Algorithm)
    presented in page 14 of paper available at http://arxiv.org/pdf/1509.03371v1.pdf.

    Usage:
        python sk_transform.py [-h] [--sk_dim SK_DIM] [--verbose] input_filename output_filename

        # meta-info about acceptable arguments:

        required arguments:
            input_filename   input '.ptototxt' file
            output_filename  output '.ptototxt' file

        optional arguments:
            -h, --help       show this help message and exit
            --sk_dim SK_DIM  initial input size for SK-Net (default to SW-Net input size if not explicitly provided)
            --verbose, -v    switch to specify verbose mode (default=false)

    Requirements:
        Caffe protobuf layer definitions (https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto)
        Google protobug package (https://github.com/google/protobuf)
        ProtoText package (PyPI) (https://github.com/XericZephyr/prototext)

    Input:
        SW-Net configuration definition file (.prototxt)

    Output:
        Converted SK-Net configuration definition file (.prototxt)

"""

import logging
import sys
import math
import argparse
import ProtoText
logger = logging.getLogger("SKTransformer")

try:
    ''' This script requires modified 'caffe.proto.caffe_pb2' module which includes SK-Net specific 'kstride' parameter
    for PoolingParameter and ConvolutionParameter message definitions. So use attached modified version of
    'caffe.proto.caffe_pb2' module to execute this script '''
    from caffe.proto.caffe_pb2 import NetParameter
except MemoryError:
    print "MemoryError: Please make sure 'caffe' module is properly compiled."
    exit()
except:
    print "Error: Please make sure your protobuf 2.6.x is properly installed."
    exit()

# input channels (R,G,B)
f_init = 3


class Msg(object):
    """
    # ===========================================
    # Class containing Error and Success messages
    # ===========================================

    """

    ERROR_MSG_FILE_NAME = "file names must have '.prototext' extensions"
    ERROR_MSG_INNER_PRODUCT_LAYER = ("Ill-formed SW-Network configuration, one or more 'InnerProduct' layers "
                                     "are missing.")
    ERROR_MSG_NON_VISUAL_LAYERS = ("Layer '%s' can not have 'kernel_size' > 1. Kindly check SW-Network "
                                   "configuration to ensure consistency.")
    ERROR_MSG_POOLING_LAYER_INPUT_SIZE = ("Bottom layer '%s' of layer '%s' has inconsistent "
                                          "input size. Necessary correction has been made to input size.")
    ERROR_MSG_POOLING_LAYER_KERNEL_SIZE_STRIDE = ("Pooling layer '%s' must have same 'kernel_size' and 'stride' value."
                                                  "Necessary correction has been made to 'stride' value.")
    SUCCESS_MSG = 'Network converted successfully!!'


class Formatter(object):
    """
    # ==========================================================================
    # Class containing string formats to display SW-Net and SK-Net in table form
    # ==========================================================================

    """

    STRING_FORMAT_SW_TABLE_HEADER = '%20s %30s %7s %7s %3s %3s'
    STRING_FORMAT_SW_TABLE_ROW = '%20s %30s %7d %7d %3d %3d'
    STRING_FORMAT_SK_TABLE_HEADER = '%20s %30s %7s %7s %3s %3s %3s'
    STRING_FORMAT_SK_TABLE_ROW = '%20s %30s %7d %7d %3d %3d %3d'


class LayerType(object):
    """
    # =========================================================
    # Class to contain strings representing network layer types
    # =========================================================

    """

    # Vision layers for SW-Net
    CONVOLUTION = 'Convolution'
    POOLING = 'Pooling'

    # Fully connected layers
    INNER_PRODUCT = 'InnerProduct'
    # Data layer
    DATA = 'Data'

    # Vision layers for SK-Net
    CONVOLUTION_SK = 'ConvolutionSK'
    POOLING_SK = 'PoolingSK'


def parse_args():
    """
    # ===========================================
    # Parse arguments supplied from command lines
    # ===========================================

    """

    parser = argparse.ArgumentParser(description='A script to convert SW-Net to SK-Net')
    parser.add_argument('input_filename', metavar='input_filename', type=str, nargs=1,
                        help="input '.ptototxt' file")
    parser.add_argument('output_filename', metavar='output_filename', type=str, nargs=1,
                        help="output '.ptototxt' file")

    parser.add_argument('--sk_dim', type=int,
                        help='initial input size for SK-Net (default to SW-Net input size if not '
                             'explicitly provided)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='switch to specify verbose mode (default=false)')

    args = parser.parse_args()

    return parser, args


def get_bottom_layer_info(layers, i, skip_layer='None'):
    """
    # ===================================================================
    # find Visual bottom layer (other than skip_layer) of specified layer
    # ===================================================================

    Args:
        layers: List of network layers
        i: index of layer to find bottom layer for
        skip_layer: layer to skip when finding bottom layer

    Returns:
        index and param_type of bottom visual layer

    """
    # Step 1
    while layers[i]['type'].__contains__(skip_layer) or not(is_visual_layer(layers[i])):
        i -= 1

    # Step 2
    if layers[i]['type'].__contains__(LayerType.CONVOLUTION):
        return i, 'convolution_param'
    elif layers[i]['type'].__contains__(LayerType.POOLING):
        return i, 'pooling_param'
    elif layers[i]['type'].__contains__(LayerType.INNER_PRODUCT):
        return i, 'inner_product_param'


def is_visual_layer(layer):
    """
    # =========================================
    # check if specified layer is Visual or not
    # =========================================

    """
    return is_conv_layer(layer) or is_pool_layer(layer) or is_ip_layer(layer)


def is_data_layer(layer):
    """
     # =================================================
     # check if specified layer is of type 'Data' or not
     # =================================================
     """
    return layer['type'].__contains__(LayerType.DATA)


def is_conv_layer(layer):
    """
     # ========================================================
     # check if specified layer is of type 'Convolution' or not
     # ========================================================
     """

    return layer['type'].__contains__(LayerType.CONVOLUTION)


def is_pool_layer(layer):
    """
     # ====================================================
     # check if specified layer is of type 'Pooling' or not
     # ====================================================
     """

    return layer['type'].__contains__(LayerType.POOLING)


def is_ip_layer(layer):
    """
     # =========================================================
     # check if specified layer is of type 'InnerProduct' or not
     # =========================================================
     """

    return layer['type'].__contains__(LayerType.INNER_PRODUCT)


def print_sw_net(transformer):
    """
    # ============================
    # Print SW-Net in table format
    # ============================

    Args:
        transformer: object of SKTransformer class

    Terminology:
        layer -> layer name
        l_type -> layer type
        w -> layer input size
        f -> number of feature maps
        k -> kernel size
        s -> stride

    """
    print('\n' * 3 + '=' * 76)
    print(' ' * 26 + 'SW-Network configuration')
    print('-' * 76)
    print(Formatter.STRING_FORMAT_SW_TABLE_HEADER % ('Layer', 'Type', 'W', 'f.out', 'k', 's'))
    print('-' * 76)

    # print initial 'Data' layer if explicitly definition does not available in SW-Net config
    if not transformer.has_data_layer:
        print(Formatter.STRING_FORMAT_SW_TABLE_ROW % (transformer.sw_layers[0], transformer.sw_layers[0],
                                                      transformer.sw_layers_input_size[0], f_init, 1, 1))

    i = 1
    while i < len(transformer.sw_layers):
        # common params for all layers
        layer = transformer.sw_layers[i]['name']
        l_type = transformer.sw_layers[i]['type']
        w = transformer.sw_layers_input_size[i]
        # assign default values for f, k & s params
        f = f_init
        k = 1
        s = 1

        # process f, k & s params only for layers except 'Data' layers. 'Data' layer uses default values
        if not is_data_layer(transformer.sw_layers[i]):
            bottom_index, bottom_param_type = get_bottom_layer_info(transformer.sw_layers, i)

            if not is_visual_layer(transformer.sw_layers[i]):
                # only f param is available for non-visual layers. Use default values for k & s params
                f = transformer.sw_layers[bottom_index][bottom_param_type]['num_output']
            else:
                if is_pool_layer(transformer.sw_layers[i]):
                    # pooling layers does not posses 'num_output' param so fetch it from bottom visual layer
                    bottom_index_pool, bottom_param_type_pool = get_bottom_layer_info(transformer.sw_layers, i,
                                                                                      skip_layer=LayerType.POOLING)
                    f = transformer.sw_layers[bottom_index_pool][bottom_param_type_pool]['num_output']
                    k = transformer.sw_layers[bottom_index][bottom_param_type]['kernel_size']
                    if 'stride' in transformer.sw_layers[bottom_index][bottom_param_type]:
                        s = transformer.sw_layers[bottom_index][bottom_param_type]['stride']
                elif is_ip_layer(transformer.sw_layers[i]):
                    # for 'InnerProduct' layer fetch params from already converted corresponding 'ConvolutionSK' layer
                    f = transformer.sk_layers[i]['convolution_param']['num_output']
                    k = transformer.sk_layers[i]['convolution_param']['kernel_size'][0]
                    s = transformer.sk_layers[i]['convolution_param']['stride'][0]
                    # print next non-visual layer to gather with 'InnerProduct' layer (i.e InnerProduct + ReLU)
                    if (i + 1) < (len(transformer.sw_layers) - 1) and not is_visual_layer(transformer.sw_layers[i + 1]):
                        layer += (" + " + transformer.sw_layers[i + 1]['name'])
                        l_type += (" + " + transformer.sw_layers[i + 1]['type'])
                        i += 1
                else:  # come here for Convolution layer
                    f = transformer.sw_layers[bottom_index][bottom_param_type]['num_output']
                    k = transformer.sw_layers[bottom_index][bottom_param_type]['kernel_size'][0]
                    if 'stride' in transformer.sw_layers[bottom_index][bottom_param_type]:
                        s = transformer.sw_layers[bottom_index][bottom_param_type]['stride'][0]
                    # print next non-visual layer to gather with 'Convolution' layer (i.e Convolution + ReLU)
                    if (i + 1) < (len(transformer.sw_layers) - 1) and not is_visual_layer(transformer.sw_layers[i + 1]):
                        layer += (" + " + transformer.sw_layers[i + 1]['name'])
                        l_type += (" + " + transformer.sw_layers[i + 1]['type'])
                        i += 1

        print(Formatter.STRING_FORMAT_SW_TABLE_ROW % (layer, l_type, w, f, k, s))
        if i == len(transformer.sw_layers) - 1:
            print('=' * 76)

        i += 1


def print_sk_net(transformer):
    """
    # ============================
    # Print SK-Net in table format
    # ============================

    Args:
        transformer: object of SKTransformer class

    Terminology:
        layer -> layer name
        l_type -> layer type
        w -> layer input size
        f -> number of feature maps
        k -> kernel size
        s -> stride
        d -> kernel stride

    """
    print('\n' * 3 + '=' * 80)
    print(' ' * 24 + 'Resulted SK-Network configuration')
    print('-' * 80)
    print(Formatter.STRING_FORMAT_SK_TABLE_HEADER % ('Layer', 'Type', 'W', 'f.out', 'k', 's', 'd'))
    print('-' * 80)

    # print initial 'Data' layer if explicitly definition does not available in SW-Net config
    if not transformer.has_data_layer:
        print(Formatter.STRING_FORMAT_SK_TABLE_ROW % (transformer.sk_layers[0], transformer.sk_layers[0],
                                                      transformer.sk_layers_input_size[0], f_init, 1, 1, 1))
    i = 1
    while i < len(transformer.sk_layers):
        # common params for all layers
        layer = transformer.sk_layers[i]['name']
        l_type = transformer.sk_layers[i]['type']
        w = transformer.sk_layers_input_size[i]
        # assign default values for f, k, s & d params
        f = f_init
        k = 1
        s = 1
        d = 1

        # process f, k, s & d params only for layers except 'Data' layers. 'Data' layer uses default values
        if not is_data_layer(transformer.sk_layers[i]):
            bottom_index, bottom_param_type = get_bottom_layer_info(transformer.sk_layers, i)

            if not is_visual_layer(transformer.sk_layers[i]):
                # only f param is available for non-visual layers. Use default values for k, s & d params
                f = transformer.sk_layers[bottom_index][bottom_param_type]['num_output']
            else:
                d = transformer.sk_layers[bottom_index][bottom_param_type]['kstride']
                if is_pool_layer(transformer.sk_layers[i]):
                    # pooling layers does not posses 'num_output' param so fetch it from bottom visual layer
                    bottom_index_pool, bottom_param_type_pool = get_bottom_layer_info(transformer.sk_layers, i,
                                                                                      skip_layer=LayerType.POOLING)
                    f = transformer.sk_layers[bottom_index_pool][bottom_param_type_pool]['num_output']
                    k = transformer.sk_layers[bottom_index][bottom_param_type]['kernel_size']
                    s = transformer.sk_layers[bottom_index][bottom_param_type]['stride']
                else:  # come here for ConvolutionSK layer
                    f = transformer.sk_layers[bottom_index][bottom_param_type]['num_output']
                    k = transformer.sk_layers[bottom_index][bottom_param_type]['kernel_size'][0]
                    if 'stride' in transformer.sk_layers[bottom_index][bottom_param_type]:
                        s = transformer.sk_layers[bottom_index][bottom_param_type]['stride'][0]
                    # print next non-visual layer to gather with 'ConvolutionSK' layer (i.e ConvolutionSK + ReLU)
                    if (i + 1) < (len(transformer.sk_layers) - 1) and not is_visual_layer(transformer.sk_layers[i + 1]):
                        layer += (" + " + transformer.sk_layers[i + 1]['name'])
                        l_type += (" + " + transformer.sk_layers[i + 1]['type'])
                        i += 1

        print(Formatter.STRING_FORMAT_SK_TABLE_ROW % (layer, l_type, w, f, k, s, d))
        if i == len(transformer.sk_layers) - 1:
            print('=' * 80)

        i += 1


class SKTransformer(object):
    """
    # ============================
    # SW-Net to SK-Net transformer
    # ============================

    Attributes:
        sw_net: object representing entire SW-Net
        sk_net: object representing entire SK-Net
        sw_layers[]: list to contain definitions of SW-Net layers
        sk_layers[]: list to contain definitions of SK-Net layers
        sw_layers_input_size[]: list to contain input sizes of SW-Net layers
        sk_layers_input_size[]: list to contain input sizes of SW-Net layers
        d_temp: kernel stride for SK-Net
        has_data_layer: boolean indicating weather SW-Net contains 'Data' layer definition or not

    """
    sw_net = None
    sk_net = None
    sw_layers = []
    sk_layers = []
    sw_layers_input_size = []
    sk_layers_input_size = []
    d_temp = 1
    has_data_layer = False

    def __init__(self, sw_net, sk_net, sw_init_input_size, sk_init_input_size):
        """
        # ======================================================
        # initialize attributes with default and provided values
        # ======================================================

        Args:
            sw_net: object representing SW-Net
            sk_net: object representing SK-Net
            sk_init_input_size: initial input size for SK-Net

        """
        self.sw_net = sw_net
        self.sk_net = sk_net

        # add initial 'Data' layers
        self.sw_layers.append('Data')
        self.sk_layers.append('Data')

        # prepare SW-Net layer list
        self.sw_layers += self.sw_net['layer']
        # prepare initial SK-Net layer list and convert each layer as algorithm progresses
        self.sk_layers += self.sk_net['layer']

        logging.debug('Network layer count: %d' % (len(self.sw_layers) - 1))

        # add initial input size of SW-Net and SK-Net into corresponding list
        self.sw_layers_input_size.append(sw_init_input_size)
        self.sk_layers_input_size.append(sk_init_input_size)

        # initial kernel stride
        self.d_temp = 1

    def transform(self):
        """
        # ==========================================
        # Convert SW-Net in to corresponding SK-Net
        # ==========================================

        #Steps:
        1. make 'name' param for SK-Net
        2. make 'input_dim' params (h and w) for SK-Net
        3. process each SW-Net layer and convert into corresponding SK-Net layer

        """
        # step 1
        self.sk_net['name'] = 'Fast' + self.sw_net['name']

        # step 2
        if 'input_dim' in self.sk_net:
            self.sk_net['input_dim'][2] = self.sk_layers_input_size[0]
            self.sk_net['input_dim'][3] = self.sk_layers_input_size[0]

        # step 3
        for i in range(1, len(self.sw_layers)):
            logging.debug("[%d] - Converting '%s' layer" % (i, self.sw_layers[i]['name']))

            if self.sw_layers[i]['type'] == LayerType.CONVOLUTION:
                self.transform_convolution_layer(i)
            elif self.sw_layers[i]['type'] == LayerType.POOLING:
                self.transform_pooling_layer(i)
            elif self.sw_layers[i]['type'] == LayerType.INNER_PRODUCT:
                self.transform_inner_product_layer(i)
            else:  # come here for activation and loss layers

                # check for inconsistent kernel_size and raise error if found inconsistency
                if 'kernel_size' in self.sw_layers[i]['convolution_param'] and \
                                self.sw_layers[i]['convolution_param']['kernel_size'][0] > 1:
                    logging.error(Msg.ERROR_MSG_NON_VISUAL_LAYERS % (self.sw_layers[i]['name']))
                    exit()
                # check for 'Data' layer definition availability
                if self.sw_layers[i]['type'].__contains__(LayerType.DATA):
                    self.has_data_layer = True

                # for activation and loss layer input size is same as bottom layer
                self.sk_layers_input_size.append(self.sk_layers_input_size[i - 1])
                self.sw_layers_input_size.append(self.sw_layers_input_size[i - 1])

            # correct inconsistent input size
            if self.sk_layers_input_size[i] < 1:
                self.sk_layers_input_size[i] = 1
            if self.sw_layers_input_size[i] < 1:
                self.sw_layers_input_size[i] = 1

            logging.debug("[%d] - '%s' layer --> Done" % (i, self.sw_layers[i]['name']))

    def transform_convolution_layer(self, i):
        """
        # =============================================================================================
        # Convert specified SW-Net 'Convolution' layer in to corresponding SK-Net 'ConvolutionSK' layer
        # =============================================================================================

        # Steps:
        1. Make 'type', 'kernel_size', 'kstride' and 'stride' params for SK-Net 'ConvolutionSK' layer.
        2. Calculate input size for 'Convolution' and 'ConvolutionSK' layers of SW-Net and SK-Net respectively.

        # Rest params will remain exactly the same for SW and SK networks.
        # After conversion SK-Net 'ConvolutionSK' layer will look as shown below.

        layer {
            name: "conv1"
            type: "ConvolutionSK"
            bottom: "data"
            top: "conv1"

            param { lr_mult: 1.0 }
            param { lr_mult: 2.0 }

            convolution_param {
                num_output: 48
                kernel_size: 7
                stride: 1
                kstride: 1

                weight_filler {
                    type: "xavier"
                }
                bias_filler {
                    type: "constant"
                }
            }
        }

        """
        # step 1
        self.sk_layers[i]['type'] = LayerType.CONVOLUTION_SK
        self.sk_layers[i]['convolution_param']['kernel_size'][0] = \
            self.sw_layers[i]['convolution_param']['kernel_size'][0]
        self.sk_layers[i]['convolution_param']['kstride'] = self.d_temp
        if 'stride' in self.sk_layers[i]['convolution_param']:
            self.sk_layers[i]['convolution_param']['stride'][0] = 1

        # step 2
        self.sk_layers_input_size.append(int((self.sk_layers_input_size[i - 1] -
                                              ((self.sk_layers[i]['convolution_param']['kernel_size'][0] - 1) *
                                               self.sk_layers[i]['convolution_param']['kstride']))))

        self.sw_layers_input_size.append(int((self.sw_layers_input_size[i - 1] -
                                              (self.sw_layers[i]['convolution_param']['kernel_size'][0] - 1))))

    def transform_pooling_layer(self, i):
        """
        # =====================================================================================
        # Convert specified SW-Net 'Pooling' layer in to corresponding SK-Net 'PoolingSK' layer
        # =====================================================================================

        # Steps:
        1. Check for inconsistent amongst 'kernel_size' and 'stride' params, update values if necessary.
        2. Make 'type', 'kernel_size', 'kstride' and 'stride' params for SK-Net 'PoolingSK' layer
        3. Calculate input size for 'Pooling' and 'PoolingSK' layers of SW-Net and SK-Net respectively
        4. Check for inconsistent amongst bottom layer input size and 'kernel_size', update bottom layer input size
           if found inconsistency
        5. Update d_temp value.

        # Rest params will remain exactly the same for SW and SK networks
        # After conversion SK-Net 'PoolingSK' layer will look as shown below

        layer {
            name: "pool1"
            type: "PoolingSK"
            bottom: "conv1"
            top: "pool1"

            pooling_param {
                pool: MAX
                kernel_size: 2
                stride: 1
                kstride: 1
            }
        }

        """
        # step 1
        ''' Contrary to algorithm, instead of exiting with error this script continues after fixing inconsistent
        stride value '''
        if self.sw_layers[i]['pooling_param']['kernel_size'] != self.sw_layers[i]['pooling_param']['stride']:
            logging.warning(Msg.ERROR_MSG_POOLING_LAYER_KERNEL_SIZE_STRIDE % (self.sw_layers[i]['name']))
            self.sw_layers[i]['pooling_param']['stride'] = self.sw_layers[i]['pooling_param']['kernel_size']

        # Step 2
        self.sk_layers[i]['type'] = LayerType.POOLING_SK
        self.sk_layers[i]['pooling_param']['kernel_size'] = self.sw_layers[i]['pooling_param']['kernel_size']
        self.sk_layers[i]['pooling_param']['kstride'] = self.d_temp
        self.sk_layers[i]['pooling_param']['stride'] = 1

        # step 3
        self.sk_layers_input_size.append(int((self.sk_layers_input_size[i - 1] -
                                              ((self.sk_layers[i]['pooling_param']['kernel_size'] - 1) *
                                               self.sk_layers[i]['pooling_param']['kstride']))))

        self.sw_layers_input_size.append(int(math.ceil(float(self.sw_layers_input_size[i - 1]) /
                                                       self.sw_layers[i]['pooling_param']['kernel_size'])))

        # step 4
        ''' Contrary to algorithm, instead of exiting with error this script continues after fixing inconsistent
        input size '''
        if self.sw_layers_input_size[i - 1] % self.sw_layers[i]['pooling_param']['kernel_size'] != 0:
            index, param_type = get_bottom_layer_info(self.sw_layers, i - 1)
            logging.warning(Msg.ERROR_MSG_POOLING_LAYER_INPUT_SIZE % (self.sw_layers[index]['name'],
                                                                      self.sw_layers[i]['name']))
            self.sw_layers_input_size[i-1] = (self.sw_layers[i]['pooling_param']['kernel_size'] *
                                              self.sw_layers_input_size[i])

        # step 5
        self.d_temp *= self.sk_layers[i]['pooling_param']['kernel_size']

    def transform_inner_product_layer(self, i):
        """
        # ==============================================================================================
        # Convert specified SW-Net 'InnerProduct' layer in to corresponding SK-Net 'ConvolutionSK' layer
        # ==============================================================================================

        # Steps:
        1. Make 'type', 'kernel_size', 'kstride' and 'stride', 'num_output', 'weight_filler' and 'bias_filler'
           params for SK-Net 'ConvolutionSK' layer.
        2. Delete params for 'InnerProduct' layer.
        3. Calculate input size for 'InnerProduct' and 'ConvolutionSK layers of SW-Net and SK-Net respectively.
        4. Update value of d_temp

        # Rest params will remain exactly the same for SW and SK networks.
        # After conversion SK-Net 'ConvolutionSK' layer will look as shown below.

        layer {
            name: "ip1"
            type: "ConvolutionSK"
            bottom: "pool3"
            top: "ip1"

            param { lr_mult: 1.0 }
            param { lr_mult: 2.0 }

            convolution_param {
                num_output: 48
                kernel_size: 7
                stride: 1
                kstride: 1

                weight_filler {
                    type: "xavier"
                }
                bias_filler {
                    type: "constant"
                }
            }
        }

        """
        # step 1
        self.sk_layers[i]['type'] = LayerType.CONVOLUTION_SK
        self.sk_layers[i]['convolution_param']['kernel_size'].append(self.sw_layers_input_size[i - 1])
        self.sk_layers[i]['convolution_param']['stride'].append(1)
        self.sk_layers[i]['convolution_param']['kstride'] = self.d_temp
        self.sk_layers[i]['convolution_param']['num_output'] = \
            self.sw_layers[i]['inner_product_param']['num_output']

        self.sk_layers[i]['convolution_param']['weight_filler']['type'] = "xavier"
        self.sk_layers[i]['convolution_param']['bias_filler']['type'] = "constant"

        # step 2
        del self.sk_layers[i]['inner_product_param']

        # step 3
        self.sk_layers_input_size.append(int((self.sk_layers_input_size[i - 1] -
                                              ((self.sk_layers[i]['convolution_param']['kernel_size'][0] - 1) *
                                               self.sk_layers[i]['convolution_param']['kstride']))))
        self.sw_layers_input_size.append(1)

        # step 4
        self.d_temp = 1


def main():
    # parse supplied arguments
    parser, args = parse_args()

    # set log level based on '--verbose/-v' argument
    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG, stream=sys.stdout)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING, stream=sys.stdout)

    # flag error for invalid file names
    if not args.input_filename[0].endswith('.prototxt') or not args.output_filename[0].endswith('.prototxt'):
        parser.error(Msg.ERROR_MSG_FILE_NAME)

    # read SW-Net configuration file
    f = open(args.input_filename[0], 'r')
    sw_net_config_text = f.read()
    # create object representing SW-Net config
    sw_net = NetParameter()
    sw_net.ParseFromText(sw_net_config_text)
    # create object representing SK-Net config
    sk_net = NetParameter()
    sk_net.ParseFromText(sw_net_config_text)

    global f_init
    # fetch f_init & input patch dimension (height or width) if available
    if 'input_dim' in sw_net:
        f_init = sw_net['input_dim'][1]
        sw_initial_input_size = sw_net['input_dim'][3]
    else:
        # assign default value if not available in SW-Net
        sw_initial_input_size = 108

    sk_initial_input_size = sw_initial_input_size
    # if available, use supplied initial input size for SK-Net
    if args.sk_dim is not None:
        sk_initial_input_size = args.sk_dim

    transformer = SKTransformer(sw_net, sk_net, sw_initial_input_size, sk_initial_input_size)
    # call to convert SW-Net into SK-Net
    transformer.transform()

    ''' check for final kernel stride param value. Provided SW-Network configuration is inconsistent
    if kernel stride value not equals to 1'''
    if transformer.d_temp == 1:
        print('\n' * 2 + '>' * 68)
        print(' ' * 18 + Msg.SUCCESS_MSG)
        print('<' * 68 + '\n')
        # in DEBUG mode print SW-Net and SK-Net configuration in table format
        if logger.getEffectiveLevel() == logging.DEBUG:
            print_sw_net(transformer)
            print_sk_net(transformer)

        # save converted SK-Net in to supplied output file
        output_file = open(args.output_filename[0], 'w')
        output_file.write(transformer.sk_net.SerializeToText())
        output_file.close()
    else:
        logging.error(Msg.ERROR_MSG_INNER_PRODUCT_LAYER)

if __name__ == '__main__':
    main()
