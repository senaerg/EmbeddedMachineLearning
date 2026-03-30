###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
Common helper services to manage the STM.AI type
"""
import enum
import numpy as np

RT_STM_AI_NAME = 'STM.AI'


def stm_ai_error_to_str(err_code, err_type):
    """
    Return a human readable description of the ai run-time error

    see <AICore>::EmbedNets git (src/api/ai_platform.h file)

    Parameters
    ----------
    err_code
        stm.ai error code
    err_type
        stm.ai error err_type

    Returns
    -------
    str
        human readable description of the error

    """

    type_switcher = {
        0x00: 'None',
        0x01: 'Tools Platform Api Mismatch',
        0x02: 'Types Mismatch',
        0x10: 'Invalid handle',
        0x11: 'Invalid State',
        0x12: 'Invalid Input',
        0x13: 'Invalid Output',
        0x14: 'Invalid Param',
        0x30: 'Init Failed',
        0x31: 'Allocation Failed',
        0x32: 'Deallocation Failed',
        0x33: 'Create Failed'
    }

    code_switcher = {
        0x0000: 'None',
        0x0010: 'Network',
        0x0011: 'Network Params',
        0x0012: 'Network Weights',
        0x0013: 'Network Activations',
        0x0014: 'Layer',
        0x0015: 'Tensor',
        0x0016: 'Array',
        0x0017: 'Invalid Ptr',
        0x0018: 'Invalid Size',
        0x0020: 'Invalid Format',
        0x0021: 'Invalid Batch',
        0x0030: 'Missed Init',
        0x0040: 'In Use',
    }

    desc_ = 'type="{}"(0x{:x})'.format(type_switcher.get(err_type, str(err_type)), err_type)
    desc_ += ', code="{}"(0x{:x})'.format(code_switcher.get(err_code, str(err_code)), err_code)

    return desc_


def stm_ai_node_type_to_str(op_type_id, with_id=True):
    """  # noqa: DAR005
    Return a human readable description of a c-node

    see <AICore>::EmbedNets git (src/layers/layers_list.h file)

    Parameters
    ----------
    op_type_id
        stm.ai operator id
    with_id:
        add numerical value in the description

    Returns
    -------
    str
        human readable description of the c-node
    """

    base = 0x100  # stateless operator
    base_2 = 0x180  # stateful operator
    op_type_id = op_type_id & 0xFFFF
    ls_switcher = {
        0: 'Output',
        base: 'Base',
        base + 1: 'Add',
        base + 2: 'BN',
        base + 3: 'Conv2D',
        base + 4: 'Dense',
        base + 5: 'GRU',
        base + 6: 'LRN',
        base + 7: 'NL',
        base + 8: 'Norm',
        base + 9: 'Conv2dPool',
        base + 10: 'Transpose',
        base + 11: 'Pool',
        base + 12: 'Softmax',
        base + 13: 'Split',
        base + 14: 'TimeDelay',
        base + 15: 'TimeDistributed',
        base + 16: 'Concat',
        base + 17: 'GEMM',
        base + 18: 'Upsample',
        base + 19: 'Eltwise',
        base + 20: 'EltwiseInt',
        base + 21: 'InstNorm',
        base + 22: 'Pad',
        base + 23: 'Slice',
        base + 24: 'Tile',
        base + 25: 'Reduce',
        base + 26: 'RNN',
        base + 27: 'Resize',
        base + 28: 'Gather',
        base + 29: 'Pack',
        base + 30: 'UnPack',
        base + 31: 'Container',
        base + 32: 'Lambda',
        base + 33: 'Cast',
        base + 34: 'iForest',
        base + 35: 'SvmReg',
        base + 36: 'AFeatureExtract',
        base + 37: 'SVC',
        base + 38: 'ZipMap',
        base_2: 'Stateful',
        base_2 + 1: 'LSTM',
        base_2 + 2: 'Custom',
    }
    desc_ = '{}'.format(ls_switcher.get(op_type_id, str(op_type_id)))
    if with_id:
        desc_ += ' (0x{:x})'.format(op_type_id)
    return desc_


class AiBufferFormat(enum.IntEnum):
    """ Map C 'ai_buffer_format' enum to Python enum. """  # noqa: DAR101,DAR201,DAR401

    # see api/ai_platform.h file
    AI_BUFFER_FORMAT_NONE = 0x00000040
    AI_BUFFER_FORMAT_FLOAT = 0x01821040

    AI_BUFFER_FORMAT_U8 = 0x00040440
    AI_BUFFER_FORMAT_U16 = 0x00040840
    AI_BUFFER_FORMAT_U32 = 0x00041040
    AI_BUFFER_FORMAT_U64 = 0x00042040

    AI_BUFFER_FORMAT_S8 = 0x00840440
    AI_BUFFER_FORMAT_S16 = 0x00840840
    AI_BUFFER_FORMAT_S32 = 0x00841040
    AI_BUFFER_FORMAT_S64 = 0x00842040

    AI_BUFFER_FORMAT_Q = 0x00840040
    AI_BUFFER_FORMAT_Q7 = 0x00840447
    AI_BUFFER_FORMAT_Q15 = 0x0084084f

    AI_BUFFER_FORMAT_UQ = 0x00040040
    AI_BUFFER_FORMAT_UQ7 = 0x00040447
    AI_BUFFER_FORMAT_UQ15 = 0x0004084f

    AI_BUFFER_FORMAT_BOOL = 0x00060440

    """ Buffer Format Flags """
    MASK = (0x01ffffff)       # non-flag bits
    MASK_Q = (0xffffc000)
    TYPE_MASK = (0x019e3f80)  # bits required for type identification
    FBIT_SHIFT = (0x40)       # zero point for signed fractional bits

    """ Bit mask definitions """
    FLOAT_MASK = 0x1
    FLOAT_SHIFT = 24

    SIGN_MASK = 0x1
    SIGN_SHIFT = 23

    TYPE_ID_MASK = 0xF
    TYPE_ID_SHIFT = 17

    BITS_MASK = 0x7F
    BITS_SHIFT = 7

    FBITS_MASK = 0x7F
    FBITS_SHIFT = 0

    TYPE_NONE = 0x0
    TYPE_FLOAT = 0x1
    TYPE_Q = 0x2
    TYPE_BOOL = 0x3

    FLAG_CONST = (0x1 << 30)
    FLAG_STATIC = (0x1 << 29)
    FLAG_IS_IO = (0x1 << 27)

    @staticmethod
    def to_dict(fmt):
        """Return a dict with fmt field values"""
        # noqa: DAR101,DAR201,DAR401
        _dict = {
            'float': (fmt >> AiBufferFormat.FLOAT_SHIFT) & AiBufferFormat.FLOAT_MASK,
            'sign': (fmt >> AiBufferFormat.SIGN_SHIFT) & AiBufferFormat.SIGN_MASK,
            'type': (fmt >> AiBufferFormat.TYPE_ID_SHIFT) & AiBufferFormat.TYPE_ID_MASK,
            'bits': (fmt >> AiBufferFormat.BITS_SHIFT) & AiBufferFormat.BITS_MASK,
            'fbits': (fmt >> AiBufferFormat.FBITS_SHIFT) & AiBufferFormat.FBITS_MASK,
        }
        _dict['fbits'] = _dict['fbits'] - 64
        _dict['np_type'] = AiBufferFormat.to_np_type(fmt)
        return _dict

    @staticmethod
    def to_np_type(fmt):
        """Return numpy type based on AI buffer format definition"""
        # noqa: DAR101,DAR201,DAR401

        _type = (fmt >> AiBufferFormat.TYPE_ID_SHIFT) & AiBufferFormat.TYPE_ID_MASK
        _bits = (fmt >> AiBufferFormat.BITS_SHIFT) & AiBufferFormat.BITS_MASK
        _sign = (fmt >> AiBufferFormat.SIGN_SHIFT) & AiBufferFormat.SIGN_MASK

        if _type == AiBufferFormat.TYPE_FLOAT:
            if _bits == 32:
                return np.float32
            if _bits == 16:
                return np.float16
            return np.float64

        if _type == AiBufferFormat.TYPE_NONE:
            return np.void

        if _type == AiBufferFormat.TYPE_BOOL:
            return np.bool

        if _type == AiBufferFormat.TYPE_Q:
            if _sign and _bits == 8:
                return np.int8
            if _sign and _bits == 16:
                return np.int16
            if _sign and _bits == 32:
                return np.int32
            if _sign and _bits == 64:
                return np.int64
            if _bits == 8:
                return np.uint8
            if _bits == 16:
                return np.uint16
            if _bits == 32:
                return np.uint32
            if _bits == 64:
                return np.uint64

        msg_ = 'AI type ({}/{}/{}) to numpy type not supported'.format(_type, _bits, _sign)
        raise NotImplementedError(msg_)

    @staticmethod
    def to_fmt(np_type, is_io=True, static=False, const=False):
        """Convert numpy dtype to AiBufferFormat"""
        # noqa: DAR101,DAR201,DAR401

        np_to_ai_format = {
            np.void: AiBufferFormat.AI_BUFFER_FORMAT_NONE,
            np.bool: AiBufferFormat.AI_BUFFER_FORMAT_BOOL,
            np.float32: AiBufferFormat.AI_BUFFER_FORMAT_FLOAT,
            np.uint8: AiBufferFormat.AI_BUFFER_FORMAT_U8,
            np.uint16: AiBufferFormat.AI_BUFFER_FORMAT_U16,
            np.uint32: AiBufferFormat.AI_BUFFER_FORMAT_U32,
            np.uint64: AiBufferFormat.AI_BUFFER_FORMAT_U64,
            np.int8: AiBufferFormat.AI_BUFFER_FORMAT_S8,
            np.int16: AiBufferFormat.AI_BUFFER_FORMAT_S16,
            np.int32: AiBufferFormat.AI_BUFFER_FORMAT_S32,
            np.int64: AiBufferFormat.AI_BUFFER_FORMAT_S64,
        }

        fmt = np_to_ai_format.get(np_type, None)
        if fmt is None:
            msg_ = 'numpy type {} is not supported'.format(str(np_type))
            raise NotImplementedError(msg_)

        if is_io:
            fmt |= AiBufferFormat.FLAG_IS_IO
        if static:
            fmt |= AiBufferFormat.FLAG_STATIC
        if const:
            fmt |= AiBufferFormat.FLAG_CONST

        return fmt


def stm_tflm_node_type_to_str(op_type_id, with_id=True):
    """  # noqa: DAR005
    Return a human readable description of a TFLM-node

    see <TF>::Tensorflow git (tensorflow/lite/schema/schema_generated.h file)

    Parameters
    ----------
    op_type_id
        TFLM operator id/index
    with_id:
        add numerical value in the description

    Returns
    -------
    str
        human readable description of the c-node
    """

    if 0 <= op_type_id < len(_TFLM_OPERATORS):
        desc_ = '{}'.format(_TFLM_OPERATORS[op_type_id])
    else:
        desc_ = 'custom or unknown'
        with_id = True
    if with_id:
        desc_ += ' ({})'.format(op_type_id)
    return desc_


_TFLM_OPERATORS = [
    "ADD",
    "AVERAGE_POOL_2D",
    "CONCATENATION",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "DEPTH_TO_SPACE",
    "DEQUANTIZE",
    "EMBEDDING_LOOKUP",
    "FLOOR",
    "FULLY_CONNECTED",
    "HASHTABLE_LOOKUP",
    "L2_NORMALIZATION",
    "L2_POOL_2D",
    "LOCAL_RESPONSE_NORMALIZATION",
    "LOGISTIC",
    "LSH_PROJECTION",
    "LSTM",
    "MAX_POOL_2D",
    "MUL",
    "RELU",
    "RELU_N1_TO_1",
    "RELU6",
    "RESHAPE",
    "RESIZE_BILINEAR",
    "RNN",
    "SOFTMAX",
    "SPACE_TO_DEPTH",
    "SVDF",
    "TANH",
    "CONCAT_EMBEDDINGS",
    "SKIP_GRAM",
    "CALL",
    "CUSTOM",
    "EMBEDDING_LOOKUP_SPARSE",
    "PAD",
    "UNIDIRECTIONAL_SEQUENCE_RNN",
    "GATHER",
    "BATCH_TO_SPACE_ND",
    "SPACE_TO_BATCH_ND",
    "TRANSPOSE",
    "MEAN",
    "SUB",
    "DIV",
    "SQUEEZE",
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "STRIDED_SLICE",
    "BIDIRECTIONAL_SEQUENCE_RNN",
    "EXP",
    "TOPK_V2",
    "SPLIT",
    "LOG_SOFTMAX",
    "DELEGATE",
    "BIDIRECTIONAL_SEQUENCE_LSTM",
    "CAST",
    "PRELU",
    "MAXIMUM",
    "ARG_MAX",
    "MINIMUM",
    "LESS",
    "NEG",
    "PADV2",
    "GREATER",
    "GREATER_EQUAL",
    "LESS_EQUAL",
    "SELECT",
    "SLICE",
    "SIN",
    "TRANSPOSE_CONV",
    "SPARSE_TO_DENSE",
    "TILE",
    "EXPAND_DIMS",
    "EQUAL",
    "NOT_EQUAL",
    "LOG",
    "SUM",
    "SQRT",
    "RSQRT",
    "SHAPE",
    "POW",
    "ARG_MIN",
    "FAKE_QUANT",
    "REDUCE_PROD",
    "REDUCE_MAX",
    "PACK",
    "LOGICAL_OR",
    "ONE_HOT",
    "LOGICAL_AND",
    "LOGICAL_NOT",
    "UNPACK",
    "REDUCE_MIN",
    "FLOOR_DIV",
    "REDUCE_ANY",
    "SQUARE",
    "ZEROS_LIKE",
    "FILL",
    "FLOOR_MOD",
    "RANGE",
    "RESIZE_NEAREST_NEIGHBOR",
    "LEAKY_RELU",
    "SQUARED_DIFFERENCE",
    "MIRROR_PAD",
    "ABS",
    "SPLIT_V",
    "UNIQUE",
    "CEIL",
    "REVERSE_V2",
    "ADD_N",
    "GATHER_ND",
    "COS",
    "WHERE",
    "RANK",
    "ELU",
    "REVERSE_SEQUENCE",
    "MATRIX_DIAG",
    "QUANTIZE",
    "MATRIX_SET_DIAG",
    "ROUND",
    "HARD_SWISH",
    "IF",
    "WHILE",
    "NON_MAX_SUPPRESSION_V4",
    "NON_MAX_SUPPRESSION_V5",
    "SCATTER_ND",
    "SELECT_V2",
    "DENSIFY",
    "SEGMENT_SUM",
    "BATCH_MATMUL",
    "PLACEHOLDER_FOR_GREATER_OP_CODES",
    "CUMSUM",
    "CALL_ONCE",
    "BROADCAST_TO",
    "RFFT2D",
    "CONV_3D",
    "IMAG",
    "REAL",
    "COMPLEX_ABS",
    "HASHTABLE",
    "HASHTABLE_FIND",
    "HASHTABLE_IMPORT",
    "HASHTABLE_SIZE"
]
