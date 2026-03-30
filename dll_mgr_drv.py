###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
Driver to manage the DLL (generated for the X86 validation)
"""

import ctypes as ct
import os
import fnmatch
import time as t
import platform
import tempfile
import shutil
import random
import string

import numpy as np

from .ai_runner import AiRunnerDriver, AiRunner
from .ai_runner import HwIOError, AiRunnerError
from .stm_ai_utils import stm_ai_error_to_str, AiBufferFormat
from .stm_ai_utils import stm_ai_node_type_to_str, RT_STM_AI_NAME

_AI_MAGIC_MARKER = 0xA1FACADE

_SHARED_LIB_EXT = ['.so', '.dll', '.dylib']


def _find_files(directory, pattern):
    for root, _, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def get_library_name(path_name):
    """
    Return full name of the shared lib, including the extension

    Parameters
    ----------
    path_name
        full path of the shared library file

    Returns
    -------
    str
        full path with specific extension

    Raises
    ------
    OSError
        File not found
    """
    if not os.path.isfile(path_name):
        for ext in _SHARED_LIB_EXT:
            if os.path.isfile(path_name + ext):
                return path_name + ext
    else:
        return path_name
    raise OSError('"{}" not found'.format(path_name))


class _TemporaryLibrary:
    """Utility class - Temporary copy of the library"""

    def __init__(self, full_name, copy_mode=False):
        self._c_dll = None  # Library object
        self._dir = os.path.dirname(full_name)
        self._name = os.path.basename(full_name)
        if copy_mode:
            name_, ext_ = os.path.splitext(os.path.basename(full_name))
            name_ += '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            self._name = name_ + ext_
            self._dir = tempfile.gettempdir()
            self._tmp_full_name = os.path.join(self._dir, self._name)
            shutil.copyfile(full_name, self._tmp_full_name)
        else:
            self._tmp_full_name = None

    @property
    def full_path(self):
        """Return full path of the used shared library"""
        # noqa: DAR101,DAR201,DAR401
        if self._tmp_full_name:
            return self._tmp_full_name
        else:
            return os.path.join(self._dir, self._name)

    def load_library(self):
        """Load the library"""  # noqa: DAR101,DAR201,DAR401
        self._c_dll = np.ctypeslib.load_library(self._name, self._dir)
        return self._c_dll

    def __call__(self):
        """Callable method"""  # noqa: DAR201
        return self._c_dll

    def unload_library(self):
        """Unload the library"""
        if self._c_dll is not None:
            if self._tmp_full_name:
                hdl_ = self._c_dll._handle  # pylint: disable=protected-access
                if os.name == 'nt':
                    ct.windll.kernel32.FreeLibrary(ct.c_void_p(hdl_))
                self._c_dll = None
            else:
                self._c_dll = None
        if self._tmp_full_name and os.path.isfile(self._tmp_full_name):
            os.remove(self._tmp_full_name)
            self._tmp_full_name = None

    def __del__(self):
        """Destructor"""
        self.unload_library()


def _split(full_name):
    name_ = os.path.basename(full_name)
    name_, ext_ = os.path.splitext(name_)
    norm_path_ = os.path.normpath(full_name)
    dir_ = os.path.dirname(norm_path_)
    return norm_path_, dir_, name_, ext_


def check_and_find_stm_ai_dll(desc):
    """Check and return stm.ai dll(s)"""  # noqa: DAR101,DAR201,DAR401
    cdt, cdt_obs = '', ''
    invalid_keys = ['libai_inspector.', 'libai_observer.',
                    'libai_helper_convert.']

    if os.path.isdir(desc):
        for f_name in _find_files(desc, 'libai_*.*'):
            if all(x not in f_name for x in invalid_keys) and\
                    not f_name.endswith('.a'):
                cdt = os.path.normpath(f_name)
                break
    elif os.path.isfile(desc):
        norm_path_, _, _, ext_ = _split(desc)
        if ext_ in _SHARED_LIB_EXT:
            cdt = norm_path_
    else:
        norm_path_, _, _, ext_ = _split(desc)
        if not ext_:
            for ext_ in _SHARED_LIB_EXT:
                f_lib = norm_path_ + ext_
                if os.path.isfile(f_lib):
                    cdt = f_lib
                    break

    if cdt:
        # check if not an excluded file
        if not all(x not in cdt for x in invalid_keys):
            cdt, cdt_obs = '', ''
        # check if expected dll observer is available
        _, dir_, _, ext_ = _split(cdt)
        f_obs = os.path.join(dir_, "libai_observer" + ext_)
        if os.path.isfile(f_obs):
            cdt_obs = f_obs

    return bool(cdt), cdt, cdt_obs


def _wrap_api(lib, funcname, restype, argtypes):
    """Helper function to simplify wrapping CTypes functions."""  # noqa: DAR101,DAR201,DAR401
    func = getattr(lib, funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func


def _rounded_up(val, align):
    """Rounding up intger"""  # noqa: DAR101,DAR201,DAR401
    return -(-val // align)


class Printable(ct.Structure):
    """C struct interface for printable field contents; used for debug and
    logging."""  # noqa: DAR101,DAR201,DAR401

    def __str__(self):
        """Print the content of the structure."""  # noqa: DAR101,DAR201,DAR401
        msg = "[{}]".format(self.__class__.__name__)

        for field in self._fields_:
            member = getattr(self, field[0])
            # parse pointer addresses
            try:
                if hasattr(member, 'contents'):
                    member = hex(ct.addressof(member.contents))
            except ValueError:  # NULL pointer
                member = "NULL"

            indented = "\n".join(
                "  " + line for line in str(member).splitlines())
            msg += "\n  {}:{}".format(field[0], indented)

        return msg + "\n"


class AiPlatformVersion(Printable):
    """Wrapper for C 'ai_platform_version' struct."""  # noqa: DAR101,DAR201,DAR401
    _fields_ = [('major', ct.c_uint, 8),
                ('minor', ct.c_uint, 8),
                ('micro', ct.c_uint, 8),
                ('reserved', ct.c_uint, 8)]

    def to_ver(self):
        """Return 3d tuple with the version"""  # noqa: DAR101,DAR201,DAR401
        return (self.major, self.minor, self.micro)


class AiIntqInfo(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_intq_info' struct."""  # noqa: DAR101,DAR201,DAR401
    _fields_ = [('scale', ct.POINTER(ct.c_float)),
                ('zeropoint', ct.POINTER(ct.c_void_p))]


class AiIntqInfoList(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_intq_info_list' struct."""  # noqa: DAR101,DAR201,DAR401
    _fields_ = [('flags', ct.c_uint, 16),
                ('size', ct.c_uint, 16),
                ('info', ct.POINTER(AiIntqInfo))]
    AI_BUFFER_META_FLAG_SCALE_FLOAT = 0x00000001
    AI_BUFFER_META_FLAG_ZEROPOINT_U8 = 0x00000002
    AI_BUFFER_META_FLAG_ZEROPOINT_S8 = 0x00000004


class AiBufferMetaInfo(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_buffer_meta_info' struct."""  # noqa: DAR101,DAR201,DAR401
    _fields_ = [('flags', ct.c_uint, 32),
                ('intq_info_list', ct.POINTER(AiIntqInfoList))]

    """ Meta Info Flags """
    AI_BUFFER_META_HAS_INTQ_INFO = 0x00000001


class AiBuffer(Printable):
    """Wrapper for C 'ai_buffer' struct."""
    _fields_ = [('format', ct.c_uint, 32),
                ('n_batches', ct.c_uint, 16),
                ('height', ct.c_uint, 16),
                ('width', ct.c_uint, 16),
                ('channels', ct.c_uint, 32),
                ('data', ct.POINTER(ct.c_uint8)),  # ct.c_void_p),
                ('meta_info', ct.POINTER(AiBufferMetaInfo))]

    @classmethod
    def from_ndarray(cls, ndarray, fmt=None):
        """Wrap a Numpy array's internal buffer to an AiBuffer without copy."""
        # noqa: DAR101,DAR201,DAR401

        #
        # Only arrays with 4 dimensions or less can be mapped because AiBuffer
        # encodes only 4 dimensions; the Numpy dtype is mapped to the
        # corresponding AiBuffer type.
        #
        if len(ndarray.shape) > 4:
            raise ValueError(
                "Arrays with more than 4 dimensions are not supported")

        ndshape = ndarray.shape[:-1] + (1, ) * (4 - len(ndarray.shape)) + \
            (ndarray.shape[-1], )
        if fmt is None:
            fmt = AiBufferFormat.to_fmt(ndarray.dtype.type)

        return cls(fmt, *ndshape, data=ct.cast(ndarray.ctypes.data, ct.POINTER(ct.c_uint8)))

    @classmethod
    def from_bytes(cls, bytebuf):
        """Wrap a Python bytearray object to AiBuffer."""
        # noqa: DAR101,DAR201,DAR401
        if not isinstance(bytebuf, bytearray):
            raise RuntimeError("Buffer is not a stream of bytes.")

        data = (ct.POINTER(ct.c_byte * len(bytebuf))).from_buffer(bytebuf)
        return AiBuffer(AiBufferFormat.AI_BUFFER_FORMAT_U8, 1, 1, 1,
                        len(bytebuf), ct.c_void_p(ct.addressof(data)))

    def to_ndarray(self):
        """Wrap the AiBuffer internal buffer into a Numpy array object."""  # noqa: DAR101,DAR201,DAR401

        #
        # # The AiBuffer type is mapped to the corresponding Numpy's dtype.
        # The memory is managed by the code which instantiated the AiBuffer;
        # create a copy with 'ndarray.copy()' if the buffer was instantiated by
        # the C library before unloading the library.
        #
        dtype = AiBufferFormat.to_np_type(self.format)
        shape = (self.n_batches, self.height, self.width, self.channels)
        size = np.dtype(dtype).itemsize * int(np.prod(shape))

        # WARNING: the memory is owned by the C library
        # don't let it go out of scope before copying the array content
        data = ct.cast(self.data, ct.POINTER(ct.c_byte * size))
        bytebuf = np.ctypeslib.as_array(data.contents, (size, ))

        return np.reshape(np.frombuffer(bytebuf, dtype), shape)


class AiBufferArray(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_buffer_array' struct."""
    _fields_ = [('flags', ct.c_uint, 16),
                ('size', ct.c_uint, 16),
                ('buffer', ct.POINTER(AiBuffer))]


class AiIoTensor(Printable):
    """Wrapper for C 'ai_io_tensor' struct"""  # noqa: DAR101,DAR201,DAR401
    _fields_ = [
        ('format', ct.c_uint, 32),
        ('n_batches', ct.c_uint, 16),
        ('height', ct.c_uint, 16),
        ('width', ct.c_uint, 16),
        ('channels', ct.c_uint, 32),
        ('data', ct.POINTER(ct.c_uint8)),
        ('scale', ct.c_float),
        ('zeropoint', ct.c_int32),
    ]

    def to_ndarray(self):
        """Wrap the AiBuffer internal buffer into a Numpy array object."""  # noqa: DAR101,DAR201,DAR401

        #
        # The AiBuffer type is mapped to the corresponding Numpy's dtype.
        # The memory is managed by the code which instantiated the AiBuffer;
        # create a copy with 'ndarray.copy()' if the buffer was instantiated by
        # the C library before unloading the library.
        #
        dtype = AiBufferFormat.to_np_type(self.format)
        shape = (self.n_batches, self.height, self.width, self.channels)
        size = np.dtype(dtype).itemsize * int(np.prod(shape))

        # WARNING: the memory is owned by the C library
        # don't let it go out of scope before copying the array content
        data = ct.cast(self.data, ct.POINTER(ct.c_byte * size))
        bytebuf = np.ctypeslib.as_array(data.contents, (size, ))

        return np.reshape(np.frombuffer(bytebuf, dtype), shape)


class AiObserverIoNode(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_observer_node' struct"""  # noqa: DAR101,DAR201,DAR401
    _fields_ = [
        ('c_idx', ct.c_uint16),
        ('type', ct.c_uint16),
        ('id', ct.c_uint16),
        ('n_tensors', ct.c_uint16),
        ('elapsed_ms', ct.c_float),
        ('tensors', ct.POINTER(AiIoTensor)),
    ]
    AI_OBSERVER_NONE_EVT = 0
    AI_OBSERVER_INIT_EVT = (1 << 0)
    AI_OBSERVER_PRE_EVT = (1 << 1)
    AI_OBSERVER_POST_EVT = (1 << 2)
    AI_OBSERVER_FIRST_EVT = (1 << 8)
    AI_OBSERVER_LAST_EVT = (1 << 9)


class AiError(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_error' struct."""  # noqa: DAR101,DAR201,DAR401
    _fields_ = [
        ('type', ct.c_uint, 8),
        ('code', ct.c_uint, 24)
    ]

    def __str__(self):
        """Return human description of the error"""  # noqa: DAR101,DAR201,DAR401
        return 'type={} code={}'.format(self.type, self.code)


class AiNetworkBuffers(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_network_buffers' struct."""
    _fields_ = [('map_signature', ct.c_uint, 32),
                ('map_weights', AiBufferArray),
                ('map_activations', AiBufferArray)]

    MAP_SIGNATURE = 0xA1FACADE


class AiNetworkParams(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_network_params' struct."""  # noqa: DAR101,DAR201,DAR401
    _fields_ = [
        ('params', AiBuffer),
        ('activations', AiBuffer)
    ]


class AiNetworkParamsUnion(ct.Union):   # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_network_params' struct."""
    _fields_ = [('params', AiNetworkParams),
                ('buffers', AiNetworkBuffers)]

    def _valid_signature(self):
        # noqa: DAR201
        """Check signature of the union datastruct to determine struct semantic"""
        return self.buffers.map_signature == AiNetworkBuffers.MAP_SIGNATURE

    def get_buffers(self):
        # noqa: DAR201
        """Return info about params as AIBufferArray objects"""
        if self._valid_signature():
            params = self.buffers.map_weights
            activations = self.buffers.map_activations
        else:
            params = AiBufferArray(0x0, 1, ct.pointer(self.params.params))
            activations = AiBufferArray(0x0, 1, ct.pointer(self.params.activations))

        assert isinstance(params, AiBufferArray) and isinstance(activations, AiBufferArray)
        return params, activations

    def get_params(self):
        # noqa: DAR201
        """Return info about params as AIBuffer objects"""
        if self._valid_signature():
            params = self.buffers.map_weights.buffer[0]
            activations = self.buffers.map_activations.buffer[0]
        else:
            params = self.params.params
            activations = self.params.activations

        assert isinstance(params, AiBuffer) and isinstance(activations, AiBuffer)

        return params, activations


class AiNetworkReport(Printable):  # pylint: disable=too-few-public-methods
    """Wrapper for C 'ai_network_report' struct."""
    _fields_ = [
        ('model_name', ct.c_char_p),
        ('model_signature', ct.c_char_p),
        ('model_datetime', ct.c_char_p),
        ('compile_datetime', ct.c_char_p),
        ('runtime_revision', ct.c_char_p),
        ('runtime_version', AiPlatformVersion),
        ('tool_revision', ct.c_char_p),
        ('tool_version', AiPlatformVersion),
        ('tool_api_version', AiPlatformVersion),
        ('api_version', AiPlatformVersion),
        ('interface_api_version', AiPlatformVersion),
        ('n_macc', ct.c_uint, 32),
        ('n_inputs', ct.c_uint, 16),
        ('n_outputs', ct.c_uint, 16),
        ('inputs', ct.POINTER(AiBuffer)),
        ('outputs', ct.POINTER(AiBuffer)),
        ('buffers', AiNetworkParamsUnion),
        ('n_nodes', ct.c_uint, 32),
        ('signature', ct.c_uint, 32)
    ]


class AiDllBackend:
    """
    Backend ensuring the wrapping of the shared lib functions to Python functions
    """
    def __init__(self, libname, dir_path, logger, name='network', reload=False):
        """Constructor"""  # noqa: DAR101,DAR201,DAR401

        self._shared_lib = None  # library object when loaded
        self._shared_obs_lib = None
        self._logger = logger
        self._tmp_lib = None

        path_name = get_library_name(os.path.join(dir_path, libname))

        self._tmp_lib = _TemporaryLibrary(path_name, copy_mode=reload)
        self._logger.debug('loaded shared library: {} ({})'.format(
            self._tmp_lib.full_path, bool(reload)))

        self._shared_lib = self._tmp_lib.load_library()

        try:
            self._shared_obs_lib = np.ctypeslib.load_library('libai_observer', dir_path)
        except OSError:
            self._shared_obs_lib = None

        # Wrapper for 'ai_network_create' C API
        while True:
            try:
                self.ai_network_create = _wrap_api(
                    self._shared_lib, f'ai_{name}_create',
                    AiError, [ct.POINTER(ct.c_void_p), ct.POINTER(AiBuffer)])
            except AttributeError as ex_:
                if name == 'network':
                    # try to use the name used to generate the name of the file
                    _, _, name_, _ = _split(libname)
                    name = name_.replace('libai_', '')
                else:
                    raise ex_
            else:
                break

        # Wrapper for 'ai_network_get_error' C API
        self.ai_network_get_error = _wrap_api(
            self._shared_lib, f'ai_{name}_get_error',
            AiError, [ct.c_void_p])

        # Wrapper for 'ai_network_init' C API
        self.ai_network_init = _wrap_api(
            self._shared_lib, f'ai_{name}_init',
            ct.c_bool, [ct.c_void_p, ct.POINTER(AiNetworkParamsUnion)])

        # Wrapper for 'ai_network_destroy' C API
        self.ai_network_destroy = _wrap_api(
            self._shared_lib, f'ai_{name}_destroy', ct.c_void_p, [ct.c_void_p])

        # Wrapper for 'ai_network_get_info' C API
        self.ai_network_get_info = _wrap_api(
            self._shared_lib, f'ai_{name}_get_info',
            ct.c_bool, [ct.c_void_p, ct.POINTER(AiNetworkReport)])

        # Wrapper for 'ai_network_get_report' C API (NEW)
        try:
            self.ai_network_get_report = _wrap_api(
                self._shared_lib, f'ai_{name}_get_report',
                ct.c_bool, [ct.c_void_p, ct.POINTER(AiNetworkReport)])
        except AttributeError:
            self._logger.debug('ai_network_get_report() not available')

        # Wrapper for 'ai_network_run' C API
        self.ai_network_run = _wrap_api(
            self._shared_lib,
            f'ai_{name}_run', ct.c_int,
            [ct.c_void_p,
             ct.POINTER(AiBuffer),
             ct.POINTER(AiBuffer)])

        # Wrapper for 'ai_network_data_weights_get' C API
        self.ai_network_data_weights_get = _wrap_api(
            self._shared_lib, f'ai_{name}_data_weights_get',
            ct.c_void_p, [])

        # Wrapper for 'ai_network_data_params_get' C API (NEW)
        try:
            self.ai_network_data_params_get = _wrap_api(
                self._shared_lib, f'ai_{name}_data_params_get',
                ct.c_bool, [ct.c_void_p, ct.POINTER(AiNetworkParamsUnion)])
        except AttributeError:
            self._logger.debug('ai_network_data_params_get() not available')

        # Wrapper for 'ai_observer_io_node_cb' C API
        self.ai_observer_io_node_cb = ct.CFUNCTYPE(
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(AiObserverIoNode)
        )

        if self._shared_obs_lib is None:
            return

        # Wrapper for 'ai_platform_observer_io_register' C API
        self.ai_platform_observer_io_register = _wrap_api(
            self._shared_obs_lib,
            'ai_platform_observer_io_register',
            ct.c_bool,
            [ct.c_void_p,
             self.ai_observer_io_node_cb,
             ct.c_uint32])

        # Wrapper for 'ai_platform_observer_io_unregister' C API
        self.ai_platform_observer_io_unregister = _wrap_api(
            self._shared_obs_lib,
            'ai_platform_observer_io_unregister',
            ct.c_bool,
            [ct.c_void_p,
             self.ai_observer_io_node_cb])

    def capabilities(self):
        """Return list with the capabilities"""  # noqa: DAR101,DAR201,DAR401
        if self._shared_obs_lib is not None:
            return [AiRunner.Caps.IO_ONLY, AiRunner.Caps.PER_LAYER,
                    AiRunner.Caps.PER_LAYER_WITH_DATA]
        else:
            return [AiRunner.Caps.IO_ONLY]

    def release(self):
        """Release, upload the shared libraries"""  # noqa: DAR101,DAR201,DAR401
        if self._tmp_lib is not None:
            self._tmp_lib.unload_library()
            self._tmp_lib = None
            self._shared_lib = None
        if self._shared_obs_lib is not None:
            del self._shared_obs_lib
            self._shared_obs_lib = None

    def __del__(self):
        self.release()


class AiDllDriver(AiRunnerDriver):
    """Class to handle the DLL"""

    def __init__(self, parent):
        """Constructor"""  # noqa: DAR101,DAR201,DAR401
        self._backend = None
        self._handle = None
        self._dll_name = None
        self._activations = None
        self._io_from_act = True
        self._info = None  # cache the description of the models
        self._s_dur = 0.0
        self._profiler = None
        self._callback = None
        self._weights_ptr_map = None
        self._buffers_data = None
        self._mode = AiRunner.Mode.IO_ONLY
        self.ghosted_observer_io_node_cb = None
        super(AiDllDriver, self).__init__(parent)

    @staticmethod
    def is_valid(file_path):
        """Check if the provided path is a valid path"""
        # noqa: DAR101,DAR201,DAR401
        if isinstance(file_path, str):
            res, dll_path, _ = check_and_find_stm_ai_dll(file_path)
            return res, dll_path
        return False, ''

    @property
    def is_connected(self):
        """Indicate if the DLL is loaded"""
        # noqa: DAR101,DAR201,DAR401
        return bool(self._backend)

    @property
    def capabilities(self):
        """Return list with the capabilities"""
        # noqa: DAR101,DAR201,DAR401
        if self.is_connected:
            return self._backend.capabilities()
        return []

    def _set_c_buffer_with_remap_table(self, buffers_data):
        """Create and set a remap table for multiple buffers"""  # noqa: DAR101,DAR201,DAR401

        size, count = sum([len(b) for b in buffers_data]), len(buffers_data) + 2

        # Create list of c-type object with the provided buffers
        #  note: minimum 8-bytes size is requested to map a bytearray
        buffers = []
        for i, buffer in enumerate(buffers_data):
            if len(buffer) < 8:  # minimum size should be 8 for ctype map
                buffers_data[i] = buffer + bytearray(8 - len(buffer))
            buffers.append((ct.POINTER(ct.c_uint8 * len(buffers_data[i]))).from_buffer(buffers_data[i]))

        if len(buffers_data) == 1:
            # As a simple buffer(binary format) w/o remap entries
            return ct.cast(ct.addressof(buffers[0]), ct.POINTER(ct.c_uint8)), None

        msg = f' Building the c-remap table, nb_entry=1+{count - 2}+1 ({size} bytes)'
        self._logger.debug(msg)

        # Create the magic marker
        magic_marker = ct.cast(_AI_MAGIC_MARKER, ct.POINTER(ct.c_uint8))

        # Build and init the remap table
        #  note: the params_ptr_map object is returned to be owned (ref_count>0) by the caller
        #        before to use it by the DLL.
        buffers_ptr = [magic_marker]
        buffers_ptr += [ct.cast(ct.addressof(b), ct.POINTER(ct.c_uint8)) for b in buffers]
        buffers_ptr += [magic_marker]

        params_ptr_map = (ct.POINTER(ct.c_uint8) * count)(*buffers_ptr)

        return (ct.cast(ct.addressof(params_ptr_map), ct.POINTER(ct.c_uint8)),
                params_ptr_map)

    def _set_params_data(self, params, weights):
        """Set weights buffer"""  # noqa: DAR101,DAR201,DAR401
        expected_size = params.params.channels

        # Buffer from the DLL is used first
        addr_w = self._backend.ai_network_data_weights_get()
        if addr_w and expected_size:
            self._logger.debug(' Weights buffer from the DLL is used')
            params.params.data = ct.cast(addr_w, ct.POINTER(ct.c_uint8))
            return
        if addr_w is None and not expected_size:
            self._logger.debug(' Weights buffer is empty')
            return

        if expected_size and weights is None:
            msg = f'No weights are available ({expected_size} bytes expected)'
            raise HwIOError(msg)

        # Use the data from a binary file
        if isinstance(weights, str) and weights.endswith('.bin') and os.path.isfile(weights):
            self._logger.debug(' loading weights from binary file: {}'.format(weights))
            with open(weights, 'rb') as bin_file:
                self._buffers_data = [bytearray(bin_file.read())]
        # Use the data from a simple bytearray object
        elif isinstance(weights, bytearray):
            self._logger.debug(' loading weights from bytearray object')
            self._buffers_data = [weights]
        # Use the data from a list of bytearray objects
        elif isinstance(weights, list) and all([isinstance(b, bytearray) for b in weights]):
            self._logger.debug(' loading weights from bytearray list (len={})'.format(len(weights)))
            self._buffers_data = weights
        # Use the data from the X-CUBE-AI code generator v6+
        elif isinstance(weights, list) and all(['buffer_data' in b for b in weights]):
            self._logger.debug(' loading weights from STM.AI object (len={})'.format(len(weights)))
            self._buffers_data = [w['buffer_data'] for w in weights]
            assert all([isinstance(b, bytearray) for b in self._buffers_data])
        else:
            raise HwIOError('Invalid weights object: binary file, bytearray,.. are expected')

        weights_size = sum([len(b) for b in self._buffers_data])
        if weights_size != expected_size:
            msg = f'Provided weights size are incompatible (got: {weights_size} expected: {expected_size})'
            raise HwIOError(msg)

        self._logger.debug(f'Loaded Network Weights: loaded {weights_size} Bytes.')
        params.params.data, self._weights_ptr_map = self._set_c_buffer_with_remap_table(self._buffers_data)

    def _set_activations_data(self, params):
        """Allocate and set activations buffer"""  # noqa: DAR101,DAR201,DAR401

        align = 8  # buffer is 8-bytes aligned
        shape = (1, 1, 1, _rounded_up(params.activations.channels, align))
        activations = np.zeros(shape, dtype=np.uint64)
        # activations = np.empty(shape, dtype=np.uint64)

        self._logger.debug(' allocating {} bytes for activations buffer'.format(shape[3] * align))

        self._activations = np.require(activations, dtype=np.uint64,
                                       requirements=['C', 'O', 'W', 'A'])

        params.activations.data = ct.cast(self._activations.ctypes.data,
                                          ct.POINTER(ct.c_uint8))

    def connect(self, desc=None, **kwargs):
        """Connect to the shared library"""  # noqa: DAR101,DAR201,DAR401

        self.disconnect()

        dll_name = kwargs.pop('dll_name', None)
        weights = kwargs.pop('weights', None)
        reload = kwargs.pop('reload', None)

        if reload is None:
            if os.environ.get('AI_RUNNER_FORCE_NO_DLL_COPY', None):
                reload = False
            else:
                reload = True

        if dll_name is None:
            dir_path, dll_name = os.path.split(desc)
        else:
            dir_path = desc

        self._logger.debug('loading {} (from {})'.format(dll_name, dir_path))
        try:
            self._backend = AiDllBackend(dll_name, dir_path, logger=self._logger, reload=reload)
        except Exception as ctx:  # pylint: disable=broad-except
            self._logger.debug('{}'.format(str(ctx)))
            self._backend = None
            raise HwIOError(str(ctx))

        self._dll_name = os.path.join(dir_path, dll_name)

        # Create an instance of the c-network
        self._handle = ct.c_void_p()
        error = self._backend.ai_network_create(ct.pointer(self._handle), None)
        if error.code:
            msg = 'Unable to create the instance: {}'.format(error)
            raise HwIOError(msg)
        self._logger.debug(' creating instance: {}'.format(error))

        # get net info a first time to get activation and parameter sizes
        self._info = AiNetworkReport()
        res = self._backend.ai_network_get_info(self._handle, ct.pointer(self._info))
        if not res:
            raise HwIOError('ai_network_get_info() failed')

        params = AiNetworkParams(self._info.buffers.params.params, self._info.buffers.params.activations)

        self._logger.debug(' activations buffer: {}'.format(params.activations.channels))
        self._logger.debug(' weights buffer: {}'.format(params.params.channels))

        # set weights buffer
        self._set_params_data(params, weights)

        # set activations buffer
        self._set_activations_data(params)

        # Initialize the c-network
        params_union = AiNetworkParamsUnion(params)
        self._backend.ai_network_init(self._handle, ct.pointer(params_union))
        error = self._backend.ai_network_get_error(self._handle)
        if error.code:
            msg = 'ai_network_init() failed\n AiError - {}'.format(stm_ai_error_to_str(error.code, error.type))
            raise HwIOError(msg)
        self._logger.debug(' initializing instance: {}'.format(error))

        # get the updated net info
        self._backend.ai_network_get_info(self._handle, ct.pointer(self._info))

        if AiRunner.Caps.PER_LAYER not in self.capabilities:
            return True

        @ct.CFUNCTYPE(ct.c_uint32, ct.c_uint32, ct.POINTER(AiObserverIoNode))
        def ghosted_observer_io_node_cb(flags, node):
            if flags & AiObserverIoNode.AI_OBSERVER_PRE_EVT:
                self._io_node_pre_evt(flags, node)
            if flags & AiObserverIoNode.AI_OBSERVER_POST_EVT:
                self._io_node_post_evt(flags, node)
            return 0

        self.ghosted_observer_io_node_cb = ghosted_observer_io_node_cb

        mode_ = AiObserverIoNode.AI_OBSERVER_PRE_EVT | AiObserverIoNode.AI_OBSERVER_POST_EVT
        self._backend.ai_platform_observer_io_register(self._handle,
                                                       self.ghosted_observer_io_node_cb,
                                                       mode_)

        # at this point the model is ready to be used

        return True

    def _io_node_pre_evt(self, flags, node):
        """Pre-evt IO node call-back"""  # noqa: DAR101,DAR201,DAR401
        if flags & AiObserverIoNode.AI_OBSERVER_FIRST_EVT == AiObserverIoNode.AI_OBSERVER_FIRST_EVT:
            self._s_dur = 0.0
        if self._mode == AiRunner.Mode.IO_ONLY:
            return

        if self._callback:
            node_ = node[0]
            inputs = []
            for idx in range(node_.n_tensors):
                tens = node_.tensors[idx]
                inputs.append(tens.to_ndarray())
            extra_ = {'m_id': node_.id, 'layer_type': node_.type & 0x7FFF}
            self._callback.on_node_begin(node_.c_idx, inputs, logs=extra_)

    def _io_node_post_evt(self, flags, node):
        """Post-evt IO node call-back"""  # noqa: DAR101,DAR201,DAR401
        profiler = self._profiler
        node_ = node[0]
        self._s_dur += node_.elapsed_ms
        if self._mode == AiRunner.Mode.IO_ONLY:
            return

        shapes, features, types, scales, zeropoints = [], [], [], [], []
        for idx in range(node_.n_tensors):
            tens_ = node_.tensors[idx]
            shapes.append((tens_.n_batches, tens_.height, tens_.width, tens_.channels))
            if self._mode == AiRunner.Mode.PER_LAYER_WITH_DATA and (profiler or self._callback):
                feature = tens_.to_ndarray().copy()
            else:
                feature = np.array([], dtype=AiBufferFormat.to_np_type(tens_.format))
            features.append(feature)
            types.append(AiBufferFormat.to_np_type(tens_.format))
            scales.append(tens_.scale)
            zeropoints.append(tens_.zeropoint)

        if profiler:
            if node_.c_idx >= len(profiler['c_nodes']):
                item = {
                    'c_durations': [node_.elapsed_ms],
                    'm_id': node_.id,
                    'layer_type': node_.type & 0x7FFF,
                    'layer_desc': stm_ai_node_type_to_str(node_.type & 0x7FFF),
                    'type': types,
                    'shape': shapes,
                    'scale': scales,
                    'zero_point': zeropoints,
                    'data': features if features is not None else None,
                }
                profiler['c_nodes'].append(item)
            else:
                item = profiler['c_nodes'][node_.c_idx]
                item['c_durations'].append(node_.elapsed_ms)
                for idx, _ in enumerate(features):
                    item['data'][idx] = np.append(item['data'][idx], features[idx], axis=0)

            if flags & AiObserverIoNode.AI_OBSERVER_LAST_EVT == AiObserverIoNode.AI_OBSERVER_LAST_EVT:
                profiler['c_durations'].append(self._s_dur)

        if self._callback:
            extra_ = {
                'm_id': node_.id,
                'layer_type': node_.type & 0x7FFF,
                'shape': shapes,
                'c_duration': node_.elapsed_ms
            }
            self._callback.on_node_end(node_.c_idx, features if features is not None else None, logs=extra_)

    def discover(self, flush=False):
        """Build the list of the available model"""  # noqa: DAR101,DAR201,DAR401
        if self._backend:
            return [self._info.model_name.decode('UTF-8')]
        return []

    def __del__(self):
        self.disconnect()

    def disconnect(self):
        if self._backend:
            self._logger.debug(' releasing backend ({})...'.format(self._backend))
            self._backend.ai_network_destroy(self._handle)
            if self.ghosted_observer_io_node_cb and AiRunner.Caps.PER_LAYER in self.capabilities:
                self._backend.ai_platform_observer_io_unregister(self._handle,
                                                                 self.ghosted_observer_io_node_cb)
            self._handle = None
            self._backend.release()
        self._activations = None
        self._weights_ptr_map = None
        self._buffers_data = None
        self._backend = None

    def short_desc(self):
        """Return human readable description"""  # noqa: DAR101,DAR201,DAR401
        io_ = self._dll_name
        return 'X86 shared lib' + ' (' + io_ + ')'

    def _get_runtime(self, model_info):
        """Return a dict with the runtime attributes"""  # noqa: DAR101,DAR201,DAR401
        return {
            'name': RT_STM_AI_NAME,
            'version': model_info.runtime_version.to_ver(),
            'capabilities': self.capabilities,
            'tools_version': model_info.tool_version.to_ver(),
        }

    def _get_scale_zeropoint(self, buffer):
        """Return a dict of the scale/zero_point keys"""  # noqa: DAR101,DAR201,DAR401
        if buffer.meta_info:
            flags = buffer.meta_info[0].intq_info_list[0].flags
            entry = buffer.meta_info[0].intq_info_list[0].info[0]
            if flags & AiIntqInfoList.AI_BUFFER_META_FLAG_ZEROPOINT_S8:
                zp_ = np.int8(ct.cast(entry.zeropoint, ct.POINTER(ct.c_int8))[0])
            else:
                zp_ = np.uint8(ct.cast(entry.zeropoint, ct.POINTER(ct.c_uint8))[0])
            return {"scale": entry.scale[0], "zero_point": zp_}
        else:
            return {"scale": None, "zero_point": np.uint8(0)}

    def _get_io_tensor(self, buffer, idx, name=None):
        """Return a dict with the tensor IO attributes"""  # noqa: DAR101,DAR201,DAR401
        item = {
            'name': '{}_{}'.format(name if name else 'tensor', idx + 1),
            'shape': (buffer.n_batches, buffer.height, buffer.width, buffer.channels),
            'type': AiBufferFormat.to_np_type(buffer.format),
            'memory_pool': 'in activations buffer' if buffer.data else 'user'
        }
        item.update(self._get_scale_zeropoint(buffer))
        return item

    def _model_to_dict(self, model_info):
        """Return a dict with the network info"""  # noqa: DAR101,DAR201,DAR401
        params = model_info.buffers.params
        return {
            'name': model_info.model_name.decode('UTF-8'),
            'model_datetime': model_info.model_datetime.decode('UTF-8'),  # date of creation
            'compile_datetime': model_info.compile_datetime.decode('UTF-8'),  # date of compilation
            'hash': model_info.model_signature.decode('UTF-8'),
            'n_nodes': model_info.n_nodes,
            'inputs': [self._get_io_tensor(model_info.inputs[d_], d_, 'input')
                       for d_ in range(model_info.n_inputs)],
            'outputs': [self._get_io_tensor(model_info.outputs[d_], d_, 'output')
                        for d_ in range(model_info.n_outputs)],
            'weights': params.params.channels,
            'activations': params.activations.channels,
            'macc': model_info.n_macc,
            'runtime': self._get_runtime(model_info),
            'device': {
                'desc': '{} {} ({})'.format(platform.machine(), platform.processor(), platform.system()),
                'dev_type': 'HOST',
                'dev_id': platform.machine() + ' ' + platform.processor(),
                'system': platform.system(),
                'machine': platform.machine(),
            },
        }

    def get_info(self, c_name=None):
        """Return a dict with the network info of the given c-model"""  # noqa: DAR101,DAR201
        return self._model_to_dict(self._info) if self.is_connected else dict()

    def _prepare_inputs(self, inputs):
        """Check and prepare the inputs for the c-runtime"""  # noqa: DAR101,DAR201,DAR401
        c_inputs = []
        for idx in range(self._info.n_inputs):
            in_info = self._info.inputs[idx]
            shape = (1, in_info.height, in_info.width, in_info.channels)
            dtype = AiBufferFormat.to_np_type(in_info.format)
            if self._info.inputs[idx].data and self._io_from_act:
                # a new ndarray is created with a buffer located inside the activations buffer
                #  item from the user ndarray are copied in
                size = np.dtype(dtype).itemsize * int(np.prod(shape))
                data = ct.cast(self._info.inputs[idx].data, ct.POINTER(ct.c_uint8 * size))
                bytebuf = np.ctypeslib.as_array(data.contents, (size, ))
                dst_ = np.reshape(np.frombuffer(bytebuf, dtype), shape)
                np.copyto(dst_, inputs[idx], casting='no')
                c_inputs.append(dst_)
            else:
                c_inputs.append(inputs[idx])
        return c_inputs

    def _prepare_outputs(self):
        """Prepare the outputs"""  # noqa: DAR101,DAR201,DAR401
        c_outputs = []
        for idx in range(self._info.n_outputs):
            out_info = self._info.outputs[idx]
            shape = (1, out_info.height, out_info.width, out_info.channels)
            dtype = AiBufferFormat.to_np_type(out_info.format)
            if self._info.outputs[idx].data and self._io_from_act:
                # ndarray is created with a buffer located inside the activations buffer
                size = np.dtype(dtype).itemsize * int(np.prod(shape))
                data = ct.cast(self._info.outputs[idx].data, ct.POINTER(ct.c_uint8 * size))
                bytebuf = np.ctypeslib.as_array(data.contents, (size, ))
                out_ = np.reshape(np.frombuffer(bytebuf, dtype), shape)
                c_outputs.append(out_)
            else:
                out = np.zeros(shape, dtype=dtype)
                c_outputs.append(np.require(out, dtype=dtype,
                                            requirements=['C', 'O', 'W', 'A']))
        return c_outputs

    def _post_outputs(self, c_outputs):
        """Finalize the outputs"""  # noqa: DAR101,DAR201,DAR401
        outputs = []
        for idx in range(self._info.n_outputs):
            if self._info.outputs[idx].data and self._io_from_act:
                outputs.append(c_outputs[idx].copy())
            else:
                outputs.append(c_outputs[idx])
        return outputs

    def _get_io_buffers(self, inputs, outputs):

        assert self._info.n_inputs == len(inputs) and self._info.n_outputs == len(outputs)

        in_buffers = []
        for idx, arr in enumerate(inputs):
            in_format = self._info.inputs[idx].format
            in_buffers.append(AiBuffer.from_ndarray(arr, in_format))

        in_ = (AiBuffer * len(in_buffers))(*in_buffers)

        out_buffers = []
        for idx, arr in enumerate(outputs):
            out_format = self._info.outputs[idx].format
            out_buffers.append(AiBuffer.from_ndarray(arr, out_format))

        out_ = (AiBuffer * len(out_buffers))(*out_buffers)

        return in_, out_

    def invoke_sample(self, s_inputs, **kwargs):
        """Invoke the c-model with a sample (batch_size = 1)"""  # noqa: DAR101,DAR201,DAR401
        if s_inputs[0].shape[0] != 1:
            raise HwIOError('Should be called with a batch size of 1')

        if kwargs.pop('disable_io_from_act', None) is not None:
            self._io_from_act = False
        else:
            self._io_from_act = True
        self._profiler = kwargs.pop('profiler', None)
        self._callback = kwargs.pop('callback', None)
        self._mode = kwargs.pop('mode', AiRunner.Mode.IO_ONLY)
        self._s_dur = 0.0

        c_inputs = self._prepare_inputs(s_inputs)
        c_outputs = self._prepare_outputs()
        in_buffers, out_buffers = self._get_io_buffers(c_inputs, c_outputs)

        start_time = t.perf_counter()
        run_batches = self._backend.ai_network_run(self._handle, in_buffers, out_buffers)
        error = self._backend.ai_network_get_error(self._handle)
        elapsed_time = (t.perf_counter() - start_time) * 1000.0
        if error.code or run_batches != c_inputs[0].shape[0]:
            msg = 'ai_network_run() failed\n AiError - {}'.format(stm_ai_error_to_str(error.code, error.type))
            raise AiRunnerError(msg)

        outputs = self._post_outputs(c_outputs)

        dur_ = elapsed_time
        if self._profiler:
            self._profiler['debug']['exec_times'].append(elapsed_time)
            if self._mode == AiRunner.Mode.IO_ONLY:
                self._profiler['c_durations'].append(dur_)
            else:
                dur_ = self._s_dur

        return outputs, dur_
