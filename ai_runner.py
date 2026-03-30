###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
Main entry point for ai runner module
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Jean-Michel Delorme"
__copyright__ = "Copyright (c) 2021 STMicroelectronics"
__license__ = """
    Copyright (c) 2020-2021 STMicroelectronics.
    All rights reserved.
    This software is licensed under terms that can be found in the LICENSE file in
    the root directory of this software component.
    If no LICENSE file comes with this software, it is provided AS-IS.
"""

import sys
from abc import ABC, abstractmethod
import time as t
import logging
from enum import Enum
import numpy as np


class AiRunnerError(Exception):
    """Base exceptions for errors raised by AIRunner"""
    error = 800
    idx = 0

    def __init__(self, mess=None):
        self.mess = mess
        super(AiRunnerError, self).__init__(mess)

    def code(self):
        """Return code number"""  # noqa: DAR101,DAR201,DAR401
        return self.error + self.idx

    def __str__(self):
        """Return formatted error description"""  # noqa: DAR101,DAR201,DAR401
        _mess = ''
        if self.mess is not None:
            _mess = '{}'.format(self.mess)
        else:
            _mess = type(self).__doc__.split('\n')[0]
        _msg = 'E{}({}): {}'.format(self.code(),
                                    type(self).__name__,
                                    _mess)
        return _msg


class HwIOError(AiRunnerError):
    """Low-level IO error"""
    idx = 1


class NotInitializedMsgError(AiRunnerError):
    """Message is not fully initialized"""
    idx = 2


class InvalidMsgError(AiRunnerError, ValueError):
    """Message is not correctly formatted"""
    idx = 3


class InvalidParamError(AiRunnerError):
    """Invali parameter"""
    idx = 4


class InvalidModelError(AiRunnerError):
    """Invali Model"""
    idx = 5


class NotConnectedError(AiRunnerError):
    """STM AI run-time is not connected"""
    idx = 10


def get_logger(name, debug=False, verbosity=0):
    """Utility function to create a logger object"""  # noqa: DAR101,DAR201,DAR401
    if name is None:
        return logging.getLogger('_DummyLogger_')  # without configuring dummy before.

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        # logger 'name' already created
        return logger

    if debug:
        _lvl = logging.DEBUG
    elif verbosity:
        _lvl = logging.INFO
    else:
        _lvl = logging.WARNING
    logger.setLevel(_lvl)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_lvl)
    # formatter = logging.Formatter('%(name)-12s:%(levelname)-7s: %(message)s')
    formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False

    return logger


def generate_rnd(types, shapes, batch_size=4, rng=np.random.RandomState(42)):
    """Generate list of random arrays"""  # noqa: DAR101,DAR201,DAR401
    no_list = False
    if not isinstance(types, list) and not isinstance(shapes, list):
        types = [types]
        shapes = [shapes]
        no_list = True
    batch_size = max(1, batch_size)
    inputs = []
    for type_, shape_ in zip(types, shapes):
        shape_ = (batch_size,) + shape_[1:]
        if type_ == np.bool:
            in_ = rng.randint(2, size=shape_).astype(np.bool)
        elif type_ != np.float32:
            high = np.iinfo(type_).max
            low = np.iinfo(type_).min
            #  “discrete uniform” distribution
            in_ = rng.randint(low=low, high=high + 1, size=shape_)
        else:
            # uniformly distributed over the half-open interval [low, high)
            in_ = rng.uniform(low=-1.0, high=1.0, size=shape_)

        in_ = np.ascontiguousarray(in_.astype(type_))
        inputs.append(in_)
    return inputs[0] if no_list else inputs


class AiHwDriver(ABC):
    """Base class to handle the LL IO COM functions"""
    def __init__(self):
        self._parent = None
        self._hdl = None
        self._logger = None

    @property
    def is_connected(self):
        """Indicate if the driver is connected"""
        # noqa: DAR101,DAR201,DAR401
        return bool(self._hdl)

    def set_parent(self, parent):
        """"Set parent object"""  # noqa: DAR101,DAR201,DAR401
        self._parent = parent
        if hasattr(self._parent, 'get_logger'):
            self._logger = self._parent.get_logger()

    def get_config(self):
        """Return a dict with the specific config"""  # noqa: DAR101,DAR201,DAR401
        return dict()

    @abstractmethod
    def _connect(self, desc=None, **kwargs):
        pass

    @abstractmethod
    def _disconnect(self):
        pass

    @abstractmethod
    def _read(self, size, timeout=0):
        return 0

    @abstractmethod
    def _write(self, data, timeout=0):
        return 0

    def connect(self, desc=None, **kwargs):
        """Connect the driver"""  # noqa: DAR101,DAR201,DAR401
        self.disconnect()
        return self._connect(desc=desc, **kwargs)

    def disconnect(self):
        """Disconnect the driver"""
        if self.is_connected:
            self._disconnect()

    def read(self, size, timeout=0):
        """Read the data"""  # noqa: DAR101,DAR201,DAR401
        if self.is_connected:
            return self._read(size, timeout)
        raise NotConnectedError()

    def write(self, data, timeout=0):
        """Write the data"""  # noqa: DAR101,DAR201,DAR401
        if self.is_connected:
            return self._write(data, timeout)
        raise NotConnectedError()

    def short_desc(self):
        """Return short human description"""  # noqa: DAR101,DAR201,DAR401
        return 'UNDEFINED'


class AiRunnerDriver(ABC):
    """Base class interface for an AI Runner driver"""
    def __init__(self, parent):
        if not hasattr(parent, 'get_logger'):
            raise InvalidParamError('Invalid parent type, get_logger() attr is expected')
        self._parent = parent
        self._logger = parent.get_logger()
        self._logger.debug('creating {} object'.format(self.__class__.__name__))

    def get_logger(self):
        """
        Return logger object

        Returns
        -------
        log
            logger object from the parent
        """
        return self._logger

    @abstractmethod
    def connect(self, desc=None, **kwargs):
        """Connect to the stm.ai run-time"""
        # noqa: DAR101,DAR201,DAR401
        return False

    @property
    def is_connected(self):
        """Indicate if the diver is connected"""
        # noqa: DAR101,DAR201,DAR401
        return False

    @abstractmethod
    def disconnect(self):
        """Disconnect to the stm.ai run-time"""
        # noqa: DAR101,DAR201,DAR401

    @abstractmethod
    def discover(self, flush=False):
        """Return list of available networks"""
        # noqa: DAR101,DAR201,DAR401
        return []

    @abstractmethod
    def get_info(self, c_name=None):
        """Get c-network details (including runtime)"""
        # noqa: DAR101,DAR201,DAR401
        return dict()

    @abstractmethod
    def invoke_sample(self, s_inputs, **kwargs):
        """Invoke the c-network with a given input (sample mode)"""
        # noqa: DAR101,DAR201,DAR401
        return [], dict()

    def check_inputs(self, _1, _2):
        """Specific function to check the inputs"""  # noqa: DAR101,DAR201,DAR401
        return False


class AiRunnerSession:
    """
    Interface to use a model
    """

    def __init__(self, name):
        """Constructor"""  # noqa: DAR101,DAR201,DAR401
        self._parent = None
        self._name = name

    def __str__(self):
        """Return c-name of the associated c-model"""  # noqa: DAR101,DAR201,DAR401
        return self.name

    @property
    def is_active(self):
        """Indicate if the session is active"""
        # noqa: DAR101,DAR201,DAR401
        return bool(self._parent)

    @property
    def name(self):
        """Return the name of the model"""
        # noqa: DAR101,DAR201,DAR401
        return self._name

    def acquire(self, parent):
        """Set the parent"""  # noqa: DAR101
        self._parent = parent

    def release(self):
        """Release the resources"""
        self._parent = None

    def get_input_infos(self):
        """Get model input details"""  # noqa: DAR101,DAR201,DAR401
        if self._parent:
            return self._parent.get_input_infos(self.name)
        return list()

    def get_output_infos(self):
        """Get model outputs details"""  # noqa: DAR101,DAR201,DAR401
        if self._parent:
            return self._parent.get_output_infos(self.name)
        return list()

    def get_info(self):
        """Get model details (including runtime)"""  # noqa: DAR101,DAR201,DAR401
        if self._parent:
            return self._parent.get_info(self.name)
        return dict()

    def invoke(self, inputs, **kwargs):
        """Invoke the c-network"""  # noqa: DAR101,DAR201,DAR401
        if self._parent:
            kwargs.pop('name', None)
            return self._parent.invoke(inputs, name=self.name, **kwargs)
        return list(), dict()

    def summary(self, print_fn=None, level=0):
        """Summary model & runtime infos"""  # noqa: DAR101,DAR201,DAR401
        if self._parent:
            return self._parent.summary(name=self.name, print_fn=print_fn, level=level)
        return None

    def disconnect(self):
        """Disconnect the run-time"""  # noqa: DAR101,DAR201,DAR401
        parent = self._parent
        if parent:
            self.release()
            parent.disconnect(force_all=False)


class AiRunnerCallback:
    """
    Abstract base class used to build new callbacks
    """
    def __init__(self):
        pass

    def on_sample_begin(self, idx):
        """
        Called at the beginning of each sample

        Parameters
        ----------
        idx
            Integer, index of the sample
        """

    def on_sample_end(self, idx, data, logs=None):  # pylint: disable=unused-argument
        """
        Called at the end of each sample

        Parameters
        ----------
        idx
            Integer, index of the sample
        data
            List, output tensors (numpy ndarray objects)
        logs
            Dict

        Returns
        -------
        bool
            True to continue, False to stop the inference
        """
        return True

    def on_node_begin(self, idx, data, logs=None):
        """
        Called before each c-node

        Parameters
        ----------
        idx
            Integer, index of the c-node
        data
            List, input tensors (numpy ndarray objects)
        logs
            Dict
        """

    def on_node_end(self, idx, data, logs=None):
        """
        Called at the end of each c-node

        Parameters
        ----------
        idx
            Integer, index of the c-node
        data
            List, output tensors
        logs
            Dict
        """


class AiRunner:
    """AI Runner or interpreter interface for stm.ai runtime.

    !!! example
        ```python
           from stm_ai_runner import AiRunner
           runner = AiRunner()
           runner.connect('serial')  # connection to a STM32 board (default)
           ...
           outputs, _ = runner.invoke(input_data)  # invoke the model
        ```
    """

    class Caps(Enum):
        """Capability values"""
        IO_ONLY = 0
        PER_LAYER = 1
        PER_LAYER_WITH_DATA = PER_LAYER | 2
        SELF_TEST = 4
        RELOC = 8

    class Mode(Enum):
        """Mode values"""
        IO_ONLY = 0
        PER_LAYER = 1
        PER_LAYER_WITH_DATA = PER_LAYER | 2

    def __init__(self, logger=None, debug=False, verbosity=0):
        """
        Constructor

        Parameters
        ----------
        logger
            Logger object which must be used
        debug
            Logger is created with DEBUG level if True
        verbosity
            Logger is created with INFO level if > 0
        """
        self._sessions = []
        self._names = []
        self._drv = None
        if logger is None:
            logger = get_logger(self.__class__.__name__, debug, verbosity)
        self._logger = logger
        self._logger.debug('creating %s object', str(self.__class__.__name__))
        self._debug = debug
        self._last_err = None

    def get_logger(self):
        """Return the logger object"""  # noqa: DAR101,DAR201,DAR401
        return self._logger

    def __repr__(self):
        return self.short_desc()

    def __str__(self):
        return self.short_desc()

    def get_error(self):
        """Return human readable description of the last error"""  # noqa: DAR101,DAR201,DAR401
        err_ = self._last_err
        self._last_err = None
        return err_

    @property
    def is_connected(self):
        """Indicate if the associated runtime is connected"""
        # noqa: DAR101,DAR201,DAR401
        return False if not self._drv else self._drv.is_connected

    def _check_name(self, name):
        """Return a valid c-network name"""  # noqa: DAR101,DAR201,DAR401
        if not self._names:
            return None
        if isinstance(name, int):
            idx = max(0, min(name, len(self._names) - 1))
            return self._names[idx]
        if name is None or not isinstance(name, str):
            return self._names[0]
        if name in self._names:
            return name
        return None

    def get_info(self, name=None):
        """
        Get model details (including runtime infos)

        Parameters
        ----------
        name
            c-name of the model (if None, first c-model is used)

        Returns
        -------
        dict
            Dict with the model information
        """
        name_ = self._check_name(name)
        return self._drv.get_info(name_) if name_ else dict()

    @property
    def name(self):
        """Return default network c-name (first)"""
        # noqa: DAR101,DAR201,DAR401
        return self._check_name(None)

    def get_input_infos(self, name=None):
        """
        Get model input details

        Parameters
        ----------
        name
            c-name of the model (if None, first c-model is used)

        Returns
        -------
        list
            List of dict with the input details
        """
        info_ = self.get_info(name)
        return info_['inputs'] if info_ else list()

    def get_output_infos(self, name=None):
        """
        Get model output details

        Parameters
        ----------
        name
            c-name of the model (if None, first c-model is used)

        Returns
        -------
        list
            List of dict with the output details
        """
        info_ = self.get_info(name)
        return info_['outputs'] if info_ else list()

    def _align_requested_mode(self, mode):
        """Align requested mode with drv capabilities"""  # noqa: DAR101,DAR201,DAR401
        if mode not in AiRunner.Mode:
            mode = AiRunner.Mode.IO_ONLY
        if mode == AiRunner.Mode.PER_LAYER_WITH_DATA:
            if AiRunner.Caps.PER_LAYER_WITH_DATA not in self._drv.capabilities:
                mode = AiRunner.Mode.PER_LAYER
        if mode == AiRunner.Mode.PER_LAYER:
            if AiRunner.Caps.PER_LAYER not in self._drv.capabilities:
                mode = AiRunner.Mode.IO_ONLY
        return mode

    def _check_inputs(self, inputs, name):
        """Check the coherence of the inputs (data type and shape)"""  # noqa: DAR101,DAR201,DAR401

        if self._drv.check_inputs(inputs, name):
            return

        in_desc = self.get_input_infos(name)

        if len(inputs) != len(in_desc):
            msg = 'Input number is inconsistent {} instead {}'.format(len(inputs), len(in_desc))
            raise HwIOError(msg)

        for idx, ref in enumerate(in_desc):
            in_shape = (ref['shape'][0],) + inputs[idx].shape[1:]
            if inputs[idx].dtype != ref['type']:
                msg = 'invalid dtype - {} instead {}'.format(inputs[idx].dtype,
                                                             ref['type'])
                msg += ' for the input #{}'.format(idx + 1)
                raise InvalidParamError(msg)
            if in_shape != ref['shape']:
                msg = 'invalid shape - {} instead {}'.format(in_shape,
                                                             ref['shape'])
                msg += ' for the input #{}'.format(idx + 1)
                raise InvalidParamError(msg)

    def invoke(self, inputs, **kwargs):
        """
        Generate output predictions, invoke the c-network run-time (batch mode)

        Parameters
        ----------
        inputs
            Input samples. A Numpy array, or a list of arrays in case the model
            has multiple inputs.
        kwargs
            specific parameters

        Returns
        -------
        list
            list of numpy arrays (one by outputs)

        """
        import tqdm

        name_ = self._check_name(kwargs.pop('name', None))

        if name_ is None:
            return list(), dict()

        if not isinstance(inputs, list):
            inputs = [inputs]

        self._check_inputs(inputs, name_)

        callback = kwargs.pop('callback', None)
        mode = self._align_requested_mode(kwargs.pop('mode', AiRunner.Mode.IO_ONLY))
        disable_pb = kwargs.pop('disable_pb', False)

        batch_size = inputs[0].shape[0]
        profiler = {
            'info': dict(),
            'c_durations': [],  # Inference time by sample w/o cb by node if enabled
            'c_nodes': [],
            'debug': {
                'exec_times': [],  # real inference time by sample with cb by node overhead
                'host_duration': 0.0,  # host execution time (on whole batch)
            },
        }

        start_time = t.perf_counter()
        outputs = []
        prog_bar = None
        cont = True
        for batch in range(batch_size):
            if not prog_bar and not disable_pb and (t.perf_counter() - start_time) > 1:
                prog_bar = tqdm.tqdm(total=batch_size, file=sys.stdout, desc='STM.IO',
                                     unit_scale=False,
                                     leave=False)
                prog_bar.update(batch)
            elif prog_bar:
                prog_bar.update(1)
            s_inputs = [np.expand_dims(in_[batch], axis=0) for in_ in inputs]
            if callback:
                callback.on_sample_begin(batch)
            s_outputs, s_dur = self._drv.invoke_sample(s_inputs, name=name_,
                                                       profiler=profiler, mode=mode,
                                                       callback=callback)
            if batch == 0:
                outputs = s_outputs
            else:
                for idx, out_ in enumerate(s_outputs):
                    outputs[idx] = np.append(outputs[idx], out_, axis=0)
            if callback:
                cont = callback.on_sample_end(batch, s_outputs, logs={'dur': s_dur})
            if not cont:
                break
        profiler['debug']['host_duration'] = (t.perf_counter() - start_time) * 1000.0
        profiler['info'] = self.get_info(name_)
        if prog_bar:
            prog_bar.close()

        return outputs, profiler

    def generate_rnd_inputs(self, name=None, batch_size=4, rng=np.random.RandomState(42)):
        """Generate input data with random values"""  # noqa: DAR101,DAR201,DAR401
        if isinstance(name, AiRunnerSession):
            name = name.name
        name_ = self._check_name(name)
        if name_ is None:
            return []
        info_ = self.get_input_infos(name_)
        return generate_rnd([t_['type'] for t_ in info_],
                            [s_['shape'] for s_ in info_],
                            batch_size,
                            rng=rng)

    @property
    def names(self):
        """Return available nets as a list of name"""
        # noqa: DAR101,DAR201,DAR401
        return self._names

    def _release_all(self):
        """Release all resources"""  # noqa: DAR101,DAR201,DAR401
        if self._names:
            self._logger.debug("_release_all(%s)", str(self))
        for ses_ in self._sessions:
            ses_.release()
            self._sessions.remove(ses_)
        self._names = []
        if self.is_connected:
            self._drv.disconnect()
            self._drv = None
        return True

    def short_desc(self):
        """Return short description of the associated run-time"""  # noqa: DAR101,DAR201,DAR401
        if self.is_connected:
            desc_ = '{} {}'.format(self._drv.short_desc(), self.names)
            return desc_
        return 'not connected'

    def connect(self, desc=None, **kwargs):
        """Connect to a given runtime defined by desc"""  # noqa: DAR101,DAR201,DAR401
        from .ai_resolver import ai_runner_resolver

        self._logger.debug("connect(desc='%s')", str(desc))
        self._release_all()

        self._drv, desc_ = ai_runner_resolver(self, desc)
        self._logger.debug("desc for the driver: '%s'", str(desc_))

        if self._drv is None:
            self._last_err = desc_
            self._logger.debug(desc_)
            return False

        try:
            self._drv.connect(desc_, **kwargs)
            if not self._drv.is_connected:
                self._release_all()
            else:
                self._names = self._drv.discover(flush=True)
                for name_ in self._names:
                    self._sessions.append(AiRunnerSession(name_))
        except Exception as exc_:  # pylint: disable=broad-except
            msg_ = 'connection to "{}"/"{}" run-time fails\n {}'.format(desc, desc_, str(exc_))
            self._logger.debug(msg_)
            self._last_err = msg_
            self._release_all()

        return self.is_connected

    def session(self, name=None):
        """Return session handler for the given model name/idx"""  # noqa: DAR101,DAR201,DAR401
        if not self.is_connected:
            return None
        name = self._check_name(name)
        if name:
            for ses_ in self._sessions:
                if name == ses_.name:
                    ses_.acquire(self)
                    return ses_
        return None

    def disconnect(self, force_all=True):
        """Close the connection with the run-time"""  # noqa: DAR101,DAR201,DAR401
        if not force_all:
            # check is a session is on-going
            for ses_ in self._sessions:
                if ses_.is_active:
                    return True
        return self._release_all()

    def summary(self, name=None, print_fn=None, level=0):
        """Prints a summary of the model & associated runtime"""  # noqa: DAR101,DAR201,DAR401

        print_fn = print if print_fn is None else print_fn

        dict_info = self.get_info(name)
        if dict_info:

            def _attr(attr, val):
                print_fn('{:20s} : {}'.format(str(attr), str(val)))

            def _tens_to_str(val):
                ext_ = ''
                if val['scale']:
                    ext_ = ', scale={}, zp={}'.format(val['scale'], val['zero_point'])
                ext_ += ', {}'.format(val.get('memory_pool', None))
                sz_in_bytes = np.prod(val['shape']) * np.dtype(val['type']).itemsize
                _attr(val['name'], '{}, {}, {} bytes{}'.format(val['shape'], np.dtype(val['type']),
                                                               sz_in_bytes, ext_))

            print_fn('Summary "{}" - {}'.format(dict_info['name'], self._names))
            print_fn('-' * 80)
            _attr('inputs/outputs', '{}/{}'.format(len(dict_info['inputs']), len(dict_info['outputs'])))
            for in_ in dict_info['inputs']:
                _tens_to_str(in_)
            for out_ in dict_info['outputs']:
                _tens_to_str(out_)
            _attr('n_nodes', dict_info['n_nodes'])
            date = '{} ({})'.format(dict_info.get('compile_datetime', 'undefined'),
                                    dict_info.get('model_datetime', 'undefined'))
            _attr('compile_datetime', date)
            macc_ = 'n.a.' if dict_info.get('macc', 0) <= 1 else dict_info.get('macc', 0)
            _attr('activations', dict_info['activations'])
            _attr('weights', dict_info['weights'])
            _attr('macc', macc_)
            print_fn('-' * 80)
            _attr('runtime', '{} {}.{}.{} (Tools {}.{}.{})'.format(dict_info['runtime']['name'],
                                                                   *dict_info['runtime']['version'],
                                                                   *dict_info['runtime']['tools_version']))
            _attr('capabilities', [str(n).replace('Caps.', '') for n in dict_info['runtime']['capabilities']])
            if level > 0:
                for key, value in dict_info['device'].items():
                    _attr(key, value)
            else:
                _attr('device', dict_info['device']['desc'])

            print_fn('-' * 80)
            print_fn('')


if __name__ == '__main__':
    pass
