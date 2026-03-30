###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
Driver for proto buff messages
"""
import time as t
import numpy as np

from google.protobuf.internal.encoder import _VarintBytes

from .ai_runner import AiRunner, AiRunnerDriver
from .ai_runner import HwIOError, InvalidMsgError, NotInitializedMsgError, AiRunnerError
from .ai_runner import InvalidParamError, InvalidModelError
from .stm32_utility import stm32_id_to_str, stm32_attr_config
from . import stm32msg_pb2 as stm32msg
from .stm_ai_utils import stm_ai_node_type_to_str, AiBufferFormat
from .stm_ai_utils import stm_tflm_node_type_to_str, RT_STM_AI_NAME


def _to_version(ver):
    """Convert uint32 version to tuple (major, minor, sub)"""  # noqa: DAR101,DAR201,DAR401
    return (ver >> 24 & 0xFF, ver >> 16 & 0xFF, ver >> 8 & 0xFF)


_SUPPORTED_AI_RT = {
    stm32msg.AI_RT_STM_AI: RT_STM_AI_NAME,
    stm32msg.AI_RT_TFLM: 'TFLM',
    stm32msg.AI_RT_TVM: 'TVM',
}

_SUPPORTED_AI_ARM_TOOLS = {
    stm32msg.AI_GCC: 'GCC',
    stm32msg.AI_IAR: 'IAR',
    stm32msg.AI_MDK_5: 'AC5',
    stm32msg.AI_MDK_6: 'AC6',
}


def _rt_ai_desc(rt_id, is_supported=True):
    """Return short description - name+ext of the runtime"""  # noqa: DAR101,DAR201,DAR401
    if not is_supported:
        return '?'
    desc_ = _SUPPORTED_AI_RT.get(rt_id & 0xFF, "?")
    if rt_id >> 8:
        desc_ += ' ('
        if rt_id >> 8 & 0xFF:
            desc_ += '/{}'.format(_SUPPORTED_AI_ARM_TOOLS.get(rt_id >> 8 & 0xFF, '?').lower())
        desc_ += ')'
    return desc_


class AiPbMsg(AiRunnerDriver):
    """Class to handle the messages (protobuf-based)"""

    _NODE_WITH_MULTIPLE_OUTPUTS = 0
    _SHAPE_MSG_WITH_ADDR = 1
    _SYNC_WITH_AI_RT_ID = 2

    def __init__(self, parent, io_drv):
        """Constructor"""  # noqa: DAR101,DAR201,DAR401
        if not hasattr(io_drv, 'set_parent'):
            raise InvalidParamError('Invalid IO Hw Driver type (io_drv)')
        self._req_id = 0
        self._io_drv = io_drv
        self._models = dict()  # cache the description of the models
        self._sync = None  # cache for the sync message
        self._sys_info = None  # cache for sys info message
        super(AiPbMsg, self).__init__(parent)
        self._io_drv.set_parent(self)
        self._rt_type = None
        self._logger.debug('creating {} object'.format(self._io_drv.__class__.__name__))

    @property
    def is_connected(self):
        return self._io_drv.is_connected

    def connect(self, desc=None, **kwargs):
        """Connect to the stm.ai run-time"""  # noqa: DAR101,DAR201,DAR401
        if self._io_drv.is_connected:
            return False
        return self._io_drv.connect(desc, **kwargs)

    @property
    def capabilities(self):
        """Return list with capabilities"""
        # noqa: DAR101,DAR201,DAR401
        if self.is_connected:
            cap_ = [AiRunner.Caps.IO_ONLY]
            if self._sync.capability & stm32msg.CAP_INSPECTOR:
                cap_.extend([AiRunner.Caps.PER_LAYER, AiRunner.Caps.PER_LAYER_WITH_DATA])
            if self._sync.capability & stm32msg.CAP_SELF_TEST:
                cap_.append(AiRunner.Caps.SELF_TEST)
            if self._sync.capability & stm32msg.CAP_RELOC:
                cap_.append(AiRunner.Caps.RELOC)
            return cap_
        return []

    def disconnect(self):
        self._models = dict()
        self._sys_info = None
        self._sync = None
        self._io_drv.disconnect()

    def short_desc(self):
        """Return human readable description"""  # noqa: DAR101,DAR201,DAR401
        ver_ = '{}.{}'.format(stm32msg.P_VERSION_MAJOR, stm32msg.P_VERSION_MINOR)
        io_ = self._io_drv.short_desc()
        return 'STM Proto-buffer protocol ' + ver_ + ' (' + io_ + ')'

    def _waiting_io_ack(self, timeout):
        """Wait a ack"""  # noqa: DAR101,DAR201,DAR401
        start_time = t.perf_counter()
        while True:
            if self._io_drv.read(1):
                break
            if t.perf_counter() - start_time > timeout / 1000.0:
                return False
        return True

    def _write_io_packet(self, payload, delay=0):
        iob = bytearray(stm32msg.IO_OUT_PACKET_SIZE + 1)
        iob[0] = len(payload)
        for i, val in enumerate(payload):
            iob[i + 1] = val
        if not delay:
            _w = self._io_drv.write(iob)
        else:
            _w = 0
            for elem in iob:
                _w += self._io_drv.write(elem.to_bytes(1, 'big'))
                t.sleep(delay)
        return _w

    def _write_delimited(self, mess, timeout=5000):
        """Helper function to write a message prefixed with its size"""  # noqa: DAR101,DAR201,DAR401

        if not mess.IsInitialized():
            raise NotInitializedMsgError

        buff = mess.SerializeToString()
        _head = _VarintBytes(mess.ByteSize())

        buff = _head + buff

        packs = [buff[i:i + stm32msg.IO_OUT_PACKET_SIZE]
                 for i in range(0, len(buff), stm32msg.IO_OUT_PACKET_SIZE)]

        n_w = self._write_io_packet(packs[0])
        for pack in packs[1:]:
            if not self._waiting_io_ack(timeout):
                break
            n_w += self._write_io_packet(pack)

        return n_w

    def _parse_and_check(self, data, msg_type=None):
        """Parse/convert and check the received buffer"""  # noqa: DAR101,DAR201,DAR401
        resp = stm32msg.respMsg()
        try:
            resp.ParseFromString(data)
        except BaseException as exc_:
            raise InvalidMsgError(str(exc_))
        if msg_type is None:
            return resp
        elif resp.WhichOneof('payload') != msg_type:
            raise InvalidMsgError('receive \'{}\' instead \'{}\''.format(
                resp.WhichOneof('payload'), msg_type))
        return None

    def _waiting_msg(self, timeout, msg_type=None):
        """Helper function to receive a message"""  # noqa: DAR101,DAR201,DAR401
        buf = bytearray()

        packet_s = int(stm32msg.IO_IN_PACKET_SIZE + 1)
        if timeout == 0:
            t.sleep(0.2)

        start_time = t.monotonic()
        while True:
            p_buf = bytearray()
            while len(p_buf) < packet_s:
                io_buf = self._io_drv.read(packet_s - len(p_buf))
                if io_buf:
                    p_buf += io_buf
                else:
                    cum_time = t.monotonic() - start_time
                    if timeout and (cum_time > timeout / 1000):
                        raise TimeoutError(
                            'STM32 - read timeout {:.1f}ms/{}ms'.format(cum_time * 1000, timeout))
                    if timeout == 0:
                        return self._parse_and_check(buf, msg_type)
            last = p_buf[0] & stm32msg.IO_HEADER_EOM_FLAG
            # cbuf[0] = cbuf[0] & 0x7F & ~stm32msg.IO_HEADER_SIZE_MSK
            p_buf[0] &= 0x7F  # & ~stm32msg.IO_HEADER_SIZE_MSK)
            if last:
                buf += p_buf[1:1 + p_buf[0]]
                break
            buf += p_buf[1:packet_s]
        resp = self._parse_and_check(buf, msg_type)
        return resp

    def _send_request(self, cmd, param=0, name=None, opt=0):
        """Build a request msg and send it"""  # noqa: DAR101,DAR201,DAR401
        self._req_id += 1
        req_msg = stm32msg.reqMsg()
        req_msg.reqid = self._req_id
        req_msg.cmd = cmd
        req_msg.param = param
        req_msg.opt = opt
        if name is not None and isinstance(name, str):
            req_msg.name = name
        else:
            req_msg.name = ''
        n_w = self._write_delimited(req_msg)
        return n_w, req_msg

    def _send_ack(self, param=0, err=0):
        """Build an acknowledge msg and send it"""  # noqa: DAR101,DAR201,DAR401
        ack_msg = stm32msg.ackMsg(param=param, error=err)
        return self._write_delimited(ack_msg)

    def _device_log(self, resp):
        """Process a log message from a device"""  # noqa: DAR101,DAR201,DAR401
        if resp.WhichOneof('payload') == 'log':
            msg = 'STM32:{}: {}'.format(resp.log.level, resp.log.str)
            self._logger.info(msg)
            self._send_ack()
            return True
        return False

    def _waiting_answer(self, timeout=10000, msg_type=None, state=None):
        """Wait an answer/msg from the device and post-process it"""  # noqa: DAR101,DAR201,DAR401

        cont = True
        while cont:  # to manage the "log" msg
            resp = self._waiting_msg(timeout=timeout)
            if resp.reqid != self._req_id:
                raise InvalidMsgError('SeqID is not valid - {} instead {}'.format(
                    resp.reqid, self._req_id))
            cont = self._device_log(resp)

        if msg_type and resp.WhichOneof('payload') != msg_type:
            raise InvalidMsgError('receive \'{}\' instead \'{}\''.format(
                resp.WhichOneof('payload'), msg_type))

        if state and state != resp.state:
            raise HwIOError('Invalid state: {} instead {}'.format(
                resp.state, state))

        return resp

    def _cmd_sync(self, timeout):
        """SYNC command"""  # noqa: DAR101,DAR201,DAR401
        self._send_request(stm32msg.CMD_SYNC)
        resp = self._waiting_answer(timeout=timeout, msg_type='sync',
                                    state=stm32msg.S_IDLE)
        return resp.sync

    def _cmd_sys_info(self, timeout):
        """SYS_INFO command"""  # noqa: DAR101,DAR201,DAR401
        self._send_request(stm32msg.CMD_SYS_INFO)
        resp = self._waiting_answer(timeout=timeout, msg_type='sinfo',
                                    state=stm32msg.S_IDLE)
        return resp.sinfo

    def _cmd_network_info(self, timeout, param=0):
        """SYS_INFO command"""  # noqa: DAR101,DAR201,DAR401
        self._send_request(stm32msg.CMD_NETWORK_INFO, param=param)
        resp = self._waiting_answer(timeout=timeout, state=stm32msg.S_IDLE)
        if resp.WhichOneof('payload') == 'ninfo':
            return resp.ninfo
        elif resp.WhichOneof('payload') == 'ack':
            if resp.ack.error == stm32msg.E_GENERIC:
                rt_id_str = _rt_ai_desc(self._sync.rtid, self._is_supported(self._SYNC_WITH_AI_RT_ID))
                raise InvalidModelError('Model {} is not initialized ({})'.format(
                    param, rt_id_str))
        return None

    def _cmd_run(self, timeout, c_name, param):
        """NETWORK_RUN command"""  # noqa: DAR101,DAR201,DAR401
        self._send_request(stm32msg.CMD_NETWORK_RUN, param=param, name=c_name)
        resp = self._waiting_answer(timeout=timeout, msg_type='ack',
                                    state=stm32msg.S_WAITING)
        return resp

    def _protocol_is_supported(self):
        """Indicate if the protocol version is supported"""  # noqa: DAR101,DAR201,DAR401
        major, minor = self._sync.version >> 8, self._sync.version & 0xFF
        if major == stm32msg.P_VERSION_MAJOR and minor <= stm32msg.P_VERSION_MINOR:
            return True
        msg = 'COM Protocol is not supported by the driver: {}.{}'.format(major, minor)
        msg += ' instead {}.[1,{}]'.format(stm32msg.P_VERSION_MAJOR, stm32msg.P_VERSION_MINOR)
        raise HwIOError(msg)

    def _is_supported(self, feature):
        """Indicates if a specific feature is supported by the Protocol"""  # noqa: DAR101,DAR201,DAR401
        if (self._sync.version >> 8 == 2) and (self._sync.version & 0xFF >= 2):
            if feature in [self._NODE_WITH_MULTIPLE_OUTPUTS, self._SHAPE_MSG_WITH_ADDR, self._SYNC_WITH_AI_RT_ID]:
                return True
        return False

    def is_alive(self, timeout=500):
        """"Indicate if the connection is always alive"""  # noqa: DAR101,DAR201,DAR401
        try:
            self._sync = self._cmd_sync(timeout)
            if self._is_supported(self._SYNC_WITH_AI_RT_ID):
                self._rt_type = self._sync.rtid & 0xFF
            self._logger.debug('CMD_SYS_INFO v{}.{}'.format(self._sync.version >> 8, self._sync.version & 0xFF))
        except (AiRunnerError, TimeoutError) as exc_:
            self._logger.debug('is_alive() %s', str(exc_))
            return False
        return self._protocol_is_supported()

    def _to_device(self):
        """Return a dict with the device settings"""  # noqa: DAR101,DAR201,DAR401
        if self._sys_info is None:
            self._sys_info = self._cmd_sys_info(timeout=500)

        desc_ = stm32_id_to_str(self._sys_info.devid)
        desc_ += ' @{:.0f}/{:.0f}MHz'.format(self._sys_info.sclock / 1000000,
                                             self._sys_info.hclock / 1000000)
        desc_ += ' ' + ','.join(stm32_attr_config(self._sys_info.cache))

        return {
            'dev_type': 'stm32',
            'desc': desc_,
            'dev_id': stm32_id_to_str(self._sys_info.devid),
            'system': 'no-os',
            'sys_clock': self._sys_info.sclock,
            'bus_clock': self._sys_info.hclock,
            'attrs': stm32_attr_config(self._sys_info.cache),
        }

    def _to_runtime(self, model_info):
        """Return a dict with the runtime attributes"""  # noqa: DAR101,DAR201,DAR401

        name_rt = 'Protocol {}.{} - '.format(self._sync.version >> 8, self._sync.version & 0xFF)
        if self._is_supported(self._SYNC_WITH_AI_RT_ID):
            name_rt += _rt_ai_desc(self._sync.rtid)
        else:
            # Basic hack to detect the AI run-time, workaround for Msg Protocol v2.1
            name_rt += 'STM-AI' if model_info.tool_api_version != model_info.api_version else model_info.tool_revision
        return {
            'name': name_rt,
            'version': _to_version(model_info.runtime_version),
            'capabilities': self.capabilities,
            'tools_version': _to_version(model_info.tool_version),
        }

    def _to_io_tensor(self, buffer, idx, name=None):
        """Return a dict with the tensor IO attributes"""  # noqa: DAR101,DAR201,DAR401

        item = {
            'name': '{}_{}'.format(name if name else 'tensor', idx + 1),
            'shape': (buffer.n_batches, buffer.height, buffer.width, buffer.channels),
            'type': AiBufferFormat.to_np_type(buffer.format),
            'scale': np.float32(buffer.scale) if buffer.scale != 0.0 else None,
            'zero_point': np.int32(buffer.zeropoint),
            'format': buffer.format,
        }
        if self._is_supported(self._SHAPE_MSG_WITH_ADDR):
            item['memory_pool'] = 'in activations buffer' if buffer.addr else 'user'
        else:
            item['memory_pool'] = None

        # adjust type of the zero_point
        if item['scale'] is not None:
            item['zero_point'].astype(item['type'])

        return item

    def _model_to_dict(self, model_info):
        """Return a dict with the network info"""  # noqa: DAR101,DAR201,DAR401
        return {
            'name': model_info.model_name,
            'model_datetime': model_info.model_datetime,  # date of creation
            'compile_datetime': model_info.compile_datetime,  # date of compilation
            'hash': model_info.model_signature,
            'n_nodes': model_info.n_nodes,
            'inputs': [self._to_io_tensor(d_, i, 'input') for i, d_ in enumerate(model_info.inputs)],
            'outputs': [self._to_io_tensor(d_, i, 'output') for i, d_ in enumerate(model_info.outputs)],
            'weights': model_info.weights.channels,
            'activations': model_info.activations.channels,
            'macc': model_info.n_macc,
            'runtime': self._to_runtime(model_info),
            'device': self._to_device(),
        }

    def get_info(self, c_name=None):
        """Return a dict with the network info of the given model"""  # noqa: DAR101,DAR201,DAR401
        if not self._models:
            return dict()
        if c_name is None or c_name not in self._models.keys():
            # first c-model is used
            c_name = self._models.keys()[0]
        model = self._models[c_name]
        return self._model_to_dict(model)

    def discover(self, flush=False):
        """Build the list of the available model"""  # noqa: DAR101,DAR201,DAR401
        if flush:
            self._models.clear()
        if self._models:
            return list(self._models.keys())
        param, cont = 0, True

        while cont:
            n_info = self._cmd_network_info(timeout=5000, param=param)
            if n_info is not None:
                self._models[n_info.model_name] = n_info
                msg = 'discover() found="{}"'.format(str(n_info.model_name))
                self._logger.debug(msg)
                param += 1
            else:
                cont = False
        return list(self._models.keys())

    def _to_buffer_msg(self, data, buffer_desc):
        """Convert ndarray to aiBufferByteMsg"""  # noqa: DAR101,DAR201,DAR401
        msg_ = stm32msg.aiBufferByteMsg()
        msg_.shape.n_batches = buffer_desc.n_batches  # pylint: disable=no-member
        msg_.shape.height = buffer_desc.height  # pylint: disable=no-member
        msg_.shape.width = buffer_desc.width  # pylint: disable=no-member
        msg_.shape.channels = buffer_desc.channels  # pylint: disable=no-member
        msg_.shape.format = buffer_desc.format  # pylint: disable=no-member
        msg_.shape.scale = 0.0  # pylint: disable=no-member
        msg_.shape.zeropoint = 0  # pylint: disable=no-member
        msg_.shape.addr = 0  # pylint: disable=no-member
        dt_ = np.dtype(data.dtype.type)
        dt_ = dt_.newbyteorder('<')
        msg_.datas = bytes(data.astype(dt_).flatten().tobytes())
        return msg_

    def _from_buffer_msg(self, msg, fill_with_zero=False):
        """Convert aiBufferByteMsg to ndarray"""  # noqa: DAR101,DAR201,DAR401
        if isinstance(msg, stm32msg.respMsg):
            buffer = msg.node.buffer
        elif isinstance(msg, stm32msg.nodeMsg):
            buffer = msg.node
        else:
            buffer = msg
        shape_ = (buffer.shape.n_batches, buffer.shape.height,
                  buffer.shape.width, buffer.shape.channels)
        dt_ = np.dtype(AiBufferFormat.to_np_type(buffer.shape.format))
        dt_ = dt_.newbyteorder('<')
        if fill_with_zero:  # or not buffer.datas:
            return np.zeros(shape_, dtype=dt_), shape_
        if not buffer.datas:
            return np.array([], dtype=dt_), shape_
        return np.reshape(np.frombuffer(buffer.datas, dtype=dt_), shape_), shape_

    def _send_buffer(self, data, buffer_desc, is_last=False):
        """Send a buffer to the device and wait an ack"""  # noqa: DAR101,DAR201,DAR401

        buffer_msg = self._to_buffer_msg(data, buffer_desc)
        self._write_delimited(buffer_msg)
        state = stm32msg.S_PROCESSING if is_last else stm32msg.S_WAITING
        self._waiting_answer(msg_type='ack', state=state)
        self._send_ack()

    def _layer_type_to_str(self, layer_type):
        """Return short description of the type of the operator"""  # noqa: DAR101,DAR201,DAR401
        # description is RT specific
        if self._rt_type == stm32msg.AI_RT_STM_AI:
            return stm_ai_node_type_to_str(layer_type & 0x7FFF)
        if self._rt_type == stm32msg.AI_RT_TFLM:
            return stm_tflm_node_type_to_str(layer_type & 0x7FFF)
        return stm_ai_node_type_to_str(layer_type & 0x7FFF)

    def _receive_features(self, profiler, callback):
        """Collect the intermediate/hidden values"""  # noqa: DAR101,DAR201,DAR401

        # main loop to receive the datas
        idx_node = 0
        duration = 0.0

        while True:
            resp = self._waiting_answer(msg_type='node', timeout=50000)
            ilayer = resp.node

            is_internal = ilayer.type >> 16
            is_internal = bool(is_internal & stm32msg.LAYER_TYPE_INTERNAL_LAST) or\
                bool(is_internal & stm32msg.LAYER_TYPE_INTERNAL)

            if not is_internal:
                return resp

            self._send_ack()

            # loop on a c-node to collect the data
            shapes, features, types, scales, zeropoints = [], [], [], [], []
            while True:
                feature, shape = self._from_buffer_msg(resp.node.buffer)
                features.append(feature)
                shapes.append(shape)
                types.append(feature.dtype.type)
                scales.append(resp.node.buffer.shape.scale)
                zeropoints.append(resp.node.buffer.shape.zeropoint)
                m_o = self._is_supported(self._NODE_WITH_MULTIPLE_OUTPUTS)
                if m_o and ilayer.type >> 16 & (stm32msg.LAYER_TYPE_INTERNAL_DATA_NO_LAST):
                    resp = self._waiting_answer(msg_type='node', timeout=1000)
                    ilayer = resp.node
                    self._send_ack()
                else:
                    # last data has been received
                    break

            if profiler:
                feature, shape = self._from_buffer_msg(resp.node.buffer)
                duration += ilayer.duration
                if idx_node >= len(profiler['c_nodes']):
                    item = {
                        'c_durations': [ilayer.duration],
                        'm_id': ilayer.id,
                        'layer_type': ilayer.type & 0x7FFF,
                        'layer_desc': self._layer_type_to_str(ilayer.type),
                        'type': types,
                        'shape': shapes,  # feature.shape,
                        'scale': scales,
                        'zero_point': zeropoints,
                        'data': features if features is not None else None,
                    }
                    profiler['c_nodes'].append(item)
                else:
                    item = profiler['c_nodes'][idx_node]
                    item['c_durations'].append(ilayer.duration)
                    for idx, _ in enumerate(features):
                        item['data'][idx] = np.append(item['data'][idx], features[idx], axis=0)

                if callback:
                    callback.on_node_end(idx_node,
                                         features if features is not None else None,
                                         logs={'dur': ilayer.duration,
                                               'shape': shapes,
                                               'm_id': ilayer.id,
                                               'layer-type': ilayer.type & 0x7FFF})

            # end main loop
            if ilayer.type >> 16 & stm32msg.LAYER_TYPE_INTERNAL_LAST:
                break

            idx_node += 1

        # retreive the report
        # legacy support (2.1 protocol) - not used here
        #  global execution time is reported in the output tensors
        self._waiting_answer(msg_type='report',
                             timeout=20000,
                             state=stm32msg.S_PROCESSING)
        self._send_ack()

        if profiler:
            profiler['c_durations'].append(duration)

        return None

    def invoke_sample(self, s_inputs, **kwargs):
        """Invoke the model (sample mode)"""  # noqa: DAR101,DAR201,DAR401

        if s_inputs[0].shape[0] != 1:
            raise HwIOError('Should be called with a batch size of 1')

        name = kwargs.pop('name', None)

        if name is None or name not in self._models.keys():
            raise InvalidParamError('Invalid requested model name: ' + name)

        model = self._models[name]

        profiler = kwargs.pop('profiler', None)
        mode = kwargs.pop('mode', AiRunner.Mode.IO_ONLY)
        callback = kwargs.pop('callback', None)

        if mode == AiRunner.Mode.PER_LAYER:
            param = stm32msg.P_RUN_MODE_INSPECTOR_WITHOUT_DATA
        elif mode == AiRunner.Mode.PER_LAYER_WITH_DATA:
            param = stm32msg.P_RUN_MODE_INSPECTOR
        else:
            param = stm32msg.P_RUN_MODE_NORMAL

        s_outputs = []

        # start a RUN task
        self._cmd_run(timeout=1000, c_name=name, param=param)

        # send the inputs
        for idx, buffer_desc in enumerate(model.inputs):
            input_ = s_inputs[idx]
            is_last = (idx + 1) == model.n_inputs
            self._send_buffer(input_, buffer_desc, is_last=is_last)

        # receive the features
        resp = self._receive_features(profiler, callback)

        # receive the outputs
        for idx, buffer_desc in enumerate(model.outputs):
            is_last = (idx + 1) == model.n_outputs
            state = stm32msg.S_DONE if is_last else stm32msg.S_PROCESSING
            if resp is None:
                resp = self._waiting_answer(msg_type='node', timeout=50000, state=state)
            output, _ = self._from_buffer_msg(resp.node.buffer)
            s_outputs.append(output)
            if not is_last:
                self._send_ack()
                resp = None

        if profiler:
            profiler['debug']['exec_times'].append(resp.node.duration)
            if mode != AiRunner.Mode.IO_ONLY and len(profiler['c_durations']) != 0:
                dur = profiler['c_durations'][-1]
            else:
                dur = resp.node.duration
                profiler['c_durations'].append(dur)
        else:
            dur = resp.node.duration

        return s_outputs, dur
