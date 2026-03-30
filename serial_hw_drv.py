###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
Serial Low Level or Hw Driver
"""

import time as t

from serial import Serial
import serial.tools.list_ports
from serial.serialutil import SerialException

from .ai_runner import AiHwDriver
from .ai_runner import HwIOError


def serial_device_discovery():
    """
    Scan for available serial ports.

    Returns
    -------
    list
        return a list of available ports
    """

    comports = serial.tools.list_ports.comports()
    dev_list = []
    for com in comports:
        elem = {
            'type': 'serial',
            'device': com.device,
            'desc': com.description,
            'hwid': com.hwid
        }
        dev_list.append(elem)

    return dev_list


def serial_get_com_settings(desc):
    """
    Parse the desc parameter to retrieve the COM id and the baudrate.
    Can be a "str" or directly an "int" if only the baudrate is passed.

    Example:

        'COM7:115200'      ->  'COM7'  115200
        '460800'           ->   None   460800
        ':921600           ->   None   921600
        'COM6'             ->   COM6   115200
        ':COM6             ->   COM6   115200

    Parameters
    ----------
    desc
        str with the description of the COM port

    Returns
    -------
    tuple
        2-tuple port/com name and baudrate

    """

    # default values
    port_ = None
    baud_ = int(115200)

    if desc is not None and isinstance(desc, int):
        return port_, int(desc)

    if desc is None or not isinstance(desc, str):
        return port_, baud_

    desc = desc.split(':')
    for _d in desc:
        if _d:
            try:
                _d = int(_d)
            except (ValueError, TypeError):
                port_ = _d
            else:
                baud_ = _d

    return port_, baud_


class SerialHwDriver(AiHwDriver):
    """Serial low-level IO driver"""

    def __init__(self):
        """Constructor"""  # noqa: DAR101,DAR201,DAR401
        self._device = None
        self._baudrate = 0
        super().__init__()

    def get_config(self):
        """"Return a dict with used configuration"""  # noqa: DAR101,DAR201,DAR401
        return {
            'device': self._device,
            'baudrate': self._baudrate
        }

    def _open(self, device, baudrate, timeout, dry_run=False):
        """Open the COM port"""  # noqa: DAR101,DAR201,DAR401
        n_retry = 4
        hdl = None
        while n_retry:
            try:
                hdl = Serial(device, baudrate=baudrate, timeout=timeout,
                             exclusive=True)
            except SerialException as _e:
                n_retry -= 1
                if not n_retry:
                    if not dry_run:
                        raise HwIOError('{}'.format(_e))
                    else:
                        return None
                t.sleep(0.2)
            else:
                break
        return hdl

    def _discovery(self, device, baudrate, timeout):
        """Discover the possible COM port"""  # noqa: DAR101,DAR201,DAR401
        if device is None:
            devices = serial_device_discovery()
        else:
            devices = [{'device': device}]
        for dev in devices:
            dry_run = dev != devices[-1]
            hdl_ = self._open(dev['device'], baudrate, timeout, dry_run=dry_run)
            if hdl_:
                self._hdl = hdl_
                self._hdl.reset_input_buffer()
                self._hdl.reset_output_buffer()
                self._device = dev['device']
                self._baudrate = baudrate
                cpt = 1000
                while self._read(10) and cpt:
                    cpt = cpt - 1
                t.sleep(0.4)
                if hasattr(self._parent, 'is_alive') and self._parent.is_alive():
                    self._hdl.reset_input_buffer()
                    self._hdl.reset_output_buffer()
                    return None
                self._hdl.close()
                self._hdl = None
                if not dry_run:
                    raise HwIOError('{} - {}:{}'.format('Invalid firmware', dev['device'], baudrate))
        if not devices:
            raise HwIOError('No SERIAL COM port detected (STM32 board is not connected!)')
        return None

    def _connect(self, desc=None, **kwargs):
        """Open a connection"""  # noqa: DAR101,DAR201,DAR401

        dev_, baud_ = serial_get_com_settings(desc)
        baud_ = kwargs.get('baudrate', baud_)
        timeout_ = kwargs.get('timeout', 0.1)

        self._discovery(dev_, baud_, timeout_)

        return self.is_connected

    def _disconnect(self):
        """Close the connection"""  # noqa: DAR101,DAR201,DAR401
        self._hdl.reset_input_buffer()
        self._hdl.reset_output_buffer()
        self._hdl.close()
        self._hdl = None

    def _read(self, size, timeout=0):
        """Read data from the connected device"""  # noqa: DAR101,DAR201,DAR401
        return self._hdl.read(size)

    def _write(self, data, timeout=0):
        """Write data to the connected device"""  # noqa: DAR101,DAR201,DAR401
        return self._hdl.write(data)

    def short_desc(self):
        """Report a human description of the connection state"""  # noqa: DAR101,DAR201,DAR401
        desc = 'SERIAL:' + str(self._device) + ':' + str(self._baudrate)
        desc += ':connected' if self.is_connected else ':not connected'
        return desc


if __name__ == '__main__':
    pass
