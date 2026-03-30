###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
Entry point to register a driver

Expected syntax for driver description/options for the connection

    [<domain>[:<option1>[:option2]]]

    domain:
        - None or 'serial' (default)
        - 'file' or valid file_path/directory
        - 'socket'

"""

import os


_DEFAULT_DOMAIN = 'serial'
_FILE_DOMAIN = 'file'
_SOCKET_DOMAIN = 'socket'


def _default_resolver(domain, _):
    """Default resolver function"""  # noqa: DAR101,DAR201,DAR401
    if domain == _DEFAULT_DOMAIN or domain is None:
        return True
    return False


def _default_create(parent, desc):
    """Default create function"""  # noqa: DAR101,DAR201,DAR401
    from .serial_hw_drv import SerialHwDriver
    from .pb_mgr_drv import AiPbMsg

    return AiPbMsg(parent, SerialHwDriver()), desc


def _dll_resolver(domain, desc=None):
    """Default resolver function"""  # noqa: DAR101,DAR201,DAR401
    from .dll_mgr_drv import AiDllDriver

    if domain == _FILE_DOMAIN and desc is not None:
        res, _ = AiDllDriver.is_valid(desc)
        return res
    return False


def _dll_create(parent, desc):
    """Default create function"""  # noqa: DAR101,DAR201,DAR401
    from .dll_mgr_drv import AiDllDriver

    _, new_desc = AiDllDriver.is_valid(desc)

    return AiDllDriver(parent), new_desc


def _socket_resolver(domain, _):
    """Socket resolver function"""  # noqa: DAR101,DAR201,DAR401
    if domain == _SOCKET_DOMAIN:
        return True
    return False


def _socket_create(parent, desc):
    """Default create function"""  # noqa: DAR101,DAR201,DAR401
    from .socket_hw_drv import SocketHwDriver
    from .pb_mgr_drv import AiPbMsg

    return AiPbMsg(parent, SocketHwDriver()), desc


_DRIVERS = {
    _DEFAULT_DOMAIN: (_default_resolver, _default_create),  # default
    _FILE_DOMAIN: (_dll_resolver, _dll_create),
    _SOCKET_DOMAIN: (_socket_resolver, _socket_create),
}


def ai_runner_resolver(parent, desc):
    """Return drv instance"""  # noqa: DAR101,DAR201,DAR401

    if desc is not None and isinstance(desc, str):
        desc = desc.strip()
        split_ = desc.split(':')
        nb_elem = len(split_)

        domain = None
        # only valid file or directory is passed
        if os.path.isfile(desc) or os.path.isdir(desc) or os.path.isdir(os.path.dirname(desc)):
            domain = _FILE_DOMAIN
        # domain is considered
        elif nb_elem >= 1 and len(split_[0]) > 0:
            domain = split_[0].lower()
            desc = desc[len(domain + ':'):]
        # ':' is passed as first character
        elif len(split_[0]) == 0:
            domain = _DEFAULT_DOMAIN
            desc = desc[1:]

        for drv_ in _DRIVERS:
            if _DRIVERS[drv_][0](domain, desc):
                return _DRIVERS[drv_][1](parent, desc)

        err_msg_ = f'invalid/unsupported "{domain}:{desc}" descriptor'
        return None, err_msg_

    else:
        return _DRIVERS[_DEFAULT_DOMAIN][1](parent, desc)
