"""logging module."""
import logging
import os

import coloredlogs

coloredlogs.DEFAULT_FIELD_STYLES = {
    "asctime": {"color": "green"},
    "hostname": {"color": "magenta"},
    "name": {"color": "yellow"},
    "levelname": {"bold": True, "color": "black", "faint": True},
    "module": {"color": "blue", "faint": True, },
    "lineno": {"color": "blue", "faint": True},
}

coloredlogs.install(level=os.environ.get("LOGLEVEL", "INFO").upper(),
                    milliseconds=True,
                    fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s %(module)s:%(lineno)d")

LOGGER = logging.getLogger()
