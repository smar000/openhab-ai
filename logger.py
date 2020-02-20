import logging
import textwrap

#Logging 
class CustomLoggingFormatter(logging.Formatter):
    data = {}

    def __init__(self):
        super(CustomLoggingFormatter, self).__init__(fmt="%(asctime)s [%(levelname)-5s %(lineno)-3d] ",datefmt="%Y-%m-%d %H:%M:%S")
		# - %(threadName)-10s 
    def format(self, record):
        header = super(CustomLoggingFormatter, self).format(record)
        msg = textwrap.indent(record.message, " " * len(header)).strip()
        return header + msg


log = logging.getLogger(__name__)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = CustomLoggingFormatter()
handler.setFormatter(formatter)

logger.addHandler(handler)