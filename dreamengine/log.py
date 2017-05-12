
import logging

# usage: instead of "logging.info()" do "log.basic.info()" or "log.xxx.info()"
basic = None


def init(logging_level):
    global basic
    basic = logging.getLogger('basic')
    basic.setLevel(logging_level)
    basic.addFilter(NoPyassimpFilter())
    #formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(module)s:%(filename)s:%(funcName)s:%(lineno)s: %(message)s')
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    # Log.stack = ... (can add stack trace? I guess could just throw error)
    file_handler = logging.FileHandler("dreamengine.log", mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    basic.addHandler(file_handler)
    basic.addHandler(stream_handler)


class NoPyassimpFilter(logging.Filter):
    # doesn't work
    def filter(self, record):
        return not record.module == 'pyassimp'
