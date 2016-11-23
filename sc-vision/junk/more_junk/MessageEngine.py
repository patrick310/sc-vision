import threading

INFO_IDENT = 'info'
ERROR_IDENT = 'err'

class MessageEngine:
    PRINT_LOCK = threading.Lock()

    def __init__(self, logfile_enabled = False, logfile_path = None):
        self.logfile_enabled = logfile_enabled
        self.logfile_path = logfile_path
        self.INFO_IDENT = 'info'
        self.ERROR_IDENT = 'err'
        self.INFO_PREPEND = "[INFO]: "
        self.ERROR_PREPEND = "[ERR]: "

    def _print_message(self, message, message_type):
        msg = message
        if not isinstance(message, str):
            msg = str(msg)

        if message_type == self.INFO_IDENT:
            msg = self.INFO_PREPEND + msg
        elif message_type == self.ERROR_IDENT:
            msg = self.ERROR_PREPEND + msg

        self.PRINT_LOCK.acquire()
        if self.logfile_enabled:
            msg += "\n"
            with open(self.logfile_path, "a") as f:
                f.write(msg)
                f.flush()
        else:
            print msg
        self.PRINT_LOCK.release()

    def handle_message(self, message, message_type):
        self._print_message(message, message_type)


# EOF
