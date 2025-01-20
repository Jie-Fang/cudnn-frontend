import sys
import os

class Logger(object):
    def __init__(self, logfile, append, disable_stdio_print):
        self.terminal = sys.stdout

        if logfile:
            if append:
                self.log = open(logfile, "a", buffering=1)
            else:
                self.log = open(logfile, "w", buffering=1)
        else:
            self.log = None

        self.disable_stdio_print = disable_stdio_print

        self.intercept = None

    def write(self, message):
        if not self.disable_stdio_print:
            self.terminal.write(message)

        if self.intercept != None:
            self.intercept += message

        if self.log:
            self.log.write(message)

    def start_intercept(self):

        self.intercept = ""

    def end_intercept(self):
        result = self.intercept

        self.intercept = None

        return result

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
