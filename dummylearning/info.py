import time

class Info:


    def __init__(self, verbose):

        self.report = dict()
        self.verbose = verbose


    def upgradeInfo(self, message):

        if self.verbose:
            message = f"{type(self).__name__}: {message}"
            self.report[time.time()] = message
            print(message)
