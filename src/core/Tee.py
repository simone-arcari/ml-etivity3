import sys
import io
from contextlib import redirect_stdout

class Tee:
    def __init__(self, *writers):
        self.writers = writers

    def write(self, data):
        for w in self.writers:
            w.write(data)
            w.flush()  # forza lo stream

    def flush(self):
        for w in self.writers:
            w.flush()