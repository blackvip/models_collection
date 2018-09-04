import time
import psutil
from contextlib import contextmanager

class multi_layer_timer:
    def __init__(self):
        self.timer_depth = -1

    @contextmanager
    def timer(self, name):
        t0 = time.time()
        self.timer_depth += 1
        yield
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30
        print('----'*self.timer_depth + f'>>[{name}] done in {time.time() - t0:.0f} s ---> memory used: {memoryUse:.4f} GB', '')
        if(self.timer_depth == 0):
            print('\n')
        self.timer_depth -= 1