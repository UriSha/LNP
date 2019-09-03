import os
import time


class Logger():
    def __init__(self):
        files_timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        log_dir = os.path.join(cur_dir, "..", "logs", files_timestamp)
        os.makedirs(log_dir)
        self.log_file = open(os.path.join(log_dir, "log.txt"), "w")
        self.log_dir = log_dir

    def log(self, *args, **kwargs):
        time_prefix = f"[{time.strftime('%H:%M:%S', time.localtime())}]"
        print(time_prefix, *args, **kwargs)
        print(time_prefix, *args, **kwargs, file=self.log_file)
        self.log_file.flush()
