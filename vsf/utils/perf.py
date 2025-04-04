import time
import json
import os

import numpy as np


class PerfRecorder:
    """
    Utility class to record performance statistics.
    """
    def __init__(self, names_lst: list = []):
        """
        Initialize the PerfRecorder class.

        Args:
        names_lst (list): List of names to record performance statistics for.
        """
        self.perf_lst_dict: dict[str, list] = {}
        self.start_time_dict = {}

        self.reset(names_lst)

    def reset(self, names_lst: list):
        for name in names_lst:
            self.perf_lst_dict[name] = []
            self.start_time_dict[name] = None

    def start(self, name):
        if name not in self.perf_lst_dict:
            self.perf_lst_dict[name] = []
        self.start_time_dict[name] = time.time()
    
    def stop(self, name):
        start_time = self.start_time_dict[name]
        perf_time = time.time()-start_time
        self.perf_lst_dict[name].append(perf_time)

        self.start_time_dict[name] = None
    
    def add_val(self, name, val):
        if type(val) in [ np.int32, np.int64 ]:
            val = int(val)
        self.perf_lst_dict[name].append(val)
    
    def dump(self, stats_folder, fn='perf_stats.json'):
        summary_dict = {}
        for key in self.perf_lst_dict.keys():
            summary_dict[key+'_num'] = len(self.perf_lst_dict[key])
            summary_dict[key+'_sum'] = sum(self.perf_lst_dict[key])

        for key in self.perf_lst_dict.keys():
            summary_dict[key] = self.perf_lst_dict[key]

        with open(os.path.join(stats_folder, fn), 'w') as f:
            json.dump(summary_dict, f, indent=2)
        self.reset(self.perf_lst_dict.keys())

class DummyRecorder(PerfRecorder):
    """
    Dummy class that skips recording for code compatibility.
    """
    def __init__(self):
        pass

    def start(self, name):
        pass

    def stop(self, name):
        pass
    
    def dump(self, stats_folder, fn=None):
        pass
