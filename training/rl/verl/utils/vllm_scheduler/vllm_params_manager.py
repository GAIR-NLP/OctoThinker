import os
from sortedcontainers import SortedDict

# {-1: 2048,
#         100: 4096,
#         300: 6144,
#         600: 8192,}
class StairRolloutParamsManager:
    def __init__(self, length_scheduler_dict):
        self.stair_length_scheduler_7b=SortedDict(length_scheduler_dict)
        pass

    def _get_value(self, stair_dict, step):
        index=stair_dict.bisect_right(step)-1
        key=stair_dict.iloc[index]
        return stair_dict[key]

    def get_rollout_params(self, kwargs, step):
        max_new_tokens=self._get_value(self.stair_length_scheduler_7b, step)
        kwargs["max_tokens"]=max_new_tokens
        return kwargs
