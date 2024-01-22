from simple_knn._C import distCUDA2

import gsstudio


@gsstudio.register("dynamic-base-model")
class DynamicBaseModel:
    moment = None
    time_index = None

    def set_time(self, moment=None, time_index=None):
        self.moment = moment
        self.time_index = time_index
