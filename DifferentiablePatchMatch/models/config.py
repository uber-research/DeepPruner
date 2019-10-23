
# ---------------------------------------------------------------------------
# DeepPruner: Learning Efficient Stereo Matching via Differentiable PatchMatch
#
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Shivam Duggal
# ---------------------------------------------------------------------------

from __future__ import print_function

class obj(object):
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, (list, tuple)):
               setattr(self, key, [obj(x) if isinstance(x, dict) else x for x in value])
            else:
               setattr(self, key, obj(value) if isinstance(value, dict) else value)
               
config = {
    "patch_match_args": {
        # sample count refers to random sampling stage of generalized PM.
        # Number of random samples generated: (sample_count+1) * (sample_count+1)
        # we generate (sample_count+1) samples in x direction, and (sample_count+1) samples in y direction,
        # and then perform meshgrid like opertaion to generate (sample_count+1) * (sample_count+1) samples.
        "sample_count": 1,

        "iteration_count": 21,
        "propagation_filter_size": 3,
        "propagation_type": "faster_filter_3_propagation", # for better code for PM propagation, set it to None
        "softmax_temperature": 10000000000, # softmax temperature for evaluation. Larger temp. lead to sharper output.
        "random_search_window_size": [100,100], # search range around evaluated offsets after every iteration.
        "evaluation_type": "softmax"
    },

    "feature_extractor_filter_size": 7
 
}


config = obj(config)