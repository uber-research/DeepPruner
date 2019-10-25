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
    "max_disp": 192, #*2 for using DeepPruner-fast pre-trained model,
    "cost_aggregator_scale": 4, # for DeepPruner-fast change this to 8.
    "mode": "training", # for evaluation/ submission, change this to evaluation.

    
    # The code allows the user to change the feature extrcator to any feature extractor of their choice.
    # The only requirements of the feature extractor are:
    #     1.  For cost_aggregator_scale == 4:
    #             features at downsample-level X4 (feature_extractor_ca_level) 
    #             and downsample-level X2 (feature_extractor_refinement_level) should be the output.
    #         For cost_aggregator_scale == 8:
    #             features at downsample-level X8 (feature_extractor_ca_level),
    #             downsample-level X4 (feature_extractor_refinement_level),
    #             downsample-level X2 (feature_extractor_refinement_level_1) should be the output, 
        
    #     2.  If the feature extractor is modified, change the "feature_extractor_outplanes_*" key in the config
    #         accordingly.

    "feature_extractor_ca_level_outplanes": 32,
    "feature_extractor_refinement_level_outplanes": 32, # for DeepPruner-fast change this to 64.
    "feature_extractor_refinement_level_1_outplanes": 32,

    "patch_match_args": {
        "sample_count": 12,
        "iteration_count": 2,
        "propagation_filter_size": 3
    },

    "post_CRP_sample_count": 7,
    "post_CRP_sampler_type": "uniform", #change to patchmatch for Sceneflow model. 

    "hourglass_inplanes": 16
}

config = obj(config)
