# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rcnn.config import config
import numpy as np


def compute_assign_targets(rois, threshold):
    rois_area = np.sqrt((rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
    num_rois = np.shape(rois)[0]
    assign_levels = np.zeros(num_rois, dtype=np.uint8)
    for i, stride in enumerate(config.RCNN_FEAT_STRIDE):
        thd = threshold[i]
        idx = np.logical_and(thd[1] <= rois_area, rois_area < thd[0])
        assign_levels[idx] = stride

    assert 0 not in assign_levels, "All rois should assign to specify levels."
    return assign_levels


def add_assign_targets(roidb):
    """
    given roidb, add ['assign_level']
    :param roidb: roidb to be processed. must have gone through imdb.prepare_roidb
    """
    print 'add assign targets'
    assert len(roidb) > 0
    assert 'boxes' in roidb[0]

    area_threshold = [[np.inf, 448],
                      [448,    224],
                      [224,    112],
                      [112,     0]]

    assert len(config.RCNN_FEAT_STRIDE) == len(area_threshold)

    num_images = len(roidb)
    for im_i in range(num_images):
        rois = roidb[im_i]['boxes']
        roidb[im_i]['assign_levels'] = compute_assign_targets(rois, area_threshold)
