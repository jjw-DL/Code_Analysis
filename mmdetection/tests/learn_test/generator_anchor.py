import numpy as np
from mmcv.visualization import imshow_bboxes
import matplotlib.pyplot as plt
from torch import strided
from mmdet.core import build_anchor_generator
from mmdet.core import anchor
from mmdet.core.anchor import anchor_generator

if __name__ == '__main__':
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        octave_base_scale=4,
        scales_per_octave=3,
        ratios=[0.5, 1, 2],
        strides=[8, 16, 32, 64, 128]
    )
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    base_anchors = anchor_generator.base_anchors[0]

    h = 100
    w = 160
    img = np.ones([h, w, 3], np.uint8) * 255
    base_anchors[:, 0::2] += w // 2
    base_anchors[:, 1::2] += h // 2

    colors = ['green', 'red', 'blue']
    for i in range(3):
        base_anchor = base_anchors[i::3, :].cpu().numpy()
        imshow_bboxes(img, base_anchor, show=False, colors=colors[i])
    
    plt.grid()
    plt.imshow(img)
    plt.show()


