from mmdet.core import AnchorGenerator

self = AnchorGenerator([16], [1.],[1.], [9])
all_anchors = self.grid_priors([(2, 2)], device='cpu')
print(all_anchors)

self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
all_anchors = self.grid_priors([(2, 2), (1, 1)], device='cpu')
print(all_anchors)