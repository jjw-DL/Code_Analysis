import matplotlib.pyplot as plt, numpy as np
from ..data_protos import BBox


class Visualizer2D:
    def __init__(self, name='', figsize=(8, 8)):
        self.figure = plt.figure(name, figsize=figsize) # 创建画布
        plt.axis('equal')
        self.COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_blue': np.array([4, 157, 217]) / 256,
            'red': np.array([191, 4, 54]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 115, 67]) / 256
        } # 初始化颜色映射
    
    def show(self):
        plt.show()
    
    def close(self):
        plt.close()
    
    def save(self, path):
        plt.savefig(path)
    
    def handler_pc(self, pc, color='gray'):
        vis_pc = np.asarray(pc)
        # 绘制点云x和y坐标
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], marker='o', color=self.COLOR_MAP[color], s=0.01)
    
    def handler_box(self, box: BBox, message: str='', color='red', linestyle='solid'):
        corners = np.array(BBox.box2corners2d(box))[:, :2] # (4, 2) 四个角点坐标
        corners = np.concatenate([corners, corners[0:1, :2]]) # 拼接第一个角点，形成闭合
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle) # 画线
        corner_index = np.random.randint(0, 4, 1) # 随机选择文字放置的位置
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[color]) # 绘制文字