import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


class DrawTool(object):
    def __init__(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self._save_path = save_path
        
    def heatmap(self, clusters, music_name):
            # 描画する
        fig, ax = plt.subplots(figsize=(8,6))
        heatmap = ax.pcolor(clusters)
    
        ax.set_xticks(np.arange(clusters.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(clusters.shape[0]) + 0.5, minor=False)
    
        ax.invert_yaxis()
        ax.xaxis.tick_top()
    
        ax.set_xticklabels(np.repeat("", clusters.shape[1]+1), minor=False)
        ax.set_yticklabels(np.arange(1, clusters.shape[0]+1), minor=False)
        
        img_folder = os.path.join(self._save_path, music_name)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
            
        img_path = os.path.join(img_folder, "cluster.png")
        plt.savefig(img_path)
        plt.close()
        # return heatmap
    
    def __call__(self, clusters, music_name):
        self.heatmap(clusters, music_name)