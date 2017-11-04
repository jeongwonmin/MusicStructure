import logging
import argparse

from music_reader import MusicReader
from cluster_tool import ClusterTool
from draw_tool import DrawTool


parser = argparse.ArgumentParser(description="music folder and result image folder",
                                 allow_abbrev=True)
parser.add_argument("--music_folder", "-mf", help="music folder")
parser.add_argument("--output_folder", "-of", help="output folder")
args = parser.parse_args()

if __name__=="__main__":
    # mr = MusicReader("/Users/MIN/my_workspace/librosa_app/librosa/music_files")
    mr = MusicReader(args.music_folder)
    clt = ClusterTool(17)
    # drt = DrawTool("/Users/MIN/my_workspace/librosa_app/MusicVisualization/clusters")
    drt = DrawTool(args.output_folder)
    for music in mr():
        drt(clt(music), music[0])
        logging.info("cluster heatmap of " + music[0] + " was drawn" )
