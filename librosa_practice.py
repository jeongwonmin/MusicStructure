# -*- coding:utf-8 -*-

import os
import librosa
import librosa.display
import librosa.core
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='オーディオファイルのパスを入力してください')

parser.add_argument('in_file', type=str, help='入力ファイルパス')
args = parser.parse_args()

def visualize_music(in_path, out_path):
    y, sr = librosa.load(in_path)
    #D = librosa.stft(y, win_length=2048)
    C = librosa.cqt(y, sr)
    CQT = librosa.amplitude_to_db(C, ref=np.max)

    plt.figure(figsize=(25, 7))
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')

    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    save_folder = os.path.dirname(out_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(out_path)

if os.path.isfile(args.in_file):
    basename = os.path.basename(args.in_file)
    filename = os.path.splitext(basename)[0]
    visualize_music(args.in_file, filename + '.png')

else:
    for dirname in os.listdir(args.in_file):
        dirname_ = os.path.join(args.in_file, dirname)
        if not os.path.isdir(dirname_):
            continue
        for filename in os.listdir(dirname_):
            if filename.find(".mp3") == -1:
                continue
            in_file = os.path.join(args.in_file, dirname, filename)
            save_folder = os.path.join("visualized", dirname)
            name = filename.split(".")[0]
            save_name = os.path.join(save_folder, name + '.png')
            visualize_music(in_file, save_name)
