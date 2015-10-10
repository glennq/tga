#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modified from https://github.com/dfm/pcp/blob/master/demo.py

Uses data from http://perception.i2r.a-star.edu.sg/bk_model/bk_index.html
"""

from __future__ import division, print_function
import os
import time
import numpy as np
from PIL import Image

from tga import TGA


def bitmap_to_mat(bitmap_seq):
    """from blog.shriphani.com"""
    matrix = []
    shape = None
    for bitmap_file in bitmap_seq:
        img = Image.open(bitmap_file).convert("L")
        if shape is None:
            shape = img.size
        assert img.size == shape
        img = np.array(img.getdata())
        matrix.append(img)
    return np.array(matrix), shape[::-1]


def do_plot(ax, img, shape):
    ax.cla()
    ax.imshow(img.reshape(shape), cmap="gray", interpolation="nearest")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


if __name__ == "__main__":
    import sys
    import glob
    import matplotlib.pyplot as pl

    use_data = str(sys.argv[1])
    M, shape = bitmap_to_mat(glob.glob(use_data + "/*.bmp")[:2000:2])
    print(M.shape)

    tga = TGA(n_components=5, random_state=1)
    start_time = time.time()
    tga.fit(M)
    print("fitted, time taken {0}s".format(time.time() - start_time))
    start_time = time.time()
    transformed = tga.transform(M)
    L = tga.inverse_transform(transformed)
    print('calculated L, time taken {0}s'.format(time.time() - start_time))
    S = M - L

    if not os.path.exists('results_tga'):
        os.makedirs('results_tga')

    directory = "results_tga/" + use_data
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, axes = pl.subplots(1, 3, figsize=(10, 4))
    fig.subplots_adjust(left=0, right=1, hspace=0, wspace=0.01)
    for i in range(min(len(M), 500)):
        do_plot(axes[0], M[i], shape)
        axes[0].set_title("raw")
        do_plot(axes[1], L[i], shape)
        axes[1].set_title("low rank")
        do_plot(axes[2], S[i], shape)
        axes[2].set_title("sparse")
        fig.savefig("results_tga/" + use_data + "/{0:05d}.png".format(i))
