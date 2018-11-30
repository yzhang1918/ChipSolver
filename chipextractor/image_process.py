import numpy as np
from PIL import Image
import os
from collections import Counter

from chipextractor.utils import *


class Dataset:

    def __init__(self, root, suffix='PNG', outlier_threshold=10, *args, **kwargs):
        self.data = []
        for path, _, flist in os.walk(root):
            for fname in flist:
                if fname.endswith(suffix):
                    filename = os.path.join(path, fname)
                    scrshot = Screenshot.from_file(filename, *args, **kwargs)
                    scrshot.filename = filename
                    self.data.append(scrshot)
        _, self.info = self.collective_refine_inplace(self.data, outlier_threshold)

    @property
    def colorful_chips(self):
        chip_list = []
        for scrshot in self:
            chip_list.extend(scrshot.colorful_chips)
        return chip_list

    @property
    def chips(self):
        chip_list = []
        for scrshot in self:
            chip_list.extend(scrshot.chips)
        return chip_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def collective_refine_inplace(screenshots, outlier_threshold=10):
        height_counter = Counter()
        width_counter = Counter()
        for scrshot in screenshots:
            height_counter.update(scrshot.coords[:, 2])
            width_counter.update(scrshot.coords[:, 3])
        h = height_counter.most_common(1)[0][0]
        w = width_counter.most_common(1)[0][0]
        for scrshot in screenshots:
            _ = scrshot.refine_segments(h, w, outlier_threshold)
        info = {'height_counter': height_counter, 'width_counter': width_counter}
        return info


class Screenshot:

    _weights = np.array([.2125, .7152, .0722])  # RGB2Gray

    def __init__(self, raw_img, resize_height=800,
                 threshold=.35, fill_max_margin=5,
                 min_interval_length=5):
        w, h = raw_img.size
        resize_w = int(w * resize_height // h)
        raw_img = raw_img.resize([resize_w, resize_height], Image.ANTIALIAS)
        self.raw_img = raw_img
        img = np.array(raw_img, dtype=float) / 255.
        img = img @ self._weights
        self.img = img
        _, self.coords = self.extract(threshold, fill_max_margin, min_interval_length)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, item):
        x, y, h, w = self.coords[item]
        return self.img[x:x+h, y:y+w].copy()

    @property
    def colorful_chips(self):
        chip_list = []
        for x, y, h, w in self.coords:
            chip_list.append(np.array(self.raw_img)[x:x+h, y:y+w])
        return chip_list

    @property
    def chips(self):
        chip_list = []
        for x, y, h, w in self.coords:
            chip_list.append(self.img[x:x+h, y:y+w].copy())
        return chip_list

    @classmethod
    def from_file(cls, fname, *args, **kwargs):
        raw_img = Image.open(fname)
        return cls(raw_img, *args, **kwargs)

    def extract(self, threshold=.35, fill_max_margin=5, min_interval_length=5):
        strips, indices_v = extract_strips(self.img, threshold, fill_max_margin)
        chips = []
        coords = []
        for strip, (y0, y1) in zip(strips, indices_v):
            sub_chips, indices_h = split_strip(strip, threshold,
                                               fill_max_margin,
                                               min_interval_length)
            for chip, (x0, x1) in zip(sub_chips, indices_h):
                chips.append(chip)
                coords.append([x0, y0, x1-x0, y1-y0])
        coords = np.asarray(coords)
        return chips, coords

    def refine_segments(self, height, width, outlier_threshold=10):
        diff = self.coords[:, 2:] - np.array([height, width])
        invalid_idx = np.any(diff > outlier_threshold, axis=1)
        self.coords = self.coords[~invalid_idx].copy()
        self.coords[:, 2] = height
        self.coords[:, 3] = width
        return self.chips


def extract_strips(img, threshold=.35, fill_max_margin=5):
    h, w = img.shape
    t, b = int(h * .2), int(h * .8)
    mask_arr = img[t:b].min(axis=0) > threshold
    mask_arr = fill_blanks(mask_arr, max_margin=fill_max_margin)
    intervals = consecutive(np.where(~mask_arr)[0])
    strips = []
    indices = []
    for a, *_, b in intervals[1:-1]:
        strips.append(img[:, a:b + 1].copy())
        indices.append((a, b+1))
    return strips, indices


def split_strip(strip, threshold=.35, fill_max_margin=5, min_length=5):
    mask_arr = fill_blanks(strip.min(axis=1) > threshold, fill_max_margin)
    intervals = consecutive(np.where(mask_arr)[0])
    # filter out small intervals (noise)
    intervals = [x for x in intervals if len(x) > min_length]
    # find the middle margin
    midline_idx = np.argmin(np.abs([x[len(x) // 2] - strip.shape[0] // 2
                                    for x in intervals]))
    if midline_idx == 0:
        # bottom chip
        t2 = intervals[0][-1] + 1
        b2 = intervals[1][0]
        h = b2 - t2
        # top chip
        b1 = intervals[0][0]
        t1 = b1 - h
        chips = [strip[t2:b2]]
        indices = [(t2, b2)]
        if t1 >= 0:  # out of boundary?
            chips.append(strip[t1:b1])
            indices.append((t1, b1))
        chips = chips[::-1]
        indices = indices[::-1]
    elif midline_idx > 0:
        # top chip
        t1 = intervals[midline_idx - 1][-1] + 1
        b1 = intervals[midline_idx][0]
        h = b1 - t1
        # bottom chip
        t2 = intervals[midline_idx][-1] + 1
        b2 = t2 + h
        chips = [strip[t1:b1]]
        indices = [(t1, b1)]
        if b2 <= strip.shape[0]:
            chips.append(strip[t2:b2])
            indices.append((t2, b2))
    else:
        raise NotImplementedError
    return chips, indices
