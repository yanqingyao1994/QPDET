import re
import os
import time
import warnings

import os.path as osp
import numpy as np

from PIL import Image
from functools import reduce, partial
from multiprocessing import Pool
from collections import defaultdict

from .io import load_imgs
from .misc import get_classes, img_exts
from ..utils import get_bbox_type
from ..geometry import bbox2type


def load_dota(img_dir, ann_dir=None, classes=None, nproc=10):
    classes = get_classes('DOTA' if classes is None else classes)
    cls2lbl = {cls: i for i, cls in enumerate(classes)}

    print('Starting loading DOTA dataset information.')
    start_time = time.time()
    _load_func = partial(_load_dota_single,
                        img_dir=img_dir,
                        ann_dir=ann_dir,
                        cls2lbl=cls2lbl)
    if nproc > 1:
        pool = Pool(nproc)
        contents = pool.map(_load_func, os.listdir(img_dir))
        pool.close()
    else:
        contents = list(map(_load_func, os.listdir(img_dir)))
    contents = [c for c in contents if c is not None]
    end_time = time.time()
    print(f'Finishing loading DOTA, get {len(contents)} iamges,',
          f'using {end_time-start_time:.3f}s.')

    return contents, classes


def _load_dota_single(imgfile, img_dir, ann_dir, cls2lbl):
    img_id, ext = osp.splitext(imgfile)
    if ext not in img_exts:
        return None

    imgpath = osp.join(img_dir, imgfile)
    size = Image.open(imgpath).size
    txtfile = None if ann_dir is None else osp.join(ann_dir, img_id+'.txt')
    content = _load_dota_txt(txtfile, cls2lbl)

    content.update(dict(width=size[0], height=size[1], filename=imgfile, id=img_id))
    return content


def _load_dota_txt(txtfile, cls2lbl):
    gsd, bboxes, labels, diffs = None, [], [], []
    if txtfile is None:
        pass
    elif not osp.exists(txtfile):
        warnings.warn(f"Can't find {txtfile}, treated as empty txtfile")
    else:
        with open(txtfile, 'r') as f:
            for line in f:
                if line.startswith('gsd'):
                    num = line.split(':')[-1]
                    try:
                        gsd = float(num)
                    except ValueError:
                        gsd = None
                    continue

                items = line.split(' ')
                if len(items) >= 9:
                    if items[8] not in cls2lbl:
                        continue
                    bboxes.append([float(i) for i in items[:8]])
                    labels.append(cls2lbl[items[8]])
                    diffs.append(int(items[9]) if len(items) == 10 else 0)

    bboxes = np.array(bboxes, dtype=np.float) if bboxes else \
            np.zeros((0, 8), dtype=np.float)
    labels = np.array(labels, dtype=np.int) if labels else \
            np.zeros((0, ), dtype=np.int)
    diffs = np.array(diffs, dtype=np.int) if diffs else \
            np.zeros((0, ), dtype=np.int)
    ann = dict(bboxes=bboxes, labels=labels, diffs=diffs)
    return dict(gsd=gsd, ann=ann)


def load_dota_submission(ann_dir, img_dir=None, classes=None, nproc=10):
    classes = get_classes('DOTA' if classes is None else classes)

    file_pattern = r'Task[1|2]_(.*)\.txt'
    cls2file_mapper = dict()
    for f in os.listdir(ann_dir):
        match_objs = re.match(file_pattern, f)
        if match_objs is None:
            fname, _ = osp.splitext(f)
            cls2file_mapper[fname] = f
        else:
            cls2file_mapper[match_objs.group(1)] = f

    print('Starting loading DOTA submission information')
    start_time = time.time()
    infos_per_cls = []
    for cls in classes:
        if cls not in cls2file_mapper:
            infos_per_cls.append(dict())
        else:
            subfile = osp.join(ann_dir, cls2file_mapper[cls])
            infos_per_cls.append(_load_dota_submission_txt(subfile))

    if img_dir is not None:
        contents, _ = load_imgs(img_dir, nproc=nproc, def_bbox_type='poly')
    else:
        all_id = reduce(lambda x, y: x|y, [d.keys() for d in infos_per_cls])
        contents = [{'id':i} for i in all_id]

    for content in contents:
        bboxes, scores, labels = [], [], []
        for i, infos_dict in enumerate(infos_per_cls):
            infos = infos_dict.get(content['id'], dict())
            num_bboxes = infos.get('bboxes', np.zeros((0, 8))).shape[0]

            bboxes.append(infos.get('bboxes', np.zeros((0, 8), dtype=np.float)))
            scores.append(infos.get('scores', np.zeros((0, ), dtype=np.float)))
            labels.append(np.zeros((num_bboxes, ), dtype=np.int) + i)

        bboxes = np.concatenate(bboxes, axis=0)
        labels = np.concatenate(labels, axis=0)
        scores = np.concatenate(scores, axis=0)
        content['ann'] = dict(bboxes=bboxes, labels=labels, scores=scores)
    end_time = time.time()
    print(f'Finishing loading DOTA submission, get{len(contents)} images,',
          f'using {end_time-start_time:.3f}s.')
    return contents, classes


def _load_dota_submission_txt(subfile):
    if not osp.exists(subfile):
        warnings.warn(f"Can't find {subfile}, treated as empty subfile")
        return dict()

    collector = defaultdict(list)
    with open(subfile, 'r') as f:
        for line in f:
            img_id, score, *bboxes = line.split(' ')
            bboxes_info = bboxes + [score]
            bboxes_info = [float(i) for i in bboxes_info]
            collector[img_id].append(bboxes_info)

    anns_dict = dict()
    for img_id, info_list in collector.items():
        infos = np.array(info_list, dtype=np.float)
        bboxes, scores = infos[:, :-1], infos[:, -1]
        bboxes = bbox2type(bboxes, 'poly')
        anns_dict[img_id] = dict(bboxes=bboxes, scores=scores)
    return anns_dict


def save_dota_submission(save_dir, id_list, dets_list, task='Task1', classes=None):
    assert task in ['Task1', 'Task2']
    classes = get_classes('DOTA' if classes is None else classes)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    files = [open(osp.join(save_dir ,task+'_'+cls+'.txt'), 'w') for cls in classes]
    for img_id, dets_per_cls in zip(id_list, dets_list):
        for f, dets in zip(files, dets_per_cls):
            bboxes, scores = dets[:, :-1], dets[:, -1]

            if task == 'Task1':
                if get_bbox_type(bboxes) == 'poly' and bboxes.shape[-1] != 8:
                    bboxes = bbox2type(bboxes, 'obb')
                bboxes = bbox2type(bboxes, 'poly')
            else:
                bboxes = bbox2type(bboxes, 'hbb')

            for bbox, score in zip(bboxes, scores):
                txt_element = [img_id, str(score)] + ['%.2f'%(p) for p in bbox]
                f.writelines(' '.join(txt_element)+'\n')

    for f in files:
        f.close()
