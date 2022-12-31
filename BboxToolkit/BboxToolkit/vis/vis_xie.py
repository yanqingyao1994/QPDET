from multiprocessing import Pool
import os.path as osp
import BboxToolkit as bt



# DOTA1': ('large-vehicle  (LV)', 'swimming-pool  (SP)', 'helicopter  (HC)', 'bridge  (BR)',
#               'plane  (PL)', 'ship   (SH)', 'soccer-ball-field   (SBF)', 'basketball-court  (BC)',
#               'ground-track-field  (GTF)', 'small-vehicle  (SV)', 'baseball-diamond  (BD)','tennis-court  (TC)',
#               'roundabout  (RA)', 'storage-tank  (ST)', 'harbor  (HA)')


img_dir = '/disk0/evannnnnnn/home/Datasets/DOTA/train/images/p0775.png'
ann_dir = '/disk0/evannnnnnn/home/Datasets/DOTA/train/labelTxt/p0775.txt'
save_dir = '/disk2/xiexingxing/wjb/vis/base_wo_merge_xie2'
colors = [(0,128,255),(116,115,147),(255,0,0),(0,255,0),
          (42,42,165), (255,0,255),(205,250,255),(193,193,255),
          (0,0,255),(226,43,138), (107,183,189), (255,255,0),
          (139,139,0),(153,51,0),(0,255,255) ] 

infos, classes = bt.load_dota_submission(ann_dir)
colors = colors[:len(classes)]

# img_dir = '/disk2/xiexingxing/wjb/HRSC2016/FullDataSet/AllImages/'
# ann_dir = '/disk2/xiexingxing/wjb/results/sp_orpn/hrsc.pkl'
# save_dir = '/disk2/xiexingxing/wjb/vis/hrsc'
# infos, classes = bt.load_pkl(ann_dir)
# colors = 'green'

def vis(info):
    print(info['id'])
    bboxes = info['ann']['bboxes']
    labels = info['ann']['labels']
    scores = info['ann']['scores']

    bboxes = bboxes[scores > 0.3]
    labels = labels[scores > 0.3]
    scores = scores[scores > 0.3]
    if len(bboxes) == 0:
        return 
    bboxes = [bboxes[labels == i] for i in range(len(classes))]

    bt.imshow_bboxes(
        osp.join(img_dir, info['id'] + '.png'),
        bboxes,
        colors=colors,
        show=False,
        thickness=3,
        out_file=osp.join(save_dir, info['id'] + '.png')
    )

pool = Pool(5)
pool.map(vis, infos)
pool.close()
# #def vis(info):
# info=infos[0]
# print(info['id'])
# bboxes = info['ann']['bboxes']
# labels = info['ann']['labels']
# scores = info['ann']['scores']

# bboxes = bboxes[scores > 0.3]
# labels = labels[scores > 0.3]
# scores = scores[scores > 0.3]
# # if len(bboxes) == 0:
# #     return
# bboxes = [bboxes[labels == i] for i in range(len(classes))]

# bt.imshow_bboxes(
#     osp.join(img_dir, info['id'] + '.png'),
#     bboxes,
#     colors=colors,
#     show=False,
#     thickness=3,
#     out_file=osp.join(save_dir, info['id'] + '.png')
# )    

# # pool = Pool(5)
# # pool.map(vis, infos)
# # pool.close()
        

