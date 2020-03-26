# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import baselines.DaSiamRPN.code.vot as vot
from baselines.DaSiamRPN.code.vot import Rectangle
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join

from baselines.DaSiamRPN.code.net import SiamRPNBIG
from baselines.DaSiamRPN.code.run_SiamRPN import SiamRPN_init, SiamRPN_track
from baselines.DaSiamRPN.code.utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net: torch.nn.Module = SiamRPNBIG()
compute_device = torch.device("cuda:0")
net.load_state_dict(torch.load(net_file))
net = net.to(compute_device)
net.eval()

# warm up
for i in range(10):
    net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).to(compute_device))
    net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).to(compute_device))

# start to track
handle = vot.VOT("polygon")
Polygon = handle.region()
cx, cy, w, h = get_axis_aligned_bbox(Polygon)

image_file = handle.frame()
if not image_file:
    sys.exit(0)

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_file)  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net, compute_device)  # init tracker
while True:
    image_file = handle.frame()
    if not image_file:
        break
    im = cv2.imread(image_file)  # HxWxC
    state = SiamRPN_track(state, im, compute_device)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

    handle.report(Rectangle(res[0], res[1], res[2], res[3]))

