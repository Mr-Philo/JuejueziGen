# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

Copyright (c) 2022 ByteDance.com, Inc. All Rights Reserved
测试训练好的打分模型。

Author: pankeyu(pankeyu@bytedance.com)
Modified by: ruizhewang
Date: 2023/08/31
"""
import mindspore
from rich import print
from mindnlp.transforms import BertTokenizer
from mindnlp.models import BertModel
from ms_model import RewardModel
import numpy as np

encoder = BertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = RewardModel(encoder)
param_dict = mindspore.load_checkpoint('./checkpoint/best_so_far.ckpt')
# param_dict = mindspore.load_checkpoint('./checkpoint/juejuezi_reward_model_epoch_9.ckpt')
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
# model.eval()

texts = [
    '买过很多箱这个苹果了，一如既往的好，汁多味甜～',
    '一台充电很慢，信号不好！退了！又买一台竟然是次品。。服了。。',
    '近日，我市一家棉花纺织厂突然发生爆炸，所幸并无人员伤亡',
    '和集美逛吃的一天 什么神仙宝藏 又去做数据集啦 心态炸裂 呜呜呜 这件数据集好做到跺jiojio 介个小蛋糕也太带感了~~~'
]
# inputs = tokenizer(texts, max_length=128,padding='max_length', return_tensors='pt')

for text in texts:
    inputs = tokenizer(np.array(text)).reshape(1, -1)
    r = model(mindspore.Tensor(inputs))
    print(r)