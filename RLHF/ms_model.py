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

Reward Model类。

Author: pankeyu
Modified by: ruizhewang
Date: 2023/08/31
"""
from typing import List

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy


class RewardModel(nn.Cell):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (mindnlp.models): backbone, 默认使用 bert
        """
        super().__init__()
        self.encoder = encoder
        self.reward_layer = nn.Dense(768, 1)

    def construct(
        self,
        input_ids: mindspore.Tensor,
        token_type_ids=None,
        attention_mask=None,
        pos_ids=None,
    ) -> mindspore.Tensor:
        """
        constuct 函数（对应pytorch中的forward），返回每句话的得分值。

        Args:
            input_ids (mindspore.Tensor): (batch, seq_len)
            token_type_ids (mindspore.Tensor): (batch, seq_len)
            attention_mask (mindspore.Tensor): (batch, seq_len)
            pos_ids (mindspore.Tensor): (batch, seq_len)

        Returns:
            reward: (batch, 1)
        """
        pooler_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )[1]                              # (batch, hidden_size)  # <class 'mindspore.common._stub_tensor.StubTensor'>
        # 不加[1]的话，pooler_output是一个元组: ( (batch, seq_len, hidden_size) , (batch, hidden_size) )
        # print('pooler_output: ', pooler_output, ' type(pooler_output): ', type(pooler_output))    #debug
        reward = self.reward_layer(pooler_output)       # (batch, 1)
        return reward


class RankListLoss(nn.Cell):
    '''自定义有序列表rank loss'''
    
    def __init__(self, auto_prefix=True, flags=None):
        super().__init__(auto_prefix, flags)
        
        
    def construct(self, base, target):
        # 简要版本：转为有标签学习，标签是1表示绝绝子！表现为0表示正常语句
        loss = (base - target).mean()
        return loss
    

def compute_rank_list_loss(rank_rewards_list: List[List[mindspore.Tensor]], device='cpu') -> mindspore.Tensor:
    """
    通过给定的有序（从高到低）的ranklist的reward列表，计算rank loss。
    所有排序高的句子的得分减去排序低的句子的得分差的总和，并取负。

    Args:
        rank_rewards_list (mindspore.Tensor): 有序（从高到低）排序句子的reward列表，e.g. -> 
                                        [
                                            [mindspore.Tensor([0.3588]), mindspore.Tensor([0.2481]), ...],
                                            [mindspore.Tensor([0.5343]), mindspore.Tensor([0.2442]), ...],
                                            ...
                                        ]
        device (str): 使用设备
    
    Returns:
        loss (mindspore.Tensor): tensor([0.4891], grad_fn=<DivBackward0>)
    """
    if type(rank_rewards_list) != list:
        raise TypeError(f'@param rank_rewards expected "list", received {type(rank_rewards)}.')
    
    loss, add_count = mindspore.Tensor([0.0]).to(device), 0
    for rank_rewards in rank_rewards_list:
        for i in range(len(rank_rewards)-1):                                   # 遍历所有前项-后项的得分差
            for j in range(i+1, len(rank_rewards)):
                #! TODO: 明确这一步的返回loss的类型。必须要能够在mindspore框架下进行反向传播
                diff = ops.functional.logsigmoid(rank_rewards[i] - rank_rewards[j])         # sigmoid到0~1之间
                loss = loss + diff
                add_count += 1
    loss = loss / add_count
    loss = (-1) * loss
    print(type(loss))       # debug
    return loss                                                               # 要最大化分差，所以要取负数


if __name__ == '__main__':
    from rich import print
    # from transformers import AutoModel, AutoTokenizer
    from mindnlp.models import ErnieModel, BertModel
    from mindnlp.transforms import ErnieTokenizer, BertTokenizer

    encoder = BertModel.from_pretrained('bert-base-chinese')
    model = RewardModel(encoder)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    batch_texts = [
                ['这是一个测试句子1。', '这是一个测试句子2。', '这是一个测试句子3。','这是一个测试句子4。'],
                ['这是一个测试句子5。', '这是一个测试句子6。', '这是一个测试句子7。','这是一个测试句子8。'],
        ]

    rank_rewards = []

    for texts in batch_texts:
        tmp = []
        for text in texts:
            # inputs = tokenizer(text, return_tensors='pt')
            # r = model(**inputs)
            
            #! debug: reshape tokenized result to (1,-1), equals to changing 1-D tensor to 2-D tensor
            inputs = tokenizer(numpy.array(texts)).reshape(1, -1)   # inputs:  [[ 101  138  112 6821 3221  671  702 3844 6407 1368 2094  122  511  112 112 6821 3221  671  702 3844 6407 1368 2094  123  511  112  112 6821 3221  671  702 3844 6407 1368 2094  124  511  112  112 6821 3221  671 702 3844 6407 1368 2094  125  511  112  140  102]]  type(inputs):  <class 'numpy.ndarray'>
            # print('inputs: ', inputs, ' type(inputs): ', type(inputs))
            
            r = model(mindspore.Tensor(inputs))    # r:  [[-0.03215131]]  type(r):  <class 'mindspore.common._stub_tensor.StubTensor'>
            # print('r: ', r, ' type(r): ', type(r))
            
            tmp.append(r[0])
        rank_rewards.append(tmp)
    print('rank_rewards: ', rank_rewards)       # list: [[tensor,t,t,t],[t,t,t,t]]
    loss = compute_rank_list_loss(rank_rewards)
    
    #! dtype cast for mindspore.Tensor
    # loss = mindspore.ops.Cast(loss, mindspore.float32)
    # print('loss: ', loss)        # debug
    
    loss.backward()