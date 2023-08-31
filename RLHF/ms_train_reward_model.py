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

使用ChatGPT中Reward Model的思路训练一个RM，因为是打分模型，所以使用BERT（而非GPT）模型训练。

Author: pankeyu
Modified by: ruizhewang
Date: 2023/08/31
"""
import os
import time
import argparse
from functools import partial
import json

# import torch
# from torch.utils.data import DataLoader
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler

import mindspore
from mindspore import nn
from mindnlp import load_dataset
# 从mindnlp调入ernie模型及其tokenizer
from mindnlp.models import ErnieModel, BertModel
from mindnlp.transforms import PadTransform
from mindnlp.transforms import ErnieTokenizer, BertTokenizer
from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy

from model import RewardModel, compute_rank_list_loss
from utils import convert_example
from iTrainingLogger import iSummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-chinese', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()

writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


'''
def evaluate_model(model, data_loader):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        data_loader: 测试集的dataloader
    """
    model.eval()
    with torch.no_grad():
        batch_rank_rewards = []
        for batch in data_loader:
            for batch_idx in range(len(batch['input_ids'])):
                rank_texts_count = len(batch['input_ids'][batch_idx])
                rank_rewards = []
                for text_idx in range(rank_texts_count):
                    reward = model(
                        batch['input_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['token_type_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['attention_mask'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                        batch['position_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
                    )
                    rank_rewards.append(reward[0])                      # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
                batch_rank_rewards.append(rank_rewards)                 # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
    model.train()
    total_ranklist, right_ranklist = 0, 0
    for rank_rewards in batch_rank_rewards:
        rank_rewards = [t.cpu().float() for t in rank_rewards]
        rank_rewards_sorted = sorted(rank_rewards, reverse=True)
        total_ranklist += 1
        if rank_rewards_sorted == rank_rewards:
            right_ranklist += 1
    return right_ranklist / total_ranklist
'''


class JuejueziDataInterator:
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self._data_dict = json.load(f)

    def __getitem__(self, index):
        return self._data_dict[index]["text"], self._data_dict[index]["score"]

    def __len__(self):
        return len(self._data_dict)
    


def process_dataset(source, tokenizer, pad_value, max_seq_len=64, batch_size=32, shuffle=True):
    column_names = ['text', 'score']
    
    dataset = mindspore.dataset.GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    pad_op = PadTransform(max_seq_len, pad_value=pad_value)
    type_cast_op = mindspore.dataset.transforms.TypeCast(mindspore.int32)
    
    # map dataset
    dataset = dataset.map(operations=[tokenizer, pad_op], input_columns="text")     # 对文本tokenize+pad
    dataset = dataset.map(operations=[type_cast_op], input_columns="score")         # 对分数转换为int32
    # # rename dataset
    # dataset = dataset.rename(input_columns=['text', 'score'], output_columns=rename_columns)
    # batch dataset
    dataset = dataset.batch(args.batch_size)

    return dataset


def train():
    # encoder = AutoModel.from_pretrained(args.model)
    # model = RewardModel(encoder=encoder)
    # tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # 准备模型
    encoder = BertModel.from_pretrained('bert-base-chinese')
    pad_value = tokenizer.token_to_id('[PAD]')      # 根据模型获取pad_value
    model = RewardModel(encoder)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    # dataset = dataset.map(convert_func, batched=True)
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["dev"]
    # train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)
    # eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)
    # TODO: 预处理数据集，包括tokenize等
    pad_op = PadTransform(max_length=args.max_seq_len, pad_value=pad_value)
    type_cast_op = mindspore.dataset.transforms.TypeCast(mindspore.int32)
    
    # data = JuejueziDataInterator(args.train_path)
    train_set = process_dataset(JuejueziDataInterator("data/juejuezi_datasets/train.json"), tokenizer, pad_value)
    eval_set = process_dataset(JuejueziDataInterator("data/juejuezi_datasets/eval.json"), tokenizer, pad_value, shuffle=False)
    

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = nn.Adam(optimizer_grouped_parameters, learning_rate=args.learning_rate)
    metric = Accuracy()
    loss_fn = nn.MAELoss()
    model.to(args.device)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_set)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)

    # loss_list = []
    tic_train = time.time()
    global_step, best_acc = 0, 0
    
    # define callbacks to save checkpoints
    ckpoint_cb = CheckpointCallback(save_path='checkpoint', ckpt_name='juejuezi_reward_model', epochs=1, keep_checkpoint_max=2)
    best_model_cb = BestModelCallback(save_path='checkpoint', auto_load=True)
    
    trainer = Trainer(
        network=model, 
        train_dataset=train_set,
        eval_dataset=eval_set, 
        metrics=metric,
        epochs=3, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        callbacks=[ckpoint_cb, best_model_cb],
        jit=True
    )
    
    trainer.run(tgt_columns="label")
    
    # 目前没有测试集，只有验证集
    # evaluator = Evaluator(network=model, eval_dataset=dataset_test, metrics=metric)
    # evaluator.run(tgt_columns="label")
    
    # for epoch in range(1, args.num_train_epochs+1):
    #     for batch in train_dataloader:
    #         batch_rank_rewards = []
    #         for batch_idx in range(len(batch['input_ids'])):
    #             rank_texts_count = len(batch['input_ids'][batch_idx])
    #             rank_rewards = []
    #             for text_idx in range(rank_texts_count):
    #                 reward = model(
    #                     batch['input_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
    #                     batch['token_type_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
    #                     batch['attention_mask'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
    #                     batch['position_ids'][batch_idx][text_idx].unsqueeze(dim=0).to(args.device),
    #                 )
    #                 rank_rewards.append(reward[0])                      # (rank_text_num) -> [tensor([0.1696]), tensor([0.3466])]
    #             batch_rank_rewards.append(rank_rewards)                 # (batch, rank_text_num) -> [[tensor([0.1696]), tensor([0.3466])], ...]
    #         loss = compute_rank_list_loss(batch_rank_rewards, device=args.device)
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         loss_list.append(float(loss.cpu().detach()))
            
    #         global_step += 1
    #         if global_step % args.logging_steps == 0:
    #             time_diff = time.time() - tic_train
    #             loss_avg = sum(loss_list) / len(loss_list)
    #             writer.add_scalar('train/train_loss', loss_avg, global_step)
    #             print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
    #                     % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
    #             tic_train = time.time()

    #         if global_step % args.valid_steps == 0:
    #             cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
    #             if not os.path.exists(cur_save_dir):
    #                 os.makedirs(cur_save_dir)
    #             torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
    #             tokenizer.save_pretrained(cur_save_dir)
    #             acc = evaluate_model(model, eval_dataloader)
    #             writer.add_scalar('eval/accuracy', acc, global_step)
    #             writer.record()
    #             print("Evaluation acc: %.5f" % (acc))
    #             if acc > best_acc:
    #                 print(
    #                     f"best F1 performence has been updated: {best_acc:.5f} --> {acc:.5f}"
    #                 )
    #                 best_acc = acc
    #                 cur_save_dir = os.path.join(args.save_dir, "model_best")
    #                 if not os.path.exists(cur_save_dir):
    #                     os.makedirs(cur_save_dir)
    #                 torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
    #                 tokenizer.save_pretrained(cur_save_dir)
    #             tic_train = time.time()


if __name__ == '__main__':
    from rich import print
    train()
