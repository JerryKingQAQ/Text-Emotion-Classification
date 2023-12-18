# -*- coding = utf-8 -*-
# @File : train.py
# @Software : PyCharm
import os

import torch.nn.functional as F
import torch.utils.data
from torchtext.data import get_tokenizer, to_map_style_dataset
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

from models import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def yield_token(train_data_iter, tokenizer):
    for i, sample in enumerate(train_data_iter):
        label, comment = sample
        yield tokenizer(comment)


# 构建IMDB DataLoader
BATCH_SIZE = 32
train_data_iter = IMDB(root='./', split='train')  # Dataset类型对象
tokenizer = get_tokenizer("basic_english")  # 分词器
vocab = build_vocab_from_iterator(yield_token(train_data_iter, tokenizer), min_freq=20, specials=["<unk>"])
vocab.set_default_index(0)  # 设置特殊字符索引为0
print(f"单词表大小：{len(vocab)}")


def collate_fn(batch):
    '''
    对DataLoader所生成的mini-batch做进一步处理
    将单词映射到单词表中的索引
    :param batch: mini-batch
    :return: 经过处理统一大小的mini-batch
    (torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32))
    '''
    target = []
    token_index = []
    max_length = 0
    for i, (label, comment) in enumerate(batch):
        tokens = tokenizer(comment)
        token_index.append(vocab(tokens))
        if len(tokens) > max_length:
            max_length = len(tokens)
        if label == 'pos':
            target.append(0)
        else:
            target.append(1)

    token_index = [index + [0] * (max_length - len(index)) for index in token_index]  # 对一个minibatch中填充<unk>
    return (torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32))


def train(train_data_loader, eval_data_loader, model, optimizer, num_epochs, log_step_interval, save_step_interval,
          eval_step_interval, save_path, resume=""):
    start_epoch = 0
    start_step = 0

    if resume != "":
        print(f"loading from {resume}")
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_step = checkpoint["step"]

    for epoch_index in range(start_epoch, num_epochs):
        ema_loss = 0.  # 指数移动平均loss
        num_batches = len(train_data_loader)

        for batch_index, (target, token_index) in enumerate(train_data_loader):
            model.train()
            target = target.to(DEVICE)
            token_index = token_index.to(DEVICE)
            optimizer.zero_grad()
            step = num_batches * (epoch_index) + batch_index + 1
            logits = model(token_index)
            bce_loss = F.binary_cross_entropy_with_logits(torch.sigmoid(logits),
                                                          F.one_hot(target, num_classes=2).to(torch.float32))
            ema_loss = 0.9 * ema_loss + 0.1 * bce_loss
            bce_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 对梯度模进行截断 训练稳定
            optimizer.step()  # 更新参数

            if step % log_step_interval == 0:
                print(f"epoch_index: {epoch_index}, batch_index: {batch_index}, bce_loss: {bce_loss}")
            if step % save_step_interval == 0:
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, f"step_{step}.pt")
                torch.save({
                    'epoch': epoch_index,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': bce_loss,
                }, save_file)
                print(f"checkpoint has been saved in {save_file}")

            if step % eval_step_interval == 0:
                with torch.no_grad():
                    print("start to do evaluation")
                    model.eval()
                    ema_eval_loss = 0
                    total_acc_account = 0
                    total_account = 0
                    for eval_batch_index, (eval_target, eval_token_index) in enumerate(eval_data_loader):
                        eval_target = eval_target.to(DEVICE)
                        eval_token_index = eval_token_index.to(DEVICE)
                        total_account += eval_target.shape[0]
                        eval_logits = model(eval_token_index)
                        total_acc_account += (torch.argmax(eval_logits, dim=-1) == eval_target).sum().item()
                        eval_bce_loss = F.binary_cross_entropy_with_logits(torch.sigmoid(eval_logits),
                                                                           F.one_hot(eval_target, num_classes=2).to(
                                                                               torch.float32))
                        ema_eval_loss = 0.9 * ema_eval_loss + 0.1 * eval_bce_loss
                    print(f"ema_eval_loss: {ema_eval_loss}, eval_acc: {total_acc_account / total_account}")


if __name__ == '__main__':
    # model = GCNN().to(DEVICE)
    model = DNN().to(DEVICE)
    # model = GCNN_LSTM(hidden_size=128, num_layers=3).to(DEVICE)
    print("模型总参数：", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_data_iter = IMDB(root='./', split='train')
    train_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(train_data_iter), batch_size=BATCH_SIZE,
                                                    collate_fn=collate_fn, shuffle=False)
    eval_data_iter = IMDB(root='./', split='test')
    eval_data_loader = torch.utils.data.DataLoader(to_map_style_dataset(eval_data_iter), batch_size=8,
                                                   collate_fn=collate_fn, shuffle=False)

    resume = ""

    train(train_data_loader, eval_data_loader, model, optimizer, num_epochs=10, log_step_interval=20,
          save_step_interval=500,
          eval_step_interval=300, save_path="./logs_imdb_text_classification", resume=resume)
