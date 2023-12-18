# -*- coding = utf-8 -*-
# @File : test.py
# @Software : PyCharm
import torch.utils.data
from torchtext.data import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator


def yield_token(train_data_iter, tokenizer):
    for i, sample in enumerate(train_data_iter):
        label, comment = sample
        yield tokenizer(comment)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data_iter = IMDB(root='./', split='train')  # Dataset类型对象
tokenizer = get_tokenizer("basic_english")  # 分词器
vocab = build_vocab_from_iterator(yield_token(train_data_iter, tokenizer), min_freq=20, specials=["<unk>"])
vocab.set_default_index(0)  # 设置特殊字符索引为0
print(f"单词表大小：{len(vocab)}")
Emotion_Label = {
    0: "Positive",
    1: "Negative"
}


def collate_fn_for_test(data):
    token_index = []
    tokens = tokenizer(data)
    token_index.append(vocab(tokens))
    return torch.tensor(token_index).to(torch.int32)


def test(model, resume=""):
    if resume != "":
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_data = input("Sentence: ")
    test_input = collate_fn_for_test(test_data).to(DEVICE)
    output = model(test_input)
    print(f"The emotion of this sentence is {Emotion_Label[output.argmax().item()]}")
