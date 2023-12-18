# -*- coding = utf-8 -*-
# @File : main.py
# @Software : PyCharm
from models import *
from test import test
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == "__main__":
    model = DNN().to(DEVICE)
    test(model, "logs_imdb_text_classification/DNN_step_7500.pt")
