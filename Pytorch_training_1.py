# %%
import torch
import numpy as np
import pandas as pd
from torch import nn
# %%
x = torch.tensor([[1,2,3],[4,5,6]])
#➀0から9までの1次元Tensorを生成しろ
#➁0から始まって10まで2.5ずつ増えていくTensorを生成
#➂0から1の乱数を生成
#➃すべての要素が0となる
#➄numpyからtensorに変換
#➅tensorに任意の要素を取得(ここでは1,2を取得)
#➆スライスを使って1行目のすべての要素を取得
#➇tensorの形状を変換する(2×3を3×2に,"視点"を変える意味)
#➈tensorの合計をtensorで返し、次に値のみを取得する
x1 = torch.arange(0,10)
x2 = torch.linspace(0,10,5)
x3 = torch.rand(2,3)
x4 = torch.zeros(2,3)
x5 = np.array([[1,2,3],[4,5,6]])
x6 = torch.from_numpy(x5)
x7 = x[1,2]
x8 = x[1,:]
x9 = x.view(3,2)
#標準関数であるsum(x)よりメソッド(ここではPyTorch専用の)であるx.sum()のほうが速い
x10 = torch.sum(x)
x10 = torch.x.sum()
x11 = x10.item()
print(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)



# %%
#Tensorの属性にはrequires_gradがあり、この属性をTrueにすることで自動微分機能を有効にする
a = torch.tensor(3,requires_grad=True, dtype=torch.float64)
b = torch.tensor(4,requires_grad=True, dtype=torch.float64)
x = torch.tensor(5,requires_grad=True, dtype=torch.float64)
y = a*x+b
print(y)
#それぞれ勾配を求める
# %%
#ニューラルネットワークを定義しよ、入力層、隠れ層(2つ)...
#➀自作のクラスを定義せずに作る

NN = nn.Sequential(
    nn.Linear(50,100),
    nn.ReLU(),
    nn.Linear(100,100)
)


#➁自作のクラスを定義して作る
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(50,100)
        self.Relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(100,100)
    
    def forward(self,x):
        x = self.Linear1(x)
        x = self.Relu1(x)
        x = self.Linear2(x)
        
        return x
        
# %%
#ソフトマックス交差エントロピー誤差
y = torch.rand(3,5)
t = torch.empty(3,dtype=torch.long).random_(5)
criterion = nn.CrossEntropyLoss()
loss = criterion(y,t)
print(f"予測(y):\n{y}")
print(f"正解(t): {t}")
print(f"Loss: {loss.item():.4f}")      