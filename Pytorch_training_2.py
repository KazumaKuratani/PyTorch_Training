# %%
import pandas as pd 
import numpy as np
import seaborn as sns
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# %%
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#df.loc[ 行条件 , 列 ]
df['Variety'] = iris.target  
df.loc[df['Variety'] == 0, 'Variety'] = 'setosa'  
df.loc[df['Variety'] == 1, 'Variety'] = 'versicolor'  
df.loc[df['Variety'] == 2, 'Variety'] = 'virginica'
print(df.head)


# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader 



# %%
data = iris.data
label = iris.target
#train_test_split(X, y, test_size=0.2, train_size=0.8, shffle=TorF)
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2)

#torch.Tensor()はデフォルトで浮動小数点型(float32)
#torch.LongTensor()は整数型(float64)
#基本的にクラスラベルとインデックスラベル以外はデフォルトで良いかな
train_x = torch.Tensor(train_data)
test_x = torch.Tensor(test_data)
train_y = torch.LongTensor(train_label)
test_y = torch.LongTensor(test_label)

#特徴量とラベルを結合したデータセットを作成
train_dataset = TensorDataset(train_x, train_y)  
test_dataset = TensorDataset(test_x, test_y)  



# %%
#➁訓練データとテストデータの用意
#ミニバッチサイズを指定したデータローダーを作成  
train_batch = DataLoader(
    dataset=train_dataset, # datasetの指定
    batch_size=5, #バッチサイズの指定
    shuffle=True, #シャフルするかしないか、時系列以外はシャッフルするよね(これは訓練データだから)
    num_workers=2)   #並列処理の個数
test_batch = DataLoader(  
    dataset=test_dataset,  
    batch_size=5,  
    shuffle=False,  #テストデータだからシャッフルしない
    num_workers=2) 

# %%
#➂ニューラルネットワークの定義
#ここではnn.Sequential()を使わない理由は柔軟性を持たせるため
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(4,100)
        self.Lelu1 = nn.ReLU()
        self.Linear2 = nn.Linear(100,3)
    
    def forward(self,x):
        x = self.Linear1
        x = self.Lelu1
        x = self.Linear2
        
        return x
    
    
epoch = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = NN().to(device)

# %%
#➃損失関数と最適化関数の定義
#損失関数
criterion = nn.CrossEntropyLoss()

#最適化関数
#optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
optimizer = optim.Adam(net.parameters())
# %%
#➄学習(trainループ)する
def train_loop(dataloader, model ,loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    #enumerate()は常に(インデックス、中身)というペアを返す。今回のdataloaderの場合は中身自体が(画像データ、ラベル)というペアになっているから（インデックス,(画像データ、ラベル)）
    #for x,data in enumerate(dataloader):     enumerateは列挙するという意味
        #y,z = data    としてもいい
    for batch, (X,y) in enumerate(dataloader):
        
        #model(X): forward メソッドが呼ばれ、現在の重みを使って予測値（Logits）を計算する。
        pred = model(X)
        
        #loss_fn(pred, y): 予測値と正解 y を比較し、「どれくらい間違っているか」を一つの数値（loss）として出す。
        loss = loss_fn(pred,y)
        
        #PyTorchは放っておくと勾配を足し算し続けてしまうため、毎回リセットが必要
        optimizer.zero_grad()
        
        #backward()で誤差lossを元に、backward()処理（勾配計算）をPyTorchが逆伝播で自動で行ってくれる
        loss.backward()
        
        #stepで計算された勾配を使って実際にモデルの重みW,bを更新する
        optimizer.step()
        
        if batch % 100 == 0:
            loss,current = loss.item(), batch*len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")


# %%
#➅評価(testループ)する
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    
    #PyTorchは通常計算を行うたびに「計算した情報」を裏側でメモしているからここではそのメモを完全にオフしている
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            
            #.item() を使うことで、純粋な Pythonの数値（float） に変換し、身軽な状態で合計している。変換しない場合は「PyTorchのテンソル形式」で受け取っていて、不要な情報も引き連れている。
            test_loss += loss_fn(pred, y).item()
            
            #predは各数字に対するスコアのリスト
            #(pred.argmax(1) == y)で[True, False, True, ...] という真偽値のリスト（テンソル）になる
            #.type(torch.float)でPyTorchではTrue/Falseのまま計算はできないため1.0と0.0の数値に変換している
            #.item() で、PyTorchの世界からPythonの数値として取り出しcorrectに加算する
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")