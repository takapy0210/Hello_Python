from PIL import Image
import os, glob
import numpy as np
# from sklearn import cross_validation
from sklearn import model_selection

classes =["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# 画像の読み込み
X = []
Y = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg") #画像一覧を取得
    for i, file in enumerate(files):
        if i >= 300: break
        image = Image.open(file) #pillowsのImageメソッドを使用
        image = image.convert("RGB") #画像を赤緑青に変換
        image = image.resize((image_size, image_size))
        data = np.asarray(image) #numpyの配列形式に変換
        X.append(data)
        Y.append(index) #0, 1, 2が設定される

# numpyの配列に変換
X = np.array(X)
Y = np.array(Y)

# 分割処理
# トレーニング用のデータとテスト用のデータに分割
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y) #3:1で分割処理してくれる
xy = (X_train, X_test, y_train, y_test)
np.save("./animal.npy", xy) #numpyの配列をテキストとして保存
