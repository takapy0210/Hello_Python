# 画像を回転して学習データを増幅するVer
from PIL import Image
import os, glob
import numpy as np
# from sklearn import cross_validation
from sklearn import model_selection

classes =["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

# 画像の読み込み

# 変数初期化
X_train = []
X_test = []
Y_train = []
Y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg") #画像一覧を取得
    for i, file in enumerate(files):
        if i >= 200: break
        image = Image.open(file) #pillowのImageメソッドを使用
        image = image.convert("RGB") #画像を赤緑青に変換
        image = image.resize((image_size, image_size))
        data = np.asarray(image) #numpyの配列形式に変換

        # 100枚の画像をテスト用データとして格納し、残りをトレーニング用とする
        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            # データ増幅
            # -20度から20度まで5度ずつ回転しながらループ
            for angle in range(-20, 20, 5):
                # 回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r) # numpy形式の配列に変換
                X_train.append(data)
                Y_train.append(index)

                # 反転
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT) # pillowのImageメソッドを使用し、左右反転させる
                data = np.asarray(img_trans) # numpy形式の配列に変換
                X_train.append(data)
                Y_train.append(index)

# numpyの配列に変換
# X = np.array(X)
# Y = np.array(Y)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

# 分割処理
# トレーニング用のデータとテスト用のデータに分割
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y) #3:1で分割処理してくれる
xy = (X_train, X_test, y_train, y_test)
np.save("./animal_aug.npy", xy) #numpyの配列をテキストとして保存
