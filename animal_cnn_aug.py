# 作成したnpyファイルを読み込み、機械学習を行ったあと、animal_cnn_aug.h5ファイルを作成する
from keras.models import Sequential #ニューラルネットワークのモデル定義に必要
from keras.layers import Conv2D, MaxPooling2D #畳み込み処理などに使用
from keras.layers import Activation, Dropout, Flatten, Dense #データ一次元変換関数など
from keras.utils import np_utils #numpyのutils
import keras
import numpy as np

# 画像データパラメータ
classes =["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# データの読み込み
# メイン関数を定義する
def main():
    X_train, X_test, y_train, y_test = np.load("./animal_aug.npy") #ファイルからデータを配列に読み込む
    # 1 ~ 256 のカラーコードを 0 or 1に変換
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256

    # one-hot-vector: 正解値は1, 他は0に変換する
    # 例）monkeyが正解であれば[1,0,0]となる
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # トレーニングのモデル生成
    model = model_train(X_train, y_train)

    # モデルの評価
    model_eval(model, X_test, y_test)

def model_train(X, y):
    # モデルを定義
    model = Sequential()

    # model.addにより、モデルの層を追加していく
    # 1層目の畳み込み
    # X_train.shapeで配列の形状を返す
    # shape[1:]と記述することにより、配列の１番目の要素以降のデータが取得できる
    model.add(Conv2D(32, (3, 3), padding='same', input_shape = X.shape[1:]))
    model.add(Activation('relu')) #活性化関数 正の部分を通す
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #一番大きな値を取り出す（特徴を際立たせる）
    model.add(Dropout(0.25)) #データの偏りを減らす 25%を捨てる

    # 2層目の畳み込み
    model.add(Conv2D(64, (3, 3), padding='same')) #64個のフィルターを使用
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25)) #データの偏りを減らす 25%を捨てる

    # モデルの結合
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) #50%を捨てる
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    # loss:は損失関数 正解と推定値との誤差
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    # epochを増やすと正確になるが、処理が重くなる. batch_sizeはトレーニングに使用する枚数
    model.fit(X, y, batch_size = 30, epochs = 100)

    # モデルの保存
    model.save('./animal_cnn_aug.h5')

    return model

def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose = 1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

if __name__ == "__main__":
    main()
