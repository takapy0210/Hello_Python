# 画像ファイルを引数として、その画像が何か判断する
from keras.models import Sequential, load_model #ニューラルネットワークのモデル定義に必要
from keras.layers import Conv2D, MaxPooling2D #畳み込み処理などに使用
from keras.layers import Activation, Dropout, Flatten, Dense #データ一次元変換関数など
from keras.utils import np_utils #numpyのutils
import keras, sys
import numpy as np
from PIL import Image

# 画像データパラメータ
classes =["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

def build_model():
    # モデルを定義
    model = Sequential()

    # model.addにより、モデルの層を追加していく
    # 1層目の畳み込み
    # X_train.shapeで配列の形状を返す
    # shape[1:]と記述することにより、配列の１番目の要素以降のデータが取得できる
    model.add(Conv2D(32, (3, 3), padding='same', input_shape = (50,50,3))) # 50*50ピクセルでRGBの3色のモデルという意味
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

    # loss:は損失関数 正解と推定値との誤差
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    # モデルのロード
    model = load_model('./animal_cnn_aug.h5')

    return model

def main():
    image = Image.open(sys.argv[1]) # python predict.py filename とすると、filenameは２番目の引数のため、[1]とする
    image = image.convert('RGB') # 3色のRGBに変換
    image = image.resize((image_size, image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.asarray(X)

    # モデルの作成
    model = build_model()

    result = model.predict([X])[0]
    predicted = result.argmax() #一番大きい配列の添字を返す
    percentage = int(result[predicted] * 100)
    print("{0} ({1} %)".format(classes[predicted], percentage))

## ファイルが直接呼ばれた時のみ、mainをコールする
if __name__ == "__main__":
    main()
