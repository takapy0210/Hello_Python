# ファイルをアップロードして、識別するFlaskアプリケーション
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename #ファイル名に危険なコマンドなどが含まれていないかチェック

from keras.models import Sequential, load_model #ニューラルネットワークのモデル定義に必要
import keras, sys
import numpy as np
from PIL import Image

# 画像データパラメータ
classes =["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__) #アプリケーションをFlaskインスタンスとして初期化
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER #アプリケーションの設定

def allowed_file(filename):
    # ファイル名にピリオドが入っている、ファイル名のピリオド以降の拡張子が正しいか
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) #ファイルの保存
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) #ファイルパスの設定

            # モデルのロード
            model = load_model('./animal_cnn_aug.h5')

            # imageの定義
            image = Image.open(filepath) # python predict.py filename とすると、filenameは２番目の引数のため、[1]とする
            image = image.convert('RGB') # 3色のRGBに変換
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.asarray(X)

            result = model.predict([X])[0]
            predicted = result.argmax() #一番大きい配列の添字を返す
            percentage = int(result[predicted] * 100)
            return "ラベル：" + classes[predicted] + ", 確率：" + str(percentage) + " %"
            # return redirect(url_for('uploaded_file', filename=filename)) #画像が正常の場合、アップロード後のページに転送する
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>ファイルをアップロードして判定しよう</title></head>
    <body>
    <h1>ファイルをアップロードして判定しよう！</h1>
    <form method = post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''

# アップロード後の処理
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
