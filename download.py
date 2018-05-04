from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os, time, sys

# APIキー情報
key = "81292d9e00911556f1dc2d9cc6c40d96"
secret ="0f5931b8d9165c78"
wait_time = 0.5


# 保存フォルダを引数として指定
animalname = sys.argv[1]
savedir = "./" + animalname

# Flickrのクライアント作成
flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = animalname,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

# データが取得できているか画面に表示して確認
photos = result['photos']
# pprint(photos)

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg' #ファイル名指定
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
