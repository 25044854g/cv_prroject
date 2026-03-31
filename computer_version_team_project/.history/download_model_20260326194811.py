import urllib.request
import os

url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task'
filename = 'hand_landmarker.task'

print('⏳ 正在下载模型文件...')
try:
    urllib.request.urlretrieve(url, filename)
    size = os.path.getsize(filename)
    print(f'✓ 下载完成，文件大小: {size / 1024 / 1024:.2f} MB')
except Exception as e:
    print(f'❌ 下载失败: {e}')