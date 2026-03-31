import urllib.request
import os
import sys

# 尝试多个 URL
urls = [
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker.task',
    'https://download.tensorflow.org/mediapipe/hand_landmarker.task',
]

filename = 'hand_landmarker.task'

print('⏳ 正在下载 MediaPipe 手部模型...')

success = False
for url in urls:
    try:
        print(f'尝试从: {url}')
        urllib.request.urlretrieve(url, filename)
        size = os.path.getsize(filename)
        print(f'✓ 下载完成！文件大小: {size / 1024 / 1024:.2f} MB')
        success = True
        break
    except Exception as e:
        print(f'❌ 这个源失败: {e}')
        continue

if not success:
    print('❌ 所有源都失败了')
    sys.exit(1)