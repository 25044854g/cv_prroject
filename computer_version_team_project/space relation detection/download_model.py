import urllib.request
import os
import ssl

# 禁用 SSL 验证（因为证书问题）
ssl._create_default_https_context = ssl._create_unverified_context

# 尝试从 GitHub release 下载
url = 'https://github.com/google-ai-edge/mediapipe/releases/download/v0.10.0/hand_landmarker.task'
filename = 'hand_landmarker.task'

print('⏳ 正在从 GitHub 下载模型...')
try:
    urllib.request.urlretrieve(url, filename)
    size = os.path.getsize(filename)
    print(f'✓ 下载完成！文件大小: {size / 1024 / 1024:.2f} MB')
except Exception as e:
    print(f'❌ 下载失败: {e}')