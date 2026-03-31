import cv2

# 操作1：读取一张图片（改进版本）
img = cv2.imread('photo.jpg')

# 检查图片是否成功加载
if img is None:
    print("错误：无法加载图片，请检查文件路径和文件名！")
else:
    # 获取图片的高度和宽度
    height, width = img.shape[:2]
    print(f"图片尺寸：宽度={width}，高度={height}")
    
    # 如果图片太大，缩小到屏幕能显示的大小
    max_width = 1200
    max_height = 800
    
    if width > max_width or height > max_height:
        # 计算缩放比例，保持宽高比
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
        print(f"已缩放至：宽度={new_width}，高度={new_height}")
    
    # 创建一个命名的窗口（可调整大小）
    cv2.namedWindow('My Photo', cv2.WINDOW_NORMAL)
    cv2.imshow('My Photo', img)
    
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

