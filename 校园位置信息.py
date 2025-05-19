import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 设置中文字体（使用系统支持的字体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或者 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 1. 导入图片并翻转图像数据
image_path = '校园地图.jpg'  # 请替换为你的图片路径
img = Image.open(image_path)
img_array = np.array(img)
img_flipped = img_array[::-1, :, :]  # 上下翻转图像数组

# 3. 创建保存坐标的文件
output_file = 'coordinates.txt'

# 2. 显示图片（设置坐标系原点在左下角）
fig, ax = plt.subplots()
ax.imshow(img_flipped,
          origin='lower',
          extent=[0, img.width, 0, img.height])  # 设置坐标系范围为图片实际尺寸

# 3. 设置坐标轴标签和网格
ax.set_xlabel('X坐标（向右增大）')
ax.set_ylabel('Y坐标（向上增大）')
ax.grid(True)

# 4. 定义点击事件获取坐标
def onclick(event):
    x = event.xdata
    y = event.ydata
    if x is not None and y is not None:
        # 直接显示数学坐标系中的坐标
        print(f"你点击的坐标是: ({x:.2f}, {y:.2f})")
        with open(output_file, 'a') as f:
            f.write(f"[{x:.2f}, {y:.2f}],\n")

# 5. 连接点击事件
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# 6. 展示图片
plt.show()
