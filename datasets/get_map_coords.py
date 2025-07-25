"""
获取一张图片某像素的坐标
"""

import matplotlib.pyplot as plt
from PIL import Image

def get_click_locations(img):
    click_locations = []
    def onclick(event):
        click_locations.append((event.xdata, event.ydata))
        print(f"Clicked at pixel coordinates: ({event.xdata}, {event.ydata})")
    fig, ax = plt.subplots()
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return click_locations

img = Image.open('/home/moonlet/projects/fast-reid/datasets/yisuo/map.png')  # 替换为你的图片路径
get_click_locations(img)