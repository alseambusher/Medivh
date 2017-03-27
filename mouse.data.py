import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import json

x = np.arange(-10,10)
y = x**2

img=mpimg.imread('0.png')
fig = plt.figure()
plt.imshow(img)

im = Image.open('0.png')
im = im.convert('RGB')

coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    r, g, b = im.getpixel((ix, iy))
    print 'x = %d, y = %d'%(
        ix, iy), r, g, b, len(coords)

    global coords
    coords.append([r, g, b, ix, iy])
    json.dump({"data": coords}, open("clicks.json", "w"))
    #
    # if len(coords) == 2:
    #     fig.canvas.mpl_disconnect(cid)

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
