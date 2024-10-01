import sys
sys.path.append('../')
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

# Загружаем изображение
image = cv.imread('../images/lab5.png')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Преобразуем изображение в HSV
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

image_lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)

image_hsv_norm = image_hsv / 255.0

gs = plt.GridSpec(1, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.axis('off')
plt.subplot(gs[1])
plt.imshow(hsv_to_rgb(image_hsv_norm))
plt.title('HSV-изображение в RGB')
plt.show()



h, s, v = cv.split(image_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = image_rgb.reshape((np.shape(image_rgb)[0]*np.shape(image_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("H")
axis.set_ylabel("S")
axis.set_zlabel("V")
plt.show()




l = image_lab.copy()
l[:, :, 1] = 0
l[:, :, 2] = 0

a = image_lab.copy()
a[:, :, 0] = 0
a[:, :, 2] = 0

b = image_lab.copy()
b[:, :, 0] = 0
b[:, :, 1] = 0

h = image_hsv.copy()
h[:, :, 1] = 0
h[:, :, 2] = 0

s = image_hsv.copy()
s[:, :, 0] = 0
s[:, :, 2] = 0

v = image_hsv.copy()
v[:, :, 0] = 0
v[:, :, 1] = 0

gs = plt.GridSpec(2, 3)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(l)
plt.title('l-часть (LAB)')
plt.subplot(gs[1])
plt.imshow(a)
plt.title('a-часть (LAB)')
plt.subplot(gs[2])
plt.imshow(b)
plt.title('b-часть (LAB)')
plt.subplot(gs[3])
plt.imshow(h)
plt.title('h-часть (HSV)')
plt.subplot(gs[4])
plt.imshow(s)
plt.title('s-часть (HSV)')
plt.subplot(gs[5])
plt.imshow(v)
plt.title('v-часть (HSV)')
plt.show()



# Определим пороги для малины в HSV
lower_hsv = np.array([170, 50, 150])
upper_hsv = np.array([180, 255, 255])
mask_hsv = cv.inRange(image_hsv, lower_hsv, upper_hsv)

# Определим пороги для малины в LAB
lower_lab = np.array([20, 150, 120])
upper_lab = np.array([255, 200, 180])
mask_lab = cv.inRange(image_lab, lower_lab, upper_lab)

# Соединяем маски
combined_mask = cv.bitwise_and(mask_hsv, mask_lab)

# Применяем морфологические операции закрытия и открытия для удаления шумов
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)
mask_cleaned = cv.morphologyEx(mask_cleaned, cv.MORPH_OPEN, kernel)


# Применяем маску к изображению
result = cv.bitwise_and(image_rgb, image_rgb, mask=mask_cleaned)

# Отображаем результат
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title('Выделение малины')
plt.axis('off')

plt.show()