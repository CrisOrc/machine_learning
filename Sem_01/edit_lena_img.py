import cv2
import matplotlib.pyplot as plt

img = cv2.imread(
    "/home/cristian/Camildev/Pascual/semestre6/Machine/machine_learning/Images/lena.jpg")

imgC = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lena_face = imgC[100:200, 170:270, 0:3]
plt.imshow(lena_face)
plt.show()

imgC[150:250, 299:399, 0:3] = lena_face
plt.imshow(imgC)
plt.show()
