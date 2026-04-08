import pipeline_own as pic
import matplotlib.pyplot as plt

img1 = pic.Picture("PAT_938_1789_693.png")
#img1 = pic.Picture("PAT_1209_737_590.png")
img1.clean_picture()

plt.figure(figsize=(25,5))
plt.subplot(1,5,1)
plt.imshow(img1.img_org)
plt.title('Original')
plt.axis('off')

plt.subplot(1,5,2)
plt.imshow(img1.hair_mask, cmap='gray')
plt.title('Hair Mask')
plt.axis('off')

plt.subplot(1,5,3)
plt.imshow(img1.hairless)
plt.title('Hair Removed')
plt.axis('off')

plt.subplot(1,5,4)
plt.imshow(img1.pen_mask, cmap='gray')
plt.title('Pen Mask')
plt.axis('off')

plt.subplot(1,5,5)
plt.imshow(img1.clean_image)
plt.title('Clean Picture')
plt.axis('off')

plt.show()