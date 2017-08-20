def boxes():
	return "list of 4-tuplpe s"
def standard_good():
	return "(width, height, 3) ndarray"
def standard_null():
	return "(width, height, 3) ndarray"

rect = crop(img, box)
rect_swirled = swirl(rect, rotation=0, strength=10, radius=100)
rect_noisy = noisy(rect)
rect2 = rect_swirled

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), 
								sharex=True, sharey=True,
								subplot_kw={'adjustable':'box-forced'})

ax0.imshow(rect, cmap=plt.cm.gray, interpolation='none')
ax0.axis('off')
ax1.imshow(rect2, cmap=plt.cm.gray, interpolation='none')
ax1.axis('off')

img = Image.open("1.png")
box = np.array([100, 100, 200, 200])
gray_img = img.convert("L")
# img = gray_img


def noisy(img):
	noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
	noise[np.random.random(size=noise.shape) > 0.5] *= -1
	return img + noise
