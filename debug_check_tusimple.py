from code.utils.paths import TUSIMPLE_IMAGES, TUSIMPLE_MASKS

images = sorted(TUSIMPLE_IMAGES.glob("*.jpg"))
masks = sorted(TUSIMPLE_MASKS.glob("*.png"))

print("Images:", len(images))
print("Masks :", len(masks))

missing = 0
for img in images:
    mask = TUSIMPLE_MASKS / img.name.replace(".jpg", ".png")
    if not mask.exists():
        missing += 1

print("Missing masks:", missing)
