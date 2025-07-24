from dataset.HDF5DatasetWriter import HDF5DatasetWriter
from config import sr_config as config
from imutils import paths
from PIL import Image
import numpy as np
import random
import cv2
import os
import shutil

# create temporary directories if they do not exist
for p in [config.IMAGES, config.LABELS]:
    if not os.path.exists(p):
        os.makedirs(p)

print("[INFO] creating temporary images...")
imagePaths = list(paths.list_images(config.INPUT_IMAGES))
random.shuffle(imagePaths)
total = 0

# generate training data
for imagePath in imagePaths:
    image = cv2.imread(imagePath)

    # crop the image so that its dimensions are divisible by the scale factor
    (h, w) = image.shape[:2]
    w -= int(w % config.SCALE)
    h -= int(h % config.SCALE)
    image = image[0:h, 0:w]

    # convert to YCrCb color space and keep only the Y (luminance) channel
    image_y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    # downscale and then upscale to generate the low-resolution (LR) image
    img = Image.fromarray(image_y)
    downscale_w = int(img.width * (1.0 / config.SCALE))
    downscale_h = int(img.height * (1.0 / config.SCALE))
    img_downscaled = img.resize((downscale_w, downscale_h), Image.BICUBIC)
    img_upscaled = img_downscaled.resize((img.width, img.height), Image.BICUBIC)
    scaled = np.array(img_upscaled)

    # slide a window to crop patches
    for y in range(0, h - config.INPUT_DIM + 1, config.STRIDE):
        for x in range(0, w - config.INPUT_DIM + 1, config.STRIDE):
            crop = scaled[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]
            target = image_y[
                y + config.PAD:y + config.PAD + config.LABEL_SIZE,
                x + config.PAD:x + config.PAD + config.LABEL_SIZE
            ]

            cropPath = os.path.sep.join([config.IMAGES, "{}.png".format(total)])
            targetPath = os.path.sep.join([config.LABELS, "{}.png".format(total)])
            cv2.imwrite(cropPath, crop)
            cv2.imwrite(targetPath, target)
            total += 1

# build HDF5 datasets with 1 channel and normalize to [0, 1]
print("[INFO] building HDF5 datasets...")
inputPaths = sorted(list(paths.list_images(config.IMAGES)))
outputPaths = sorted(list(paths.list_images(config.LABELS)))

inputWriter = HDF5DatasetWriter((len(inputPaths), config.INPUT_DIM, config.INPUT_DIM, 1), config.INPUTS_DB)
outputWriter = HDF5DatasetWriter((len(outputPaths), config.LABEL_SIZE, config.LABEL_SIZE, 1), config.OUTPUTS_DB)

for (inputPath, outputPath) in zip(inputPaths, outputPaths):
    inputImage = cv2.imread(inputPath, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    outputImage = cv2.imread(outputPath, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    inputImage = inputImage.reshape((config.INPUT_DIM, config.INPUT_DIM, 1))
    outputImage = outputImage.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 1))
    inputWriter.add([inputImage], [-1])
    outputWriter.add([outputImage], [-1])

inputWriter.close()
outputWriter.close()

# remove temporary directories
print("[INFO] cleaning up...")
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)
