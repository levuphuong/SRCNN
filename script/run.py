import os
from config import sr_config as config
from keras.models import load_model
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-b", "--baseline", required=True, help="path to baseline image")
ap.add_argument("-o", "--output", required=True, help="path to output image")
args = vars(ap.parse_args())

# load the trained SRCNN model
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH, compile=False)

# load and crop the input image
image = cv2.imread(args["image"])
if image is None:
    raise ValueError("Could not read input image!")

(h, w) = image.shape[:2]
# crop image so its dimensions are divisible by the scale factor
w -= int(w % config.SCALE)
h -= int(h % config.SCALE)
image = image[0:h, 0:w]

# upscale using bicubic interpolation as baseline
scale = config.SCALE
new_w = int(w * scale)
new_h = int(h * scale)
scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(args["baseline"], scaled)

# divide the upscaled image into patches
patches = []
positions = []
for y in range(0, new_h - config.INPUT_DIM + 1, config.LABEL_SIZE):
    for x in range(0, new_w - config.INPUT_DIM + 1, config.LABEL_SIZE):
        crop = scaled[y:y + config.INPUT_DIM, x:x + config.INPUT_DIM]
        patches.append(crop)
        positions.append((y, x))

patches = np.array(patches, dtype=np.float32)

# run model inference in batches
print("[INFO] running batch inference...")
preds = model.predict(patches, batch_size=32, verbose=1)  # increase batch_size if GPU allows

# stitch the output patches back into a full image
output = np.zeros(scaled.shape, dtype=np.float32)
for (P, (y, x)) in zip(preds, positions):
    P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
    output[y + config.PAD:y + config.PAD + config.LABEL_SIZE,
           x + config.PAD:x + config.PAD + config.LABEL_SIZE] = P

# clip pixel values to valid range and save the final SR image
output = np.clip(output, 0, 255).astype("uint8")
cv2.imwrite(args["output"], output)

print("[INFO] Done.")
