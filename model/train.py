import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ép chạy CPU

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras.callbacks import Callback
from keras.optimizers import Adam
from keras.optimizers import SGD
import tensorflow as tf


from config import sr_config as config
from dataset.HDF5DatasetGenerator import HDF5DatasetGenerator
from model.srcnn import SRCNN


class TimeHistory(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        print(f"[INFO] Epoch {epoch+1} started...")

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start_time
        print(f"[INFO] Epoch {epoch+1} completed - loss={logs['loss']:.6f} - duration={duration:.2f}s")


def super_res_generator(inputDataGen, targetDataGen):
    for (inputData, targetData) in zip(inputDataGen.generator(), targetDataGen.generator()):
        input_images = inputData[0]   # lấy images, bỏ labels
        target_images = targetData[0] # lấy images, bỏ labels
        # print("[DEBUG] input_images:", input_images.shape)
        # print("[DEBUG] target_images:", target_images.shape)
        yield (input_images, target_images)
        # yield (inputData, targetData)


# load data
inputs = HDF5DatasetGenerator(config.INPUTS_DB, config.BATCH_SIZE)
targets = HDF5DatasetGenerator(config.OUTPUTS_DB, config.BATCH_SIZE)

print("[INFO] compiling model...")
# opt = Adam(learning_rate=0.0001)

initial_lr = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,          # số step trước khi giảm
    decay_rate=0.96,           # mỗi lần giảm còn 96%
    staircase=True             # True: giảm theo bậc, False: giảm liên tục
)

opt = Adam(learning_rate=lr_schedule)
# opt = SGD(learning_rate=1e-3, momentum=0.9)
model = SRCNN.build(width=config.INPUT_DIM, height=config.INPUT_DIM, depth=1)
model.compile(loss="mse", optimizer=opt)

print("steps_per_epoch =", inputs.numImages // config.BATCH_SIZE)
print("numImages =", inputs.numImages)
print("batch_size =", config.BATCH_SIZE)

# train
H = model.fit(
    super_res_generator(inputs, targets),
    steps_per_epoch=inputs.numImages // config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=2,
    callbacks=[TimeHistory()]
)

print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# plot loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"], label="loss")
plt.title("Loss on SRCNN training")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(config.PLOT_PATH)

inputs.close()
targets.close()
