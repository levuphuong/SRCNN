import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None):
        # lưu batch size, preprocessors, data augmentor
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug

        # mở HDF5 file để đọc
        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["images"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]  # labels là ảnh đích (high-res)

                # tiền xử lý ảnh nếu có
                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    images = np.array(procImages)

                # data augmentation nếu có
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                yield (images, labels)

            epochs += 1

    def close(self):
        self.db.close()

    def __iter__(self):
        return self.generator()
