import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None):
        # store the batch size, list of preprocessors, and optional data augmenter
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug

        # open the HDF5 database file for reading
        self.db = h5py.File(dbPath, "r")
        self.numImages = self.db["images"].shape[0]  # total number of images in the dataset

    def generator(self, passes=np.inf):
        # create a generator that yields batches of data for a given number of epochs
        epochs = 0
        while epochs < passes:
            # loop over the dataset in batch-sized chunks
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract a batch of images and labels
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]  # labels are the ground-truth high-res images

                # apply any preprocessing steps if specified
                if self.preprocessors is not None:
                    procImages = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                        procImages.append(image)
                    images = np.array(procImages)

                # apply data augmentation if specified
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images, labels, batch_size=self.batchSize))

                # yield the batch to the calling function
                yield (images, labels)

            # increment the epoch counter
            epochs += 1

    def close(self):
        # close the HDF5 file
        self.db.close()

    def __iter__(self):
        # make the generator class itself iterable
        return self.generator()
