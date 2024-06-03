# %%
import keras_cv

from keras import Sequential
from keras.src.layers import RandomFlip, RandomContrast, RandomBrightness, Conv2D, MaxPooling2D, Dense, Flatten
from keras.src.utils import image_dataset_from_directory, to_categorical

data = image_dataset_from_directory(
        r"data\images",
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=64,
        image_size=(512, 512),
        shuffle=True,
        interpolation="bilinear",
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=True,
        verbose=False,
    )
preprocessing_layer = Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomContrast(0.2),
        RandomBrightness(0.2),
    ])

data = data.map(lambda x, y: (preprocessing_layer(x), to_categorical(y, num_classes=2)))
print(data)

# %%
# Display the first 9 images
batch = next(iter(data.take(1)))
images, labels = batch
keras_cv.visualization.plot_image_gallery(
    batch[0],
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)

# %%
num_filters = 8
filter_size = 3
pool_size = 2
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(512, 512, 3)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(2, activation='softmax'),
])
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)
model.fit(
    data,
    epochs=100,
)