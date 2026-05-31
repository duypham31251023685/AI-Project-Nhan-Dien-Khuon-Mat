from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# =========================
# DATASET
# =========================

train_dir = r"D:\nckh\data\60 tấm selfies - sáng T2"

IMG_WIDTH = 200
IMG_HEIGHT = 200
BATCH_SIZE = 64

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)

print("Số lớp:", num_classes)

# =========================
# CNN MODEL
# =========================

model = Sequential([

    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(200,200,3)
    ),
    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),
    MaxPooling2D(2,2),

    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),
    MaxPooling2D(2,2),

    Conv2D(
        256,
        (3,3),
        activation='relu'
    ),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        512,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        num_classes,
        activation='softmax'
    )

])

# =========================
# COMPILE
# =========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# =========================
# TRAIN
# =========================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[checkpoint]
)

# =========================
# SAVE MODEL
# =========================

model.save("face_model.h5")

print("Train xong!")

print(train_generator.class_indices)

print("Số lớp:", len(train_generator.class_indices))