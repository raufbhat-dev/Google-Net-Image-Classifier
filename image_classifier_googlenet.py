import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
from tensorflow.keras import layers



os.chdir("/home/rauf_bhat/projects/data_science/")
data_dir = '/home/rauf_bhat/projects/data_science/intel-image-classification'


AUTOTUNE = tf.data.experimental.AUTOTUNE
batch_size = 32
img_height = 224
img_width = 224



train_ds = tf.data.Dataset.list_files(data_dir+"/seg_train/*/*.jpg")
val_ds = tf.data.Dataset.list_files(data_dir+"/seg_test/*/*.jpg",shuffle=False,seed=1)
test_ds = tf.data.Dataset.list_files(data_dir+"/seg_pred/*.jpg",shuffle=False,seed=1)
test_ds1 = tf.data.Dataset.list_files(data_dir+"/seg_pred/*.jpg",shuffle=False,seed=1)

data_dir = pathlib.Path(data_dir)


class_names = np.array(sorted([item.name for item in data_dir.glob('seg_train/*') ]))
print(class_names)



print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())
print(tf.data.experimental.cardinality(test_ds).numpy())
print(tf.data.experimental.cardinality(test_ds1).numpy())


def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  one_hot = parts[-2] == class_names
  return tf.argmax(one_hot)



def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  return tf.image.resize(img, [img_height, img_width])



def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label



train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)



for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())



def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)



image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(20, 20))
for i in range(25):
  ax = plt.subplot(5, 5, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(class_names[label])
  plt.axis("off")


def inception_module(x_inp,kernel_size):
    print("\n*** inception  start ***\n")
    x1 = tf.keras.layers.Conv2D(kernel_size['x1'], 1, strides=1, padding='same',kernel_initializer='glorot_normal')(x_inp)
    x1 = tf.keras.activations.relu(x1, alpha=0.01)
    print("x1: " , x1.shape)

    x1_3 = tf.keras.layers.Conv2D(kernel_size['x1_3'], 1, strides=1, padding='same',kernel_initializer='glorot_normal')(x_inp)
    x1_3 = tf.keras.activations.relu(x1_3, alpha=0.01)
    print("x1_3: " ,x1_3.shape)
    x3 = tf.keras.layers.Conv2D(kernel_size['x3'], 3, strides=1, padding='same',kernel_initializer='glorot_normal')(x1_3)
    x3 = tf.keras.activations.relu(x3, alpha=0.01)
    print("x3: ", x3.shape)

    x1_5 = tf.keras.layers.Conv2D(kernel_size['x1_5'], 1,strides=1, padding='same',kernel_initializer='glorot_normal')(x_inp)
    x1_5 = tf.keras.activations.relu(x1_5, alpha=0.01)
    print("x1_5: ",x1_5.shape)
    x5 = tf.keras.layers.Conv2D(kernel_size['x5'], 5,strides=1, padding='same',kernel_initializer='glorot_normal')(x1_5)
    x5 = tf.keras.activations.relu(x5, alpha=0.01)
    print("x5: ",x5.shape)

    x_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x_inp)
    print("x_pool: ",x_pool.shape)
    x_pool_proj = tf.keras.layers.Conv2D(kernel_size['x_pool_proj'],1,strides=1, padding='same',kernel_initializer='glorot_normal')(x_pool)
    x_pool_proj = tf.keras.activations.relu(x_pool_proj, alpha=0.01)
    print("x_pool_proj: ",x_pool_proj.shape)

    x_out = tf.keras.layers.Concatenate(axis=-1)([x1, x3, x5, x_pool_proj])
    print("x_out: ",x_out.shape)
    print("\n*** inception  end ***\n")

    return x_out



def bell_module(x_inp,i):
    print("\n*** bell start ***\n")
    #bellAvgPool1
    y = tf.keras.layers.MaxPooling2D(pool_size=5, strides=3, padding='valid')(x_inp)
    print("avgPool: ",y.shape)

    #bellDropout1
    y = tf.keras.layers.Dropout(.4)(y)

    #conv1
    y = tf.keras.layers.Conv2D(64, 1, strides=1,padding='same',kernel_initializer='glorot_normal')(y)
    print("conv1: ",y.shape)

    #bellfcLayer
    # extra fc layer added in this usecase
    y = tf.keras.layers.Flatten()(y)
    print("input: ", y.shape)
    y = tf.keras.layers.Dense(512,kernel_initializer='glorot_normal')(y)
    y = tf.keras.activations.relu(y, alpha=0.01)
    print("fc1: ", y.shape)
    y = tf.keras.layers.Dense(128,kernel_initializer='glorot_normal')(y)
    y = tf.keras.activations.relu(y, alpha=0.01)
    print("fc2: ", y.shape)
    y_out1 = tf.keras.layers.Dense(6,kernel_initializer='glorot_normal',name="bell"+i)(y)
    print("fc3 output : ", y_out1.shape)
    print("\n*** bell  end ***\n")
    return y_out1



num_classes = 6

inputs = tf.keras.Input(shape=(img_height,img_width,3))
print("input: ", inputs.shape)

#conv1
x = tf.keras.layers.Conv2D(64, 7, strides=2,padding='same',kernel_initializer='glorot_normal')(inputs)
print("conv1: " ,x.shape)
x = tf.keras.activations.relu(x, alpha=0.01)

#maxPool1
x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
print("maxPool1: ", x.shape)
x = tf.keras.layers.BatchNormalization()(x)

#conv2
x = tf.keras.layers.Conv2D(192, 1, strides=1,padding='valid',kernel_initializer='glorot_normal')(x)
print("conv2: ", x.shape)
x = tf.keras.activations.relu(x, alpha=0.01)
x = tf.keras.layers.BatchNormalization()(x)

#conv3
x = tf.keras.layers.Conv2D(192, 3, strides=2,padding='same',kernel_initializer='glorot_normal')(x)
x = tf.keras.activations.relu(x, alpha=0.01)
print("conv3: ", x.shape)

#3a
kernel_size ={'x1':64,'x1_3':96,'x3':128,'x1_5':16,'x5':32,'x_pool_proj':32}
x= inception_module(x,kernel_size)
print("inception 3a: ",x.shape)

#3b
kernel_size ={'x1':128,'x1_3':128,'x3':192,'x1_5':32,'x5':96,'x_pool_proj':64}
x= inception_module(x,kernel_size)
print("inception 3b: ",x.shape)

#maxPool2
x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
print("maxPool2: ", x.shape)

#4a
kernel_size ={'x1':192,'x1_3':96,'x3':208,'x1_5':16,'x5':48,'x_pool_proj':64}
x= inception_module(x,kernel_size)
print("inception 4a: ",x.shape)

y1_out = bell_module(x,"1")

#4b
kernel_size ={'x1':160,'x1_3':112,'x3':224,'x1_5':24,'x5':64,'x_pool_proj':64}
x= inception_module(x,kernel_size)
print("inception 4b: ",x.shape)

#4c
kernel_size ={'x1':128,'x1_3':128,'x3':256,'x1_5':24,'x5':64,'x_pool_proj':64}
x= inception_module(x,kernel_size)
print("inception 4c: ",x.shape)

#4d
kernel_size ={'x1':112,'x1_3':144,'x3':288,'x1_5':32,'x5':64,'x_pool_proj':64}
x= inception_module(x,kernel_size)
print("inception 4d: ",x.shape)

y2_out = bell_module(x,"2")

#4e
kernel_size ={'x1':256,'x1_3':160,'x3':320,'x1_5':32,'x5':128,'x_pool_proj':128}
x= inception_module(x,kernel_size)
print("inception 4e: ",x.shape)

#maxPool2
x = tf.keras.layers.MaxPooling2D(pool_size=7, strides=2, padding='same')(x)
print("maxPool3: ", x.shape)


#5a
kernel_size ={'x1':256,'x1_3':160,'x3':320,'x1_5':32,'x5':128,'x_pool_proj':128}
x= inception_module(x,kernel_size)
print("inception 5a: ",x.shape)

#5b
kernel_size ={'x1':384,'x1_3':192,'x3':384,'x1_5':48,'x5':128,'x_pool_proj':128}
x= inception_module(x,kernel_size)
print("inception 5b: ",x.shape)

#avgPool
x = tf.keras.layers.AveragePooling2D(pool_size=7, strides=1, padding='valid')(x)
print("avgPool1: ", x.shape)

#dropout
x = tf.keras.layers.Dropout(.4)(x)

#fcLayer
# extra fc layer added in this usecase
x = tf.keras.layers.Flatten()(x)
print("input: ", x.shape)
x = tf.keras.layers.Dense(512,kernel_initializer='glorot_normal')(x)
x = tf.keras.activations.relu(x, alpha=0.01)
print("fc1: ", x.shape)
x = tf.keras.layers.Dense(128,kernel_initializer='glorot_normal')(x)
x = tf.keras.activations.relu(x, alpha=0.01)
print("fc2: ", x.shape)
x_out = tf.keras.layers.Dense(6,kernel_initializer='glorot_normal',name ="main_bell")(x)
print("fc3 output: ", x_out.shape)

googlenet = tf.keras.Model(inputs=inputs,outputs=[y1_out,y2_out,x_out])




googlenet.compile(
  optimizer=tf.keras.optimizers.Adam(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=439, decay_rate=0.5, staircase=False), beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'],)



history = googlenet.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)



#Accuracy
plt.plot(history.history['main_bell_accuracy'])
plt.plot(history.history['bell2_accuracy'])
plt.plot(history.history['bell1_accuracy'])
plt.plot(history.history['val_main_bell_accuracy'])
plt.plot(history.history['val_bell2_accuracy'])
plt.plot(history.history['val_bell1_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['main_bell_accuracy','bell2_accuracy','bell1_accuracy','val_main_bell_accuracy','val_bell2_accuracy','val_bell1_accuracy'], loc='lower right')
plt.show()
# "Loss"
plt.plot(history.history['main_bell_loss'])
plt.plot(history.history['bell2_loss'])
plt.plot(history.history['bell1_loss'])
plt.plot(history.history['val_main_bell_loss'])
plt.plot(history.history['val_bell2_loss'])
plt.plot(history.history['val_bell1_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['main_bell_loss','bell2_loss','bell1_loss','val_main_bell_loss','val_bell2_loss','val_bell1_loss'], loc='upper right')
plt.show()


tf.keras.utils.plot_model(googlenet, "my_first_model.png")





