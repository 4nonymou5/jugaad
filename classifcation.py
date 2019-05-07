
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import tensorflow as tf

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory( directory="Data/", target_size=(224, 224), 
	color_mode="rgb" , 
	batch_size=8,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

def train():
	input_image_tensor = Input(shape=(224, 224, 3)) 
	base_model = MobileNetV2(input_tensor=input_image_tensor, weights='imagenet', include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	predictions = Dense(3, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	for layer in base_model.layers:
	    layer.trainable = True

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())
	#sys.exit()
	#model.fit_generator(generator=train_generator,steps_per_epoch=20,epochs=10)
	model.save('models/classify.h5')



def convert_keras():
		converter = tf.lite.TFLiteConverter.from_keras_model_file('models/classify.h5')
		#converter.post_training_quantize = True
		tflite_model = converter.convert()
		open("models/classify.tflite", "wb").write(tflite_model)

train()
convert_keras()
