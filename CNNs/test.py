import segmentation_models as sm
import keras

print(keras.__version__)

BACKBONE = 'mobilenetv2'

# preprocess_input = keras.applications.xception.preprocess_input()

# # load your data
# x_train, y_train, x_val, y_val = load_data(...)
#
# # preprocess input
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

# define model
model = sm.Unet(BACKBONE, input_shape=(224, 224, 3), encoder_weights='imagenet', classes=6, activation='softmax', encoder_freeze=True)
model.compile(
    'Adam',
    loss=keras.sparse_categorical_crossentropy, metrics=['binary_accuracy']
)


model.save('my_model.h5')
# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
# model.fit(
#    x=x_train,
#    y=y_train,
#    batch_size=16,
#    epochs=100,
#    validation_data=(x_val, y_val),
# )