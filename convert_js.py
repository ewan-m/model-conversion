import tensorflowjs as tfjs
import tensorflow as tf

model = tf.saved_model.load("model.pb") # dunno if this legit??

tfjs.converters.save(model, "modelJs")
