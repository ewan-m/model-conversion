import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
export_dir = '' 
trained_checkpoint_prefix = 'checkpoints/model-108480'
tf.reset_default_graph()
graph = tf.Graph()
loader = tf.train.import_meta_graph(trained_checkpoint_prefix + ".meta" )
sess = tf.Session()
loader.restore(sess,trained_checkpoint_prefix)
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING, tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
builder.save()
