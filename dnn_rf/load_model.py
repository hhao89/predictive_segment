import tensorflow as tf
sess=tf.Session() 

export_path =  './savedmodel'
graph = tf.saved_model.loader.load(sess, ['tag'],export_path)

x = sess.graph.get_tensor_by_name('x:0')
y = sess.graph.get_tensor_by_name('y:0')

y_out = sess.run(y, {x: 3.0})
print y_out