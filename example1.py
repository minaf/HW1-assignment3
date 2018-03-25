import tensorflow as tf
x = tf.constant(2, name='x')
y = tf.constant(3, name='y')
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(op3))
writer.close()
