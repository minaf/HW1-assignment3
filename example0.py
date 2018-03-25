import tensorflow as tf
x = 3
y = 5
a = tf.add(x, y)
print(a)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(a))
writer.close()

