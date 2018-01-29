#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import train

def crack_captcha(captcha_image, captcha_label):
	"""
	使用模型做预测
	Parameters:
		captcha_image:数据
		captcha_label:标签
	"""

	output = dz.crack_captcha_cnn()
	saver = tf.train.Saver()
	with tf.Session(config=dz.config) as sess:

		saver.restore(sess, tf.train.latest_checkpoint('.'))
		for i in range(len(captcha_label)):
			img = captcha_image[i].flatten()
			label = captcha_label[i]
			predict = tf.argmax(tf.reshape(output, [-1, dz.max_captcha, dz.char_set_len]), 2)
			text_list = sess.run(predict, feed_dict={dz.X: [img], dz.keep_prob: 1})
			text = text_list[0].tolist()
			vector = np.zeros(dz.max_captcha*dz.char_set_len)
			i = 0
			for n in text:
					vector[i*dz.char_set_len + n] = 1
					i += 1
			prediction_text = dz.vec2text(vector)
			print("正确: {}  预测: {}".format(dz.vec2text(label), prediction_text))

if __name__ == '__main__':
	dz = train.Discuz()
	batch_x, batch_y = dz.get_next_batch(False, 5)
	crack_captcha(batch_x, batch_y)
