#-*- coding:utf-8 -*-
from urllib.request import urlretrieve
import time, random, os
 
class Discuz():
	def __init__(self):
		# Discuz验证码生成图片地址
		self.url = 'http://cuijiahua.com/tutrial/discuz/index.php?label='
 
	def random_captcha_text(self, captcha_size = 4):
		"""
		验证码一般都无视大小写；验证码长度4个字符
		Parameters:
			captcha_size:验证码长度
		Returns:
			captcha_text:验证码字符串
		"""
		number = ['0','1','2','3','4','5','6','7','8','9']
		alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
		char_set = number + alphabet
		captcha_text = []
		for i in range(captcha_size):
			c = random.choice(char_set)
			captcha_text.append(c)
		captcha_text = ''.join(captcha_text)
		return captcha_text
 
	def download_discuz(self, nums = 5000):
		"""
		下载验证码图片
		Parameters:
			nums:下载的验证码图片数量
		"""
		dirname = './Discuz'
		if dirname not in os.listdir():
			os.mkdir(dirname)
		for i in range(nums):
			label = self.random_captcha_text()
			print('第%d张图片:%s下载' % (i + 1,label))
			urlretrieve(url = self.url + label, filename = dirname + '/' + label + '.jpg')
			# 请至少加200ms延时，避免给我的服务器造成过多的压力，如发现影响服务器正常工作，我会关闭此功能。
			# 你好我也好，大家好才是真的好！
			time.sleep(0.2)
		print('恭喜图片下载完成！')
 
if __name__ == '__main__':
	dz = Discuz()
	dz.download_discuz()
