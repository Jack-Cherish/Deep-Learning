from tensorboardX import SummaryWriter
from urllib.request import urlretrieve
import cv2

# 选择运行那个示例
choose_example = 1

if choose_example == 1:
    """
    Example 1：创建 writer 示例
    """
    # 创建 writer1 对象
    # log 会保存到 runs/exp 文件夹中
    writer1 = SummaryWriter('runs/exp')

    # 使用默认参数创建 writer2 对象
    # log 会保存到 runs/日期_用户名 格式的文件夹中
    writer2 = SummaryWriter()

    # 使用 commet 参数，创建 writer3 对象
    # log 会保存到 runs/日期_用户名_resnet 格式的文件中
    writer3 = SummaryWriter(comment='_resnet')

if choose_example == 2:
    """
    Example 2：写入数字示例
    """
    writer = SummaryWriter('runs/scalar_example')
    for i in range(10):
        writer.add_scalar('quadratic', i**2, global_step=i)
        writer.add_scalar('exponential', 2**i, global_step=i)
    writer.close()
    
if choose_example == 3:
    """
    Example 3：写入图片示例
    """
    urlretrieve(url = 'https://raw.githubusercontent.com/Jack-Cherish/Deep-Learning/master/Pytorch-Seg/lesson-2/data/train/label/0.png',filename = '1.jpg')
    urlretrieve(url = 'https://raw.githubusercontent.com/Jack-Cherish/Deep-Learning/master/Pytorch-Seg/lesson-2/data/train/label/1.png',filename = '2.jpg')
    urlretrieve(url = 'https://raw.githubusercontent.com/Jack-Cherish/Deep-Learning/master/Pytorch-Seg/lesson-2/data/train/label/2.png',filename = '3.jpg')

    writer = SummaryWriter('runs/image_example')
    for i in range(1, 4):
        writer.add_image('UNet_Seg',
                         cv2.cvtColor(cv2.imread('{}.jpg'.format(i)), cv2.COLOR_BGR2RGB),
                         global_step=i,
                         dataformats='HWC')
    writer.close()