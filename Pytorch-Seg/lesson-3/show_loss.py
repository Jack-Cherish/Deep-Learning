import matplotlib.pyplot as plt
# Jupyter notebook 中开启
# %matplotlib inline
with open('train_loss.txt', 'r') as f:
    train_loss = f.readlines()
    train_loss = list(map(lambda x:float(x.strip()), train_loss))
x = range(len(train_loss))
y = train_loss
plt.plot(x, y, label='train loss', linewidth=2, color='r', marker='o', markerfacecolor='r', markersize=5)
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()