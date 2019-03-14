#定义两个数组
TrainLoss_list = []
TrainAcc_list = []

ValidateLoss_list = []
ValidateAcc_list = []
epochs = 220
#
# Loss_list.append(train_loss / (len(train_dataset)))
# Accuracy_list.append(100 * train_acc / (len(train_dataset)))
#
# #我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
# x1 = range(0, 200)
# x2 = range(0, 200)
# y1 = Accuracy_list
# y2 = Loss_list
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('Test accuracy vs. epoches')
# plt.ylabel('Test accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('Test loss vs. epoches')
# plt.ylabel('Test loss')
# plt.show()
# plt.savefig("accuracy_loss.jpg")