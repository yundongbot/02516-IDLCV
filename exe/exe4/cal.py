import re

# 将您的文本赋值给变量 text
text = """
Validation Accuracy: 0.2583
Epoch 1/30, Loss: 35.3404
Validation Accuracy: 0.3833
Epoch 2/30, Loss: 28.1885
Validation Accuracy: 0.5000
Epoch 3/30, Loss: 18.5246
Validation Accuracy: 0.6250
Epoch 4/30, Loss: 9.5245
Validation Accuracy: 0.6583
Epoch 5/30, Loss: 4.3202
Validation Accuracy: 0.6833
Epoch 6/30, Loss: 2.4157
Validation Accuracy: 0.6917
Epoch 7/30, Loss: 1.3293
Validation Accuracy: 0.6917
Epoch 8/30, Loss: 0.8717
Validation Accuracy: 0.6833
Epoch 9/30, Loss: 0.6646
Validation Accuracy: 0.7000
Epoch 10/30, Loss: 0.3979
Validation Accuracy: 0.7250
Epoch 11/30, Loss: 0.3494
Validation Accuracy: 0.7167
Epoch 12/30, Loss: 0.1986
Validation Accuracy: 0.6667
Epoch 13/30, Loss: 0.2735
Validation Accuracy: 0.7083
Epoch 14/30, Loss: 0.1686
Validation Accuracy: 0.7333
Epoch 15/30, Loss: 0.2129
Validation Accuracy: 0.7167
Epoch 16/30, Loss: 0.1500
Validation Accuracy: 0.7167
Epoch 17/30, Loss: 0.1474
Validation Accuracy: 0.6833
Epoch 18/30, Loss: 0.1372
Validation Accuracy: 0.7000
Epoch 19/30, Loss: 0.1214
Validation Accuracy: 0.7000
Epoch 20/30, Loss: 0.0786
Validation Accuracy: 0.7000
Epoch 21/30, Loss: 0.0938
Validation Accuracy: 0.7000
Epoch 22/30, Loss: 0.0957
Validation Accuracy: 0.7500
Epoch 23/30, Loss: 0.0928
Validation Accuracy: 0.7083
Epoch 24/30, Loss: 0.0568
Validation Accuracy: 0.7417
Epoch 25/30, Loss: 0.0598
Validation Accuracy: 0.7250
Epoch 26/30, Loss: 0.0864
Validation Accuracy: 0.7167
Epoch 27/30, Loss: 0.0848
Validation Accuracy: 0.7333
Epoch 28/30, Loss: 0.0554
Validation Accuracy: 0.7417
Epoch 29/30, Loss: 0.0489
Validation Accuracy: 0.7167
Epoch 30/30, Loss: 0.1100
"""

# 使用正则表达式提取所有的 Validation Accuracy 值
accuracy_values = re.findall(r'Validation Accuracy: ([0-9.]+)', text)

# 将字符串列表转换为浮点数列表
accuracy_values = [float(value) for value in accuracy_values]

# 计算最大值和平均值
max_accuracy = max(accuracy_values)
avg_accuracy = sum(accuracy_values) / len(accuracy_values)

# 打印结果
print(f"最大 Validation Accuracy: {max_accuracy:.4f}")
print(f"平均 Validation Accuracy: {avg_accuracy:.4f}")
