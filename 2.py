import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
X_train_mel = np.load('X_train_mel.npy')
X_val_mel = np.load('X_val_mel.npy')
X_test_mel = np.load('X_test_mel.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

# 加载从CSV提取的特征数据
X_train_features = np.load('X_train_features.npy', allow_pickle=True)
X_val_features = np.load('X_val_features.npy', allow_pickle=True)
X_test_features = np.load('X_test_features.npy', allow_pickle=True)

X_train_mel = X_train_mel.astype('float32')
X_val_mel = X_val_mel.astype('float32')
X_test_mel = X_test_mel.astype('float32')

X_train_features = X_train_features.astype('float32')
X_val_features = X_val_features.astype('float32')
X_test_features = X_test_features.astype('float32')

y_train = y_train.astype('int32')
y_val = y_val.astype('int32')
y_test = y_test.astype('int32')

# 在模型融合部分之前添加 Batch Normalization
cnn_branch = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 1300, 1)),
    MaxPooling2D((2, 6)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 6)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 6)),
    Flatten(),
    BatchNormalization(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4)
])

mlp_branch = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_features.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5)
])

combined = concatenate([cnn_branch.output, mlp_branch.output])
combined_output = Dense(10, activation='softmax')(combined)

model = Model(inputs=[cnn_branch.input, mlp_branch.input], outputs=combined_output)

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.0002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 模型摘要
model.summary()

# 训练模型
history = model.fit([X_train_mel, X_train_features], y_train, epochs=60, validation_data=([X_val_mel, X_val_features], y_val), batch_size=50)
# 保存模型
model.save('combine_model.h5')
# 评估模型
test_loss, test_acc = model.evaluate([X_test_mel, X_test_features], y_test)
print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')

# 预测测试集
y_pred = model.predict([X_test_mel, X_test_features])
y_pred_classes = np.argmax(y_pred, axis=1)

# 打印分类报告
report = classification_report(y_test, y_pred_classes, target_names=['蓝调', '经典', '乡村',
                                                                      '迪斯科', '嘻哈', '爵士',
                                                                      '金属', '流行', '雷鬼', '摇滚'])
print(report)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# 定义从整数索引到音乐流派的映射
int_to_category = {
    0: '蓝调',
    1: '经典',
    2: '乡村',
    3: '迪斯科',
    4: '嘻哈',
    5: '爵士',
    6: '金属',
    7: '流行',
    8: '雷鬼',
    9: '摇滚'
}

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=int_to_category.values(), yticklabels=int_to_category.values())
plt.xlabel('猜测类型')
plt.ylabel('实际类型')
plt.title('混淆矩阵')
plt.show()

# 可视化训练过程
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 绘制训练和验证的准确率
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='训练准确率')
plt.plot(epochs, val_acc, 'b', label='验证准确率')
plt.title('准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

# 绘制训练和验证的损失
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='训练损失')
plt.plot(epochs, val_loss, 'b', label='验证损失')
plt.title('损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.show()
