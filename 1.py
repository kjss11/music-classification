import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 定义数据集数据目录和图像保存目录，改成你们自己的
data_dir = 'D:\\ASR\\Data\\genres_original'
image_dir = 'D:\\ASR\\Data\\genres_images'

# 加载30秒CSV数据
csv_file_path = 'features_30_sec.csv'
data = pd.read_csv(csv_file_path)

# 检查并创建图像目录
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 定义所有音乐流派
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
category_to_int = {genre: i for i, genre in enumerate(genres)}  # 类别到整数的映射

# 数据和标签容器
X_train_mel = []
X_val_mel = []
X_test_mel = []
y_train = []
y_val = []
y_test = []

# 特征数据容器
X_train_features = []
X_val_features = []
X_test_features = []

# 定义目标Mel Spectrogram的长度
target_length = 1300

# 从音频文件中提取Mel Spectrogram特征
def extract_features(file_path, max_duration=30):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=max_duration)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    db_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # 归一化Mel Spectrogram到0-1范围
    normalized_mel_spectrogram = (db_mel_spectrogram - np.min(db_mel_spectrogram)) / (np.max(db_mel_spectrogram) - np.min(db_mel_spectrogram))
    # 确保所有Mel Spectrogram的长度一致
    if normalized_mel_spectrogram.shape[1] < target_length:
        pad_amount = target_length - normalized_mel_spectrogram.shape[1]
        normalized_mel_spectrogram = np.pad(normalized_mel_spectrogram, ((0, 0), (0, pad_amount)), mode='constant', constant_values=(0, 0))
    else:
        normalized_mel_spectrogram = normalized_mel_spectrogram[:, :target_length]
    return normalized_mel_spectrogram, sample_rate

# 读取数据和标签，并保存Mel Spectrogram图像
for genre in genres:
    genre_dir = os.path.join(data_dir, genre)
    genre_image_dir = os.path.join(image_dir, genre)
    if not os.path.exists(genre_image_dir):
        os.makedirs(genre_image_dir)

    files = sorted([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
    train_files = files[:80]  # 对应的80%的训练数据
    val_files = files[80:90]  # 对应的10%的验证数据
    test_files = files[90:]   # 对应的10%的测试数据

    for filename in train_files:
        file_path = os.path.join(genre_dir, filename)
        mel_spectrogram, sample_rate = extract_features(file_path)
        if mel_spectrogram is not None:
            X_train_mel.append(mel_spectrogram)
            y_train.append(category_to_int[genre])  # 使用映射转换标签

            # 从CSV文件中获取对应的特征
            features_row = data[(data['filename'] == filename)].iloc[0, 1:-1].values
            X_train_features.append(features_row)

    for filename in val_files:
        file_path = os.path.join(genre_dir, filename)
        mel_spectrogram, sample_rate = extract_features(file_path)
        if mel_spectrogram is not None:
            X_val_mel.append(mel_spectrogram)
            y_val.append(category_to_int[genre])

            # 从CSV文件中获取对应的特征
            features_row = data[(data['filename'] == filename)].iloc[0, 1:-1].values
            X_val_features.append(features_row)

    for filename in test_files:
        file_path = os.path.join(genre_dir, filename)
        mel_spectrogram, sample_rate = extract_features(file_path)
        if mel_spectrogram is not None:
            X_test_mel.append(mel_spectrogram)
            y_test.append(category_to_int[genre])

            # 从CSV文件中获取对应的特征
            features_row = data[(data['filename'] == filename)].iloc[0, 1:-1].values
            X_test_features.append(features_row)

# 将数据列表转换为numpy数组以便于处理
X_train_mel = np.array(X_train_mel).reshape(-1, 128, target_length, 1)
X_val_mel = np.array(X_val_mel).reshape(-1, 128, target_length, 1)
X_test_mel = np.array(X_test_mel).reshape(-1, 128, target_length, 1)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

X_train_features = np.array(X_train_features)
X_val_features = np.array(X_val_features)
X_test_features = np.array(X_test_features)

# 保存处理后的数据
np.save('X_train_mel.npy', X_train_mel)
np.save('X_val_mel.npy', X_val_mel)
np.save('X_test_mel.npy', X_test_mel)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

np.save('X_train_features.npy', X_train_features)
np.save('X_val_features.npy', X_val_features)
np.save('X_test_features.npy', X_test_features)

# 输出数据形状
print(f'Train set size: {X_train_mel.shape}, Validation set size: {X_val_mel.shape}, Test set size: {X_test_mel.shape}')
print(f'Feature Train set size: {X_train_features.shape}, Validation set size: {X_val_features.shape}, Test set size: {X_test_features.shape}')
