# music-classification
#基于cnn处理GTZAN数据，实现音乐分类功能
##处理GTZAN中的原始音乐音频和csv文件，将其融合实现分类功能
##此版本实现过程：
###1,将GTZAN音频处理为mel图，和csv文件一起分割为8：1：1的训练验证测试文件。
###2,将训练数据输入CNN模型并得出训练后验证测试结果
##目前问题：出现过拟合现象，在验证集上准确率为70附近，此纪录暂时为最好版本
