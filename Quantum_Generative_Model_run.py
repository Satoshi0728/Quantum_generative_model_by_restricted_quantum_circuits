import numpy as np
# MNIST Datasetの取得
from keras.datasets import mnist
from Quantum_Generative_Model import QGM

"""
MNISTの設定
"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784) # 1次元に変換
X_train = X_train.astype('float32') #float32型に変換
X_train /= 255                      # 0.0-1.0に変換
X_train = np.asarray(X_train) #cupyを使うために型の変換をしている. CPU で計算するならいらない.
X_train = np.where(X_train <= 0.5, 0.0, 1.0) # 0と1に変換

#データセットの中身を同じ数字の画像にするための関数
def MNIST_reshape(number , number_of_pictures, numbers_list = y_train):
    data_MNIST = []
    cnt = 0
    for i in range(len(numbers_list)):
        if numbers_list[i] == number:
            data_MNIST.append(X_train[i])
            cnt += 1
        if cnt == number_of_pictures:
            break
    data_MNIST = np.array(data_MNIST)
    return data_MNIST


"""
Learning_Method :   Method              Recommended hyperparameters
                    "SGD"                      lr = 0.01
                    "Momentum"             lr = 0.01, momentum = 0.9
                    "AdaGrad"              lr = 0.01, epsolon = 1e^-7
                    "RMSprop"              lr = 0.001,   rho = 0.9,   psolon = 1e^-7
                    "Adam"                 lr = 0.001,   beta1 = 0.9,  beta2 = 0.999 , epsolon = 1e^-7
epoch : 学習回数
"""


"""
パラメーター設定
"""
#画像の枚数
number_of_pictures =100
#画像に写ってる番号
number = 8
#学習方法
Learning_Method = "Adam"
#学習率
lr = 0.001
#Epoch
epoch = 10000
#初期値シータの定数倍 1倍にすると0 ~ 2pi のスケールでランダムに初期化される.
init_theta_multi = 1
#尤度関数がこの値を下回ると,自動で学習が終わる.
cost_limit = 0.001
#ミニバッチのサイズ
batch_size = number_of_pictures


#同じ数字のデータセットか色々な数字のデータセットかを選択
data_MNIST = MNIST_reshape(number , number_of_pictures) #同じ数字のデータ
# data_MNIST = np.array([X_train[i] for i in range(number_of_pictures)]) #違う数字のデータ

#インスタンス化
qgm = QGM(data_MNIST , seed = None, number_of_original_image = 0 , optimize_method = Learning_Method, init_theta_multi= init_theta_multi)
#学習
qgm.learning(batch_size, epoch , lr, cost = cost_limit)

qgm.get_datas(number , number_of_pictures , dir_name = "hoge" )
# スラック通知
qgm.slack_notify(msg = 'calculation done')



