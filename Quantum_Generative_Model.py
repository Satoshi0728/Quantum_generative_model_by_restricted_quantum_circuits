import cupy as np
import pandas as pd
import os
# MNIST Datasetの取得
from keras.datasets import mnist
import matplotlib.pyplot as plt
import time
import requests


class QGM:
    """
    量子生成モデル (Quantum Generative Model)
    """
    def __init__(self, data, number_of_dot = 784, seed = 0, optimize_method = None , number_of_original_image = 0, init_theta_multi = 1,):
        """Quantum Inspired Machine
        Attributes:
        seed : 乱数のシード
        data : データ
        n_dots  : 画像のドット数
        optimize_method : 最適化方法
        number_of_original_image もともとの画像の何番目の画像を出力するかを指定
        theta : 角度の値
        """
        np.random.seed(seed)
        self.data = data #データの読み込み
        self.n_dots = number_of_dot #n_dots はnumber of dotsを表す(画像のドットの数)
        self.optimize_method = optimize_method #最適化方法
        self.n_original_img = number_of_original_image
        #thetaの初期値 0 ~ 2piの乱数で設定
        self.theta = np.tril(init_theta_multi *2*np.pi * np.random.rand(number_of_dot, number_of_dot))

    def get_probabilities(self, data, theta):
        """
        確率の計算: x_1から決めて行って、x_nまでの確率を出してくれる
        """
        #角度計算
        theta_diag = np.diag(np.diag(theta)) #対角行列
        theta_tril = np.tril(theta, k = -1) #下三角行列(対角行列成分は0)
        data_diag = np.diag(data) #対角行列に変換
        angle_mat = (theta_tril @ data_diag + theta_diag )/2 #角度を成分に持つ下三角行列
        angle = np.sum(angle_mat, axis = 1) #角度を成分に持つベクトル
        #cosとsinの計算
        cos = np.cos(angle)**2
        sin = np.sin(angle)**2
        probabilities = data * sin + (1 - data) * cos
        return probabilities

    def log_likelihood_func(self, data, theta):
        """
        対数尤度関数
        """
        prob = self.get_probabilities(data, theta)
        log_likelihood = np.sum(np.log(prob))
        return -log_likelihood

    def get_samples(self, theta):
        """
        最適化後のシータの値から定まる確率分布に従ってデータをサンプリングする
        """
        theta_diag = np.diag(np.diag(theta)) #対角行列
        theta_tril = np.tril(theta, k = -1) #下三角行列(対角行列成分は0)
        #リストの作成
        probabilities_optimized = np.zeros_like(self.data[0], dtype=float)
        samples = np.zeros_like(theta)
        #サンプリング
        for i in range(len(self.data[0])):
            #角度計算
            angle_mat = (theta_tril @ samples + theta_diag )/2 #角度を成分に持つ下三角行列
            angle = np.sum(angle_mat, axis = 1) #角度を成分に持つベクトル
            #確率の計算
            probabilities_optimized[i] = np.sin(angle[i])**2
            samples[i][i] = int((np.random.rand(1) < probabilities_optimized[i]).astype('float32')) #乱数がprobability_opt以下のとき1になって、それ以上の時は0を出力
        self.probabilities_opt = probabilities_optimized
        #一次元化
        samples = np.diag(samples)
        return samples


#######################################################################　最適化方法系
    """
    複数の学習方法の構築  Reference(https://tech-lab.sios.jp/archives/21823)
    """

    def grad_calculation(self, data, theta):
        """
        偏微分 厳密解Ver.
        """
        #角度計算
        theta_diag = np.diag(np.diag(theta)) #対角行列
        theta_tril = np.tril(theta, k = -1) #下三角行列(対角行列成分は0)
        data_diag = np.diag(data) #対角行列に変換
        angle_list = (theta_tril @ data_diag + theta_diag ) #角度を成分に持つ下三角行列
        angle = np.sum(angle_list, axis = 1) #角度を成分に持つベクトル
        ones = np.ones(self.n_dots) #成分1のベクトル
        #TODO
        angle = angle % (2* np.pi)

        # # TODO 勾配発散を回避
        # #cosとsinの計算
        # cos = np.cos(angle)
        # angle = np.where(data * cos  == 1 , 2 * np.pi * np.random.rand(1) , angle) #分母0となる時の要素に対して、ランダムなシータで初期化する

        cos = np.cos(angle)
        sin = np.sin(angle)
        #勾配計算
        #gradの対角成分
        grad_diag_flat = ((ones - 2 * data) * sin ) / (ones + (ones - 2 * data) * cos )
        grad_diag = np.diag(grad_diag_flat)
        #gradの非対角成分
        grad_tril = data * np.tril(np.tile(grad_diag_flat, (len(grad_diag_flat),1)).T, k = -1)
        grad = grad_diag + grad_tril
        return grad


    def SGD(self, theta, data_minibatch):
        """
        確率的勾配降下法（Stochastic Gradient Descent）
        """
        grad = np.zeros_like(theta)
        for data in data_minibatch:
            grad += self.grad_calculation(data, theta)
        theta -= self.lr*grad
        #TODO
        theta = theta % (2 * np.pi)
        return  theta

    def Momentum(self, theta, data_minibatch ,momentum = 0.9):
        """
        Momentum SGD
        """
        grad = np.zeros_like(theta)
        v = np.zeros_like(theta)
        for data in data_minibatch:
            grad += self.grad_calculation(data, theta)
        v = v*momentum - self.lr*grad
        theta += v
        #TODO
        theta = theta % (2 * np.pi)
        return  theta

    def AdaGrad(self, theta, data_minibatch, epsilon = 1e-7):
        """
        AdaGrad
        """
        h = np.zeros_like(theta)
        grad = np.zeros_like(theta)
        for data in data_minibatch:
            grad += self.grad_calculation(data, theta)
        h += grad**2
        theta -= (self.lr*grad) / (np.sqrt(h) + epsilon)
        #TODO
        theta = theta % (2 * np.pi)
        return  theta

    def RMSprop(self, theta, data_minibatch, epsilon = 1e-7, rho = 0.9):
        """
        ヒントンさん考案らしい
        """
        h = np.zeros_like(theta)
        grad = np.zeros_like(theta)
        for data in data_minibatch:
            grad += self.grad_calculation(data, theta)
        h  = rho*h + (1 - rho)*(grad**2)
        theta -= (self.lr*grad) / (np.sqrt(h) + epsilon)
        #TODO
        theta = theta % (2 * np.pi)
        return  theta

    def Adam(self, theta, data_minibatch, epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        """
        adam
        """
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        grad = np.zeros_like(theta)
        for data in data_minibatch:
            grad += self.grad_calculation(data, theta)
        m = beta_1 * m + (1- beta_1)* grad
        v = beta_2 * v + (1- beta_2)* (grad**2)
        om = m / (1 - beta_1)
        ov = v / (1 - beta_2)
        theta -= (self.lr*om) / np.sqrt(np.around(ov + epsilon, 10))
        #TODO
        theta = theta % (2 * np.pi)
        return  theta

####################################################################### 学習の設定系の関数
    def minibatch_learning(self, init_theta):
        """
        ミニバッチ学習
        """
        theta = init_theta
        #学習回数記録　   epoch
        self.epoch_num = 0
        # #ミニバッチの設定
        data_random = np.random.permutation(self.data)

        #学習方法を決定
        if self.optimize_method == "SGD":
            optimizer = self.SGD
        elif self.optimize_method == "Momentum":
            optimizer = self.Momentum
        elif self.optimize_method == "AdaGrad":
            optimizer = self.AdaGrad
        elif self.optimize_method == "RMSprop":
            optimizer = self.RMSprop
        elif self.optimize_method == "Adam":
            optimizer = self.Adam

        #KeyboardInterruptで途中までの実行結果を出力
        key_not_interrupt = True
        try:
            while(key_not_interrupt):
                for num in range(self.epoch):
                    #何batch目なのかを記録
                    self.minibatch_count = 0
                    #尤度関数がある値以下になった時に自動停止するためのフラッグ
                    flag_stop = False
                    #ミニバッチの設定
                    # data_random = np.random.permutation(self.data)
                    #バッチを選択する場所の初期値
                    minibatch_start_num =0
                    for minibatch_end_num in range(self.batch_size , len(self.data) + self.batch_size , self.batch_size):
                        #ミニバッチはデータの順番を入れ替えた data_random から順番にバッチサイズの大きさだけ取得する。
                        data_minibatch = data_random[minibatch_start_num : minibatch_end_num]
                        #ミニバッチを選択する場所の更新
                        minibatch_start_num = minibatch_end_num

                        #学習
                        theta = optimizer(theta ,  data_minibatch)

                        #尤度関数計算
                        log_likelihood = 0
                        for data in data_minibatch:
                            log_likelihood += self.log_likelihood_func(data ,theta)
                        print("%s epoch目, %sbatch目の尤度関数は %s です。" %(num, self.minibatch_count ,log_likelihood))
                        #ミニバッチの回数更新
                        self.minibatch_count += 1
                        #尤度関数がある値以下になった時に停止
                        if log_likelihood < self.cost_limit:
                            flag_stop = True
                            break

                    #学習回数記録
                    self.epoch_num += 1
                    #尤度関数を記録
                    self.log[num]  = log_likelihood
                    #尤度関数がある値以下になった時に停止
                    if flag_stop:
                        break
                key_not_interrupt = False
        except KeyboardInterrupt:
            pass
        #最終的な尤度関数の値
        self.final_log_likelihood = log_likelihood
        return  theta


    def learning(self, batch_size, epoch = 100, lr = 0.01, cost = None):
        """
        学習
        """
        ##更新の設定
        self.epoch = epoch
        self.lr = lr
        #バッチサイズ
        self.batch_size = batch_size
        #尤度関数の減衰を記録
        self.log = np.zeros(epoch)
        #コスト関数の上限設定
        if cost != None:
            self.cost_limit = cost
        else:
            self.cost_limit = 0

        ##最適化時間の計測
        start = time.time()
        #メインの学習
        theta_opt = self.minibatch_learning(self.theta)

        #学習にかかる時間
        self.learning_time = time.time() - start
        #最適化されたシータ
        self.theta_opt = theta_opt
        #サンプリング
        self.samples = self.get_samples(theta_opt)

####################################################################### その他
    def show(self, get_image = False):
        """
        画像を生成
        """
        #もともとの画像を生成
        image_original = self.data[self.n_original_img].reshape(int(np.sqrt(self.n_dots)), int(np.sqrt(self.n_dots)))
        plt.figure(figsize=(10,20))
        plt.imshow(image_original, cmap='gray')
        plt.grid(False)
        plt.axis('off')
        plt.show()

        #復元後の画像を生成
        image_generated = self.samples.reshape(int(np.sqrt(self.n_dots)), int(np.sqrt(self.n_dots)))
        plt.figure(figsize=(10,20))
        plt.imshow(image_generated, cmap='gray')
        plt.grid(False)
        plt.axis('off')
        plt.show()

        if get_image:
            plt.savefig('img/original.png', dpi=300)
            plt.savefig('img/generated.png', dpi=300)

    def get_datas(self, number , number_of_picture ,dir_name = 'QGM_data'):
        """
        計算結果をまとめたディレクトリを作成
            以下に注意
            ・コンパイル時に指定のディレクトリまで移動すること VScodeならhomeディレクトリからでも実行できてしまうから,この場合、homeディレクトリに計算結果のまとめが出力される.
                ~/python_calculation/research/hoge/QGM.py ならば cd ~/python_calculation/research/hogeしてから実行すること.
            ・ 初期設定の値 , 最適化後の回転角 , 尤度関数の減少の記録  を出力してくれる.
        """
        #ディレクトリの作成
        def make_dirs(path):
            if not os.path.isdir(path):
                os.makedirs(path)
        #パスの設定
        dir_name = os.getcwd() +'/' + dir_name
        make_dirs(dir_name)
        path_w = dir_name +'/' + 'information.txt'
        #情報の設定
        information = ("MNISTの数字は %s です。\n" %number)\
                    + ("画像の枚数は %s です。\n" %number_of_picture)\
                    + ("学習時間は %s です。\n" %self.learning_time)\
                    + ("最適化方法 %s \n epoch %s \n 学習率 %s \n バッチサイズ %s \n" %(self.optimize_method, self.epoch ,self.lr, self.batch_size))\
                    + ("最適化後の尤度関数は %s です。\n" %self.final_log_likelihood)\
                    + ("学習回数は epoch %s 回 , minibatchの %s 回です。\n" %(self.epoch_num, self.minibatch_count))
        with open(path_w, mode='xt') as f:
            f.write(information)
        #最適化後のシータの値を書き込み
        pd.to_pickle(self.theta_opt, dir_name +'/' + 'theta_opt.pkl')
        #尤度関数のepochごとの減少率の記録を書き込み
        pd.to_pickle(self.log , dir_name +'/' + 'log.pkl' )


    def slack_notify(self, msg = 'done'):
        slack_user_id = 'hoge'
        slack_webhook_url = 'hoge'
        requests.post(slack_webhook_url, json={"text":msg})
