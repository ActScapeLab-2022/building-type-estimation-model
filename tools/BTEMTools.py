
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pydot
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import backend, models
from tensorflow.python.keras.callbacks import History

from TextCNN.learn import TextCNN
from WordConverter import WordConverter


class VecProcessor:
    def __init__(
        self, 
        path:Path, 
        isNormalize:bool, 
        outputSumVec:bool, 
        extractCount=100000, 
        splitRatio=0.1
        ) -> None:
        """
        ベクトル化処理を行う
        
        Parameters
        ----------
        path : | name | re_type_num | のヘッダーを持つcsvファイル

        isNormalize : 正規化を行うか

        outputSumVec : 各単語ベクトルの総和を返す（Random Forest）か，和を取らずに返す（TextCNN）か

        extractCount : 上位何件を抽出して学習するか
        
        splitRatio : 分割する場合に検証データに回す割合
        """
        # Processorの名前を登録
        self.name = path.stem

        # 正規化の有無
        self.isNormalize = isNormalize

        # データの読み込み
        sourseDF = pd.read_csv(str(path), encoding='utf-8')
        # 必要に応じて以下をコメントイン・アウトして，ソースデータをシャッフルする
        # （冒頭十万件を抽出）
        sourseDF = sourseDF.sample(frac=1, random_state=0).head(extractCount)
        print(f'Finish reading ({path.stem})')

        # メモリーエラー回避のために変数を新しく宣言しない
        # self.sourceNames = sourseDF['name'].values
        # self.types = (sourseDF['re_type_num'].values - 1)

        self.trainDF = sourseDF.head(int(extractCount*(1-splitRatio)))
        # TODO: 一時的にテストデータをすべてShizuoka_TelePointにしている
        self.testDF  = sourseDF.tail(int(extractCount*splitRatio))
        # self.testDF  = pd.read_csv(str(sourcePath/'cities'/'Shizuoka'/'Shizuoka.csv')).sample(frac=1, random_state=0).tail(int(31464*splitRatio))
        self.train_converter, self.train_renames, self.train_vecs = self.__readyVec(self.trainDF, outputSumVec)
        self.test_converter, self.test_renames, self.test_vecs = self.__readyVec(self.testDF, outputSumVec)


    def __readyVec(self, df:pd.DataFrame, outputSumVec):
        """
        建物名称データの入ったデータセットのパスを受け取り，ベクトルに変換する
        一番最初にこのメソッドを呼ぶ必要あり
        """
        # 予測するデータをベクトルに変換
        # TextCNNでは単語単位にばらしたベクトルを１つにまとめない
        converter = WordConverter(df, self.isNormalize, outputSumVec)

        print("Start to convert")
        train_renames, train_vecs = converter.convert()
        print("Finish the vectalize")

        return converter, train_renames, train_vecs

    def readFile(self, readPath:str):
        raise NotImplementedError

    def saveFile(self, savePath:str, trainData=True):
        if trainData:
            self.train_converter.writeResult(savePath)
        else:
            self.test_converter.writeResult(savePath)

class TCNNBase:
    """
    TextCNNを学習し，保存，読み込み処理を実装するための基底クラス
    """
    def __init__(self, test:VecProcessor) -> None:
        self.model:TextCNN
        self.testData = test

    def showAbs(self):
        self.model.summary()

    def savePredict(self, savePath:str) -> np.ndarray:
        """
        テストデータをクラスが所有するモデルで予測し，その結果を保存する
        予測結果を返す
        """
        predicted:np.ndarray = self.model.predict(self.testData.test_vecs)
        columns = ['name', 'answer', 'predict', 'shougyou', 'zigyou', 'public', 'religion', 'parking']
        predictDF = pd.DataFrame({
            columns[0] : self.testData.testDF['name'].values,
            columns[1] : self.testData.testDF['re_type_num'].values,
            columns[2] : predicted.argmax(axis=1) + 1,
            columns[3] : predicted[:, 0],
            columns[4] : predicted[:, 1],
            columns[5] : predicted[:, 2],
            columns[6] : predicted[:, 3],
            columns[7] : predicted[:, 4]
        })

        predictDF.to_csv(savePath, index=False, encoding='utf-8-sig')
        return predicted
    
    def saveGraph(self, savePath:str, title:str):
        _, ax_loss = plt.subplots()
        ax_acc = ax_loss.twinx()
        
        ax_loss.set_title(title)

        ax_loss.plot(self.epochs, self.loss, color='blue', label='training loss')
        ax_loss.plot(self.epochs, self.val_loss, color='orange', label='test loss')
        ax_acc.plot(self.epochs, self.accuracy, color='blue', linestyle=':', label='training accuracy')
        ax_acc.plot(self.epochs, self.val_accuracy, color='orange', linestyle=':', label='test accuracy')

        ax_loss.set_xlabel('Epoch', fontsize=14)
        ax_loss.set_ylabel('Loss', fontsize=14)
        ax_acc.set_ylabel('Accuracy', fontsize=14)

        h1, l1 = ax_loss.get_legend_handles_labels()
        h2, l2 = ax_acc.get_legend_handles_labels()
        ax_loss.legend(h1+h2, l1+l2, loc='center right')

        # テキストをテキストとして出力する設定
        plt.rcParams["svg.fonttype"] = "none"
        plt.savefig(savePath)
        # plt.show()

class TCNNLearner(TCNNBase):
    """
    TextCNNの学習に特化したクラス
    """
    def __init__(self, train:VecProcessor, test:VecProcessor, epoch=10) -> None:
        super().__init__(test)
        self.trainName = train.name
        self.epoch = epoch
        self.epochs = [i+1 for i in range(epoch)]
        self.isNormalize = test.isNormalize

        self.model = TextCNN(
            max_len=train.train_vecs[0].shape[0],
            output_dim=5,
            filter_sizes=[3,3],
            num_filters=100,
            dropout_rate=0.33,
            regularizers_lambda=0.01
            )

        self.model.compile(
            optimizer = 'adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
            )

        self.__learn(train)

    def __learn(self, trainData:VecProcessor):
        """
        学習メソッド
        本来はlearnかreadを呼ぶと結果に該当するインスタンスを返すべきだけど，面倒なので実装しない
        """
        cvs:History = self.model.fit(
            trainData.train_vecs, 
            trainData.trainDF['re_type_num'].values - 1, 
            validation_data=(self.testData.test_vecs, self.testData.testDF['re_type_num'].values - 1), 
            epochs=self.epoch, 
            batch_size=None
            )

        self.loss:list[float] = cvs.history['loss']
        self.accuracy:list[float] = cvs.history['accuracy']
        self.val_loss:list[float] = cvs.history['val_loss']
        self.val_accuracy:list[float] = cvs.history['val_accuracy']

    def saveAll(self, trainType:str):
        """
        モデルなどの保存を一元的に行う\n
        Fileには「拡張子を含めない形」で指定する
        """
        trainFile = self.trainName
        testFile = self.testData.name
        normName = "_Normalize" if isNormalize else ""
        resultPath = Path(__file__).parent/'TextCNN'/'result'/trainType/(trainFile+normName)
        resultPath.mkdir(parents=True, exist_ok=True)
        # imagePath  = Path(__file__).parent/'TextCNN'/'image'/f'{trainFile}2{testFile}{normName}.png'
        imagePath  = Path(__file__).parent/'TextCNN'/'image'/f'{trainFile}2{testFile}{normName}.svg'

        # self.showAbs()
        self.saveModel(str(resultPath))
        self.saveNums(str(resultPath/f'Nums_{testFile}{normName}.csv'))
        # メモリエラー回避のために除外
        # self.savePredict(str(resultPath/f'pred_{testFile}.csv'))
        self.saveGraph(str(imagePath), f'Result of Learning in {trainFile}')

    def saveModel(self, savePath:str):
        self.model.save(savePath)
        print(f'Saved the model data at {savePath}')

    def saveNums(self, savePath:str):
        """
        Lossなどの学習結果を保存する
        """
        index = [f'Epoch {i+1}' for i in range(self.epoch)]
        columns = ['Loss', 'val_Loss', 'Accuracy', 'val_Accuracy']
        numsDF = pd.DataFrame({
            columns[0] : self.loss,
            columns[1] : self.val_loss,
            columns[2] : self.accuracy,
            columns[3] : self.val_accuracy
        }, index=index)

        numsDF.to_csv(savePath)

    def dispose(self):
        """
        学習を繰り返し回す場合に不要になったモデルオブジェクトを削除することでメモリ使用量を抑える\n
        ガベージコレクションを作動させるが，その代わりこのオブジェクトが使用不可になるため，取扱注意
        """
        del self.model
        backend.clear_session()
        gc.collect()

class TCNNReader(TCNNBase):
    """
    TextCNNの読み込み，予測に特化したクラス
    """
    def __init__(self, readType:str, readFileName:str, isNormalize:bool, test: VecProcessor) -> None:
        super().__init__(test)
        normName = "_Normalize" if isNormalize else ""
        readPath = Path(__file__).parent/'TextCNN'/'result'/readType/(readFileName+normName)
        self.__readTrained(readPath)

    def __readTrained(self, readPath:Path):
        """
        学習済みのデータを読み込む
        Lossなどの数値データが保存されていたら合わせて読み込む（ようにしたい）
        自動探索できるよう，Lossなどの保存場所はモデルフォルダの中にすべき？
        saveGraphは学習後に呼ぶことを前提としているため，使用できない
        """
        self.model = models.load_model(str(readPath))
        testFile = self.testData.name
        normName = "_Normalize" if isNormalize else ""
        numberData = readPath/f'Nums_{testFile}{normName}.csv'
        if numberData.exists():
            resultCsv = pd.read_csv(str(numberData))
            self.loss = resultCsv['Loss']
            self.val_loss = resultCsv['val_Loss']
            self.accuracy = resultCsv['Accuracy']
            self.val_accuracy = resultCsv['val_Accuracy']
            self.epochs = [i+1 for i in range(self.loss.size)]