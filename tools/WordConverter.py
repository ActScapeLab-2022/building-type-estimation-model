import copy
import re
from pathlib import Path

import MeCab
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors, Word2VecKeyedVectors
from tqdm import tqdm

# Path一覧
sourcePath = Path(__file__).parents[1]/'Source'
savePath = Path(__file__).parent
entityDataPath = str(sourcePath/'entity_vector.model.bin')


class SplitBuildingName:
    """
    MeCabの出力を独自クラスとして保存する\n
    EOSで区切られた１パートを変換することができる

    example)\n
    line = \n
    'コニカミノルタ  名詞,固有名詞,組織,*,*,*,コニカミノルタ,コニカミノルタ,コニカミノルタ\n
    ビジネスソリューションズ        名詞,固有名詞,一般,*,*,*,*\n
    (株)    名詞,一般,*,*,*,*,(株),カブシキガイシャ,カブシキガイシャ'
    """
    def __init__(self, lines:list[str]) -> None:
        self.allNameAndInfo = list(map(Word, lines))
        self.allNames = list(map(lambda x: x.name, self.allNameAndInfo))

    # ifの同値判定
    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SplitBuildingName):
            return NotImplemented
        return self.allNames == __o.allNames

    # 内部関数の同値判定
    def __hash__(self) -> int:
        return hash(','.join(self.allNames))

    # print, listの表示用
    def __repr__(self) -> str:
        return ' | '.join(self.allNames)

class Word:
    """
    SplitBuildingNameのパーツクラス
    分解後の名称とその属性情報を保存する
    """
    def __init__(self, line:str) -> None:
        split = line.split('\t')
        self.name = split[0]
        self.info = split[1].split(',')

class WordConverter:
    def __init__(self, isNormalize: bool, outputOneVec: bool) -> None:
        self.isRegulation = isNormalize
        self.outputOneVec = outputOneVec
        self.model:Word2VecKeyedVectors = KeyedVectors.load_word2vec_format(entityDataPath, binary=True, unicode_errors='ignore')
        self.tagger = MeCab.Tagger()
        self.zeroVec = [0 for _ in range(200)]

    def convert(self, df: pd.DataFrame, readColumnName='name') -> tuple[list[str], np.ndarray]:
        """
        受け取るDataFrameは以下のフィールドを持つ
        name | re_type_num

        出力するDataFrameは以下のフィールドを持つ
        name | re_type_num | rename | vecs
        """
        # tqdm.pandas()
        # self.trainDF[['rename', 'vecs']] = self.trainDF[readColumnName].progress_apply(self.a_convert)
        # input(self.trainDF['vecs'].shape)

        self.trainDF = df
        self.renames = []
        self.vecs = []
        for word in tqdm(self.trainDF[readColumnName].values, total=len(self.trainDF[readColumnName].index)):
            result = self.a_convert(word)
            self.renames.append(result[0])
            self.vecs.append(result[1])

        return self.renames, np.array(self.vecs)

    def a_convert(self, word: str):
        """
        建物名称を受け、200次元のベクトルに変換する
        """
        if word == None:
            return '', np.zeros((40, 200))

        # 形態素解析（分割の全ての候補を持つ）
        allWords = self.split(word)

        # どのように分割しても正規化すると名称が消失する場合は元の名前をそのまま正規化せずに使用する
        # 末尾に分割ナシ名称を入れておくことでforを抜けたときにwordVecには元の名前のベクトルが入る
        # SplitBuildingNameの引数には仮想の結果を代入
        allWords.append(SplitBuildingName([f'{word}\t名詞,固有名詞,*,*,*,*']))
        for buildingName in allWords:
            isReturn, nameStr, wordVec = self.getVec(buildingName)
            if isReturn:
                return nameStr, wordVec

        # 分割方法を変えてもベクトルが算出できなかった場合は、元の名前でベクトルを算出する
        return nameStr, wordVec

    def split(self, name: str):
        # 形態素解析結果の出力
        m:str = self.tagger.parseNBest(10, name)
        
        # 解析結果を分割
        a = m.split('\nEOS\n')
        a = list(map(lambda x: x.split('\n'), a))
        a.remove(a[-1])

        convertList = list(map(SplitBuildingName, a))

        # 分割が同じものをはじく
        return list(dict.fromkeys(convertList))

    def getVec(self, name: SplitBuildingName, wordDim=40):
        """
        分割済みの名前データを受けてこれをベクトルに変換する\n
        各建物名称は最大(wordDim)個の単語に分割され、これに満たない単語数の名称については、(wordDim)個の200次元ベクトルになるよう0でパディングする

        戻り値は　(結果が有効か, 名称, ベクトル)\n
        変換に失敗した場合は名称が空文字列となる
        """
        def check(word:str) -> np.ndarray:
            try:
                return self.model[word]
            except KeyError:
                return np.array(self.zeroVec)
                
        # 正規化
        words = self.regularization(name)
        # 正規化した結果何もなくなっていた場合は分割方法を変える
        if len(words) == 0:
            return False, '', []
        
        vecs = list(map(check, words))
        if not len(vecs) <= wordDim:
            print(f'Split count is over the word dimention (name: {"".join(name.allNames)}, count:{len(vecs)}, dim:{wordDim})')
            vecs = vecs[:40]
        
        wordVec = list(sum(vecs))
        # 生成したベクトルが０でない場合は有効な結果
        isValid = wordVec != self.zeroVec

        if self.outputOneVec:
            return isValid, ''.join(words), self.normalize(wordVec)
        else:
            return isValid, ''.join(words), np.stack(vecs + [self.zeroVec] * (wordDim-len(vecs)))

    def regularization(self, buildingName: SplitBuildingName) -> list[str]:
        """
        分割済みの建物名称を受けてそれを正規化する
        """
        allInfo = buildingName.allNameAndInfo
        words = []

        if self.isRegulation:
            for wordInfo in allInfo:
                # いらない文字は登録しない
                if not (re.match('BOS/EOS|記号', wordInfo.info[0]) or re.match('地域|人名', wordInfo.info[2])):
                    words.append(wordInfo.name)
        else:
            words = list(map(lambda x: x.name, allInfo))

        return words

    def normalize(self, vec:list) -> list:
        if vec == self.zeroVec:
            return self.zeroVec
        # 足し合わせた個数などによってベクトルの長さが各施設名称で異なってしまうため、すべて単位ベクトルにそろえている
        return vec / np.linalg.norm(vec)

    def readFile(self, path:str) -> tuple[list, np.ndarray]:
        """
        ベクトル算出済みのファイルを読み込む
        （実装未完了）
        """
        raise NotImplementedError
        trainDF = pd.read_csv(path)
        return trainDF


    def writeResult(self, path:str):
        """
        正規化済みのデータを出力する
        """
        # 参照渡しの回避
        _trainDF = copy.deepcopy(self.trainDF)
        # def _tmp(x:np.ndarray):
        #     try:
        #         return x.tolist()[0]
        #     except:
        #         # [] は正規化で名称をすべて削除してしまっている
        #         # 空欄 はエンティティベクトルに該当する単語が存在していない
        #         return '[]'

        # _trainDF['vecs'] = _trainDF['vecs'].apply(lambda x: _tmp(x))

        _trainDF = _trainDF.merge(pd.DataFrame(self.renames, columns=['renames']), left_index=True, right_index=True)
        _trainDF.to_csv(path, encoding='shift-jis', index=False)
        print(f'Saved ready data at {path}')

