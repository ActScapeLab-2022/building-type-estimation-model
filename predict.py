# 引数を渡したら名称を読み取って勝手に予測結果を返してくれる
# name列に予測する建物名称が入っていることを制約として付し，読み取り可能なファイルとして.dbf, .csvはサポートする
# 予測する列名称は変数として格納することで変更可能なものとする


import json
import sys
from argparse import ArgumentError, ArgumentTypeError
from enum import Enum, unique
from logging import warning
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
from tensorflow.python.keras import models

from models.model import TextCNN
from tools.WordConverter import WordConverter


# 対応するデータタイプ
@unique
class DataType(Enum):
    SHP = 0
    CSV = 1
    TXT = 2
SHPSUFFIX = ['shp', 'dbf', 'shx']

# 予測結果
PREDICTNAME = ['商業', '事業所', '公共', '宗教', '駐車場']


# configファイルの読み込み
with open(str(Path(__file__).parent/'predict_config.json')) as f:
    config = json.loads(f.read())
ISNORMALIZE:bool  = config['isNormalize']
REPREDICATE:bool  = config['rePredicate']
NUMBEROUTPUT:bool = config['numberOutput']
MODELNAME:str     = config['modelName']

# outputファイル名の先頭に付加する文字
OUTPUTFILE_ADDITIONALNAME = 'predicated_'

# model
MODEL:TextCNN = models.load_model(str(Path(__file__).parent/'models'/MODELNAME))

# 読み込みファイルの確認
def checkDbf(path:Path) -> gpd.GeoDataFrame:
    for suffix in SHPSUFFIX:
        if not (path.parent/f'{path.stem}.{suffix}').exists():
            raise FileNotFoundError(f'{path.stem}.{suffix} is not exists in predict/input')
    checkCsv(path)

def checkCsv(path:Path) -> gpd.GeoDataFrame:
    data = gpd.read_file(path)
    if not data.columns.__contains__('name'):
        raise IndexError(f'{path.name} is not existed the "name" column')
    return data

def predicate(path:Path, data:Union[str, gpd.GeoDataFrame], type:DataType):
    converter = WordConverter(ISNORMALIZE, False)
    if type == DataType.TXT:
        _, wordVec = converter.a_convert(data)
        result = MODEL.predict(np.array([wordVec]))
        if not NUMBEROUTPUT:
            result = PREDICTNAME[np.argmax(result[0])]
        print()
        print('###  R E S U L T  ###')
        print()
        print(f'{data} : {result}')
    else:
        _, train_vecs = converter.convert(data)
        result = MODEL.predict(train_vecs)
        if not NUMBEROUTPUT:
            result = list(map(lambda x: PREDICTNAME[x], result))
        data['buildingType'] = result
        data.to_file(str(Path(__file__).parent/'predict'/f'{path.stem}.shp'), index=False, encording='shift-jis')
        print('###  Finished creating the predicted data  ###')


def main():
    """
    コマンドライン引数の確認を行い，予測タスクを実行する
    """
    args = sys.argv
    if len(args) == 1:
        for path in (Path(__file__).parent/'predict'/'input').iterdir():
            # フォルダは無視
            if path.is_dir():
                continue
            # 既に存在する予測結果のファイルを除外するか
            if (Path(__file__).parent/'predict'/f'{OUTPUTFILE_ADDITIONALNAME}{path.name}').exists() and REPREDICATE:
                warning(f'{path.name} is already predicated')
                continue
            # データの読み込み
            if path.suffix == '.shp':
                predicate(checkDbf(str(path)), DataType.SHP)
            elif path.suffix == '.dbf' or path.suffix == '.shx':
                continue
            elif path.suffix == '.csv':
                predicate(checkCsv(str(path)), DataType.CSV)
            else:
                warning(f'{path.name} is not supported in this project')
                continue

    elif len(args) == 2:
        predicate('', args[1], DataType.TXT)

    elif len(args) > 2:
        raise ArgumentTypeError(
            'Too many arguments was passed.' +
            'Predict is able to get <No argument> or <Building Name (1 arg)>'
            )



main()



# # isNormalize = False

# # RandomForest
# dataType = 'ManyData'
# sourcePath = Path(__file__).parents[1]/'Source'
# # for isNormalize in [False]:
# #     for name in ['Shizuoka']:
# #         cityData = sourcePath/dataType/name/f'{name}.csv'
# #         vecData = VecProcessor(cityData, isNormalize, True)
# #         # learner = RFLearner(vecData, vecData)
# #         # learner.saveAll(dataType, name)
# #         reader = RFReader(str(Path(__file__).parent/'RandomForest'/'result'/dataType/name/f'{name}{"_Normalize" if vecData.isNormalize else ""}.sav'), vecData)
# #         reader.saveTree(dataType, name)
# #         print(f'Finished learning : name- {name}, isNormalize- {isNormalize}')


# # TextCNN    
# # Uni
# # name = 'Zenkoku'
# name = 'Shizuoka'
# # --- Predicate ---
# # predicateData = VecProcessor(sourcePath/'cities'/'Shizuoka'/'Shizuoka.csv', isNormalize, False, splitRatio=1)
# predicateData = VecProcessor(sourcePath/'ManyData'/'Zenkoku'/'Hukuoka'/'Hukuoka.csv', isNormalize, False, splitRatio=1)
# reader = TCNNReader(dataType, name, isNormalize, predicateData)
# # reader.saveGraph(str(Path(__file__).parent/'shizuoka_train.svg'), 'Result of learning in Shizuoka')
# # predicted =  reader.savePredict(Path(__file__).parent/'Huji_pred.csv')
# predicted = reader.model.predict(reader.testData.test_vecs)
# # print(reader.testData.testDF)
# print(f'accuracy- {accuracy_score(reader.testData.testDF["re_type_num"].values, predicted.argmax(axis=1) + 1) * 100:.1f} %')
# # --- Learning ---
# # cityData = sourcePath/dataType/name/f'{name}.csv'
# # trainData = VecProcessor(cityData, isNormalize, False, splitRatio=0.2)
# # learner = TCNNLearner(trainData, trainData)
# # learner.saveAll(dataType)
# # print(f'Finished learning : name- {name}, isNormalize- {isNormalize}')

# # Multi
# # for name in (sourcePath/dataType).iterdir():
# #     if not name.is_dir():
# #         continue

# #     # 除外する項目を指定
# #     if ['Random', 'Zenkoku'].__contains__(name.stem):
# #         continue

# #     cityData = name/f'{name.stem}.csv'
# #     trainData = VecProcessor(cityData, isNormalize, False, splitRatio=0.2)
# #     learner = TCNNLearner(trainData, trainData)
# #     learner.saveAll(dataType)
# #     print(f'Finished learning : name- {name.stem}, isNormalize- {isNormalize}')
    
# #     # Release Memory
# #     learner.dispose()
# #     del trainData
# #     del learner
