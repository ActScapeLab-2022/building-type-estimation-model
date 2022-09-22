# 引数を渡したら名称を読み取って勝手に予測結果を返してくれる
# name列に予測する建物名称が入っていることを制約として付し，読み取り可能なファイルとして.dbf, .csvはサポートする
# 予測する列名称は変数として格納することで変更可能なものとする


import json
import sys
from argparse import ArgumentTypeError
from enum import Enum, unique
from logging import warning
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
from tensorflow.python.keras import models

from models.model import TextCNN
from tools.WordConverter import WordConverter



####################################  必要なグローバル変数の宣言  ####################################


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
BUILDINGNAMECOLUMN:str = config['buildingNameColumn']
PREDICTEDNAMECOLUMN:str = config['predictedNameColumn']
ENCODING:str      = config['encoding']
ISNORMALIZE:bool  = config['isNormalize']
REPREDICATE:bool  = config['rePredicate']
NUMBEROUTPUT:bool = config['numberOutput']
MODELNAME:str     = config['modelName']

# outputファイル名の先頭に付加する文字
OUTPUTFILE_ADDITIONALNAME = 'pred_'

# model
MODEL:TextCNN = models.load_model(str(Path(__file__).parent/'models'/MODELNAME))



####################################  これより実装  ####################################


# 読み込みファイルの確認
def checkDbf(path:Path) -> gpd.GeoDataFrame:
    for suffix in SHPSUFFIX:
        if not (path.parent/f'{path.stem}.{suffix}').exists():
            raise FileNotFoundError(f'{path.stem}.{suffix} is not exists in predict/input')
    return checkCsv(path)

def checkCsv(path:Path) -> gpd.GeoDataFrame:
    print(f'Reading {path.name}...')
    data:gpd.GeoDataFrame = gpd.read_file(path, encoding=ENCODING)
    print(f'Finished reading {path.name}')
    if not data.columns.__contains__(BUILDINGNAMECOLUMN):
        raise IndexError(f'{path.name} is not existed the "{BUILDINGNAMECOLUMN}" column')
    # data.rename(columns={BUILDINGNAMECOLUMN : 'name'}, inplace=True)
    return data

# 推定と保存
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
        print(f'###  Predication of {path.name} is started  ###')
        print('Convert names to vectors')
        _, train_vecs = converter.convert(data, BUILDINGNAMECOLUMN)
        result = MODEL.predict(train_vecs)
        if not NUMBEROUTPUT:
            result = list(map(lambda x: PREDICTNAME[np.argmax(x)], result))
        data[PREDICTEDNAMECOLUMN] = result
        outPath = Path(__file__).parent/'predict'/f'{OUTPUTFILE_ADDITIONALNAME}{path.stem}.{type.name.lower()}'
        outPath.unlink(True)
        if type == DataType.SHP:
            data.to_file(outPath, index=False, encoding=ENCODING)
        else:
            data.to_csv(outPath, index=False, encoding=ENCODING)
        print()
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
            if (Path(__file__).parent/'predict'/f'{OUTPUTFILE_ADDITIONALNAME}{path.name}').exists() and not REPREDICATE:
                warning(f'{path.name} is already predicated')
                continue
            # データの読み込み
            if path.suffix == '.shp':
                predicate(path, checkDbf(path), DataType.SHP)
            elif path.suffix == '.dbf' or path.suffix == '.shx':
                continue
            elif path.suffix == '.csv':
                predicate(path, checkCsv(path), DataType.CSV)
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