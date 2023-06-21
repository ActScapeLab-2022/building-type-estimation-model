import geopandas as gpd
import pandas as pd

import subprocess

import os
import csv
import shutil

import warnings
warnings.filterwarnings("ignore")

""" 店舗率算出プログラム

    必要なデータ
    ・対象地域の建物データ 
    ・building-type-prediction-modelのstreet.pyを実行することで得られる建物とリンクが結びついたデータ

"""

############### 書き換え部分　############################################################

tatemono = pd.read_csv(r"Source\tatemono\shinagawaku\shinagawaku.csv")
addbuildings = pd.read_csv(r"Source\link\gotanda\links_addbuildings\links_addbuildings.csv")

# ファイル名になる
tiiki = '五反田'

##########################################################################################


class Storerate:
    def __init__(self, tatemono, addbuildings) -> None:
        self.atatemono = tatemono
        self.addbuildings = addbuildings

    def tocsv(self, outputpath, csvname):
        file_path = f"{outputpath}\{tiiki}.csv"
        with open(file_path, "w", newline="", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csvname.columns)
            writer.writerows(csvname.values)
        # print("CSVファイル保存完了")
        # print()

    def linkwith_tatemono(self):
        """inputに放り込むデータを作成

        Returns:
            pd.DataFrame:        tatemon_id          name
                        48544       48545  ミリオンコート目黒不動前
                        48545       48546            小林
                        48546       48547            松岡
                        48547       48548       ドゥミル五反田
                        48548       48549      ＺＯＯＭ西五反田
                        ...           ...           ...
                        80288       80289            福田
                        80290       80291        佐々木建二郎
                        80292       80293            延藤
                        80293       80294          徳江治寿
                        80320       80321           NaN
        """
        buildingslist = []
        for i in range(len(self.addbuildings)):
            inputstr = self.addbuildings['buildings'][i]
            if type(inputstr) == float:
                input_list = [0]
            else:
                input_list = eval(inputstr)
            buildingslist += input_list
        unique_list = list(set(buildingslist))
        extracted_rows = tatemono[tatemono['tatemon_id'].isin(unique_list)]
        extracted_rows.rename(columns={'housename': 'name'}, inplace=True)
        selected_columns = extracted_rows[['tatemon_id', 'name']]
        self.selected_columns = selected_columns
        self.tocsv('predict\input',selected_columns)
        
        return selected_columns
    
    def estimate_tatemono(self):
        """inputに保存されたデータを用いて建物種別の推定（藤條のコード）
           predictファイル内に推定列(preType)追加済のデータフレームが保存される 

        Returns:
            pd.DataFrame: tatemon_id,name,preType
                    48545,ミリオンコート目黒不動前,商業
                    48546,小林,事業所
                    48547,松岡,商業
                    48548,ドゥミル五反田,商業
                    48549,ＺＯＯＭ西五反田,商業
                    48550,（有）戸越興産,事業所
        """
        subprocess.run(['python', 'predict.py'])
        pred = pd.read_csv(f'predict\input\{tiiki}.csv')
        self.pred = pred
        
        # 推定が終わったらinputフォルダを空にする
        # folder_path = r'predict\input'
        # for filename in os.listdir(folder_path):
        #     file_path = os.path.join(folder_path, filename)
        #     if os.path.isfile(file_path):
        #         os.remove(file_path)
        #     elif os.path.isdir(file_path):
        #         shutil.rmtree(file_path)
                
        return pred
        

    def storerate(self):
        """推定結果を用いて店舗率を算出
           店舗率列が追加されたリンクデータが返ってくる 
        Returns:
                                                                WKT  ...  storerate
            0     LINESTRING (139.714805 35.620441,139.7147291 3...  ...        0.0
            1     LINESTRING (139.714805 35.620441,139.7147291 3...  ...        0.0
            2     LINESTRING (139.7149298 35.6233372,139.7146307...  ...  71.428571
            3     LINESTRING (139.7149298 35.6233372,139.7146307...  ...  71.428571
            4     LINESTRING (139.7146307 35.623565,139.7149344 ...  ...      100.0
            ...                                                 ...  ...        ...
            2235  LINESTRING (139.731672 35.6328479,139.7317343 ...  ...       75.0
            2236  LINESTRING (139.7317923 35.6211894,139.7319023...  ...       50.0
            2237  LINESTRING (139.7317923 35.6211894,139.7319023...  ...       50.0
            2238  LINESTRING (139.731672 35.6328479,139.7319945 ...  ...  83.333333
            2239  LINESTRING (139.731672 35.6328479,139.7319945 ...  ...  83.333333
        """
        pred = pd.read_csv(f'predict\pred_{tiiki}.csv')
        self.addbuildings['storerate'] = ''
        for i in range(len(self.addbuildings)):
            inputstr = self.addbuildings['buildings'][i]
            if type(inputstr) == float:
                self.addbuildings['storerate'][i] = 0
            else:
                input_list = eval(inputstr)
                extracted_rows = pred[pred['tatemon_id'].isin(input_list)][['tatemon_id', 'preType']]
                commercial_count = extracted_rows[extracted_rows['preType'] == '商業'].shape[0]
                total_count = extracted_rows.shape[0]
                commercial_percentage = commercial_count / total_count * 100
                self.addbuildings['storerate'][i] = commercial_percentage
        self.tocsv('output',self.addbuildings)

        return self.addbuildings



storerate_instance = Storerate(tatemono, addbuildings)
print(storerate_instance.linkwith_tatemono())
print(storerate_instance.estimate_tatemono())
print(storerate_instance.storerate())

        
