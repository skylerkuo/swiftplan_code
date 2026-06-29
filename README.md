# 多模態任務規劃器

這是一個多模態任務規劃器

詳細技術細節請看swiftplan論文

## 資料說明

測試資料為:data_gradu_test.jsonl  
訓練資料為:data_gradu.jsonl

因為資料需要下載 所以還需要先到網路上載才行，用這兩個跑沒有用，只是格式示意

但其實沒有也沒有差，反正計畫也不會用我這個資料集

## 程式說明

文字和影像embedding: embedding_text_image.py  
最終任務規劃模型訓練: swiftplan_train.py

## 假如想執行看看

1. 先下載https://huggingface.co/datasets/Kuoskyler/swiftplan-isaac-sim這個資料集，要注意格式和路徑問題
2. 下載https://huggingface.co/google/siglip2-base-patch16-512
3. 執行 embedding_text_image.py
4. 執行 swiftplan_train.py，會訓練模型並對模型在測試資料上測試
