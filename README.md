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

# 第一階段自監督微調視覺編碼器

1. 先下載 https://huggingface.co/datasets/Kuoskyler/swiftplan-isaac-sim 這個資料集，要注意格式和路徑問題

2. 下載 https://huggingface.co/google/siglip2-base-patch16-512

3. 改 byol.py 中的 IMAGE_DIR，要指向一個資料夾內有任務的所有影像，不須標註，執行後模型就會自動訓練，訓練完會給一個資料夾，裡面有模型權重

# 第二階段監督式對比學習

1. 把 embedding_text_image.py 中的 model-name 的 default="google/siglip2-base-patch16-512" 路徑改成第一階段模型權重，並執行

2. 把 swiftplan_train.py MODEL_NAME 也改成和上一步一樣，並執行，會訓練模型並對模型在測試資料上測試

3. 如果發現 best val 都在前段(小於 15 epoch)，可以調一下第二階段的訓練 batch size
