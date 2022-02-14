# PuncMultiDataBert
**train.py** 可以用以訓練model,並且設定和調整各種參數<br>
Bert model的結構在**model.py**中,要先有下載的檔案才能使用。<br>
**dataset.py**用來制定輸入model的資料格式<br>
**add_punc.py**可以為文本標注標點符號,用來示範model的運作<br>
**test.py**用來分析和測試model<br>
##簡介：
本次的model是使用pytorch架設，主要的目標是藉wiki Chinese corpus訓練，使其具有能夠預測標點的能力<br>
我們所進行的嘗試是在bert based model底層加上一層簡單的神經網路,並且再進行fine-tuning,使其能應用<br>
到標點標注的工作,主要標注的目標有"，"、"。"、"？"三種，我們藉由分析三種標點和不標注四種情況的precision,
recall和 F-score 來衡量model的表現<br>
