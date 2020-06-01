# Urban-Delivery-Drone (2020-5-1)

## 緒言

現在，ECサイトの発達の急速により,配達量が増加している.そのため,あらゆる配達業者への負担が増大している.  

アメリカなどの海外では，Amazonを筆頭とした企業によるドローンによる配達のための実験がなされている.  

また，日本でも日本郵政による奥多摩の山間部での郵便配達の実験や，  

NTT docomoなどによる大分県の離島での配達実験が行なわれている.  

そうした配達方法の手段をとれる地域は，土地が広くドローンの離着陸するための土地が十分にある.  

今後の日本では，コンパクトシティー化が進行すると言われており，都市部に人口が集中すると予想される.  

そのため，都市部に特化した配達ドローンの提案を行なう.  

## 手順
1.　配達受取人から事前情報をもらい，どのような形態の着陸方法にするか判断する    
* 事前情報：ドローンの着陸地点のマーカーを含めたドローンの着陸地点の画像   

2.　配達受取人に受取位置の目印マーカー(色付きの長方形マーカー)をドローンに見える位置に貼ってもらう  

3.　マーカーをドローンのカメラによって識別させ，配達受取人のベランダ、受取場所に近づく  

コード

droneHSV.py   HSVを用いてマーカーの色を抽出し、マーカーの面積から壁とソローンの距離を計測し、制御に活かす。

dronegray.py　RGBを用いて上記と同じ手法をしている。


4.　荷物の受け渡しをする

方法

* 選択物干し竿に荷物を引っ掛ける

* 窓や壁に設置した受取台の上に荷物を置く  

### オプション
1. 着陸地点の最適な場所の提案のために、1枚の写真から3D生成モデルを作成する

2. 誤認予防のために，集合住宅の外観から見て窓を識別し，その画像を受取人のスマホに送信し，自分の住居の窓をタップしてもらう
