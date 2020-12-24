import socket # UDP通信のため
from time import sleep # sleepを使うため
import curses # cursesモジュールを使って，画面表示をちょっとオシャレにする
import re

INTERVAL = 0.2 # スリープ時間の設定．0.2秒

# ここからプログラムスタート
if __name__ == "__main__": # "python tello_state.py"として実行されたかどうか判定している．
    # ステータス受信用のUDPサーバの設定
    local_ip = '' # '0.0.0.0'と同じ意味．すなわち「全てのネットワークインターフェイスを使う」
    local_port = 8890 # ステータス受信は8890ポート
    socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # ソケットを作成
    socket.bind((local_ip, local_port)) # サーバー側はバインドが必要

    # コマンド送信用の設定
    tello_ip = '192.168.10.1' # TelloのIPアドレス
    tello_port = 8889 # コマンドは8889ポートへ送る
    tello_adderss = (tello_ip, tello_port) # アドレスを作成

    # 最初に"command"を送ってSDKモードを開始しないとステータスが出てこない
    socket.sendto('command'.encode('utf-8'), tello_adderss)

    # Ctrl + cが押されるまで繰り返す
    try:
        index = 0 # indexにループした回数が入る
        while True: # 永久ループ
            index += 1
            response, ip = socket.recvfrom(1024) # 受信は最大1024バイトまで．受信結果はresponse変数に入る
            # 受信データに手を加える
            response = response.decode("utf-8").split(";")
            if len(response) > 10:
                RESPONSE = response
                tof = response[8] 
            #response2 = int.from_bytes(response, "little")
             # セミコロンの部分に改行コードを挿入
            #out = response.replace(';', ';\n') # セミコロンの部分に改行コードを挿入
            #out = 'Tello State:\n' + out # 冒頭部分にちょっと装飾
            #responce = re.search('tof:',response)
                tof = re.sub(r"tof:", "", tof)   
                tof = int(tof)
                print(tof)
            #print(out[14]) # 上の方に定義されているreport関数を使って画面出力する
            #sleep(INTERVAL) # 一定時間待つ
    except KeyboardInterrupt:
        # ctrl + cが押されたあとの終了処理
        curses.echo()
        curses.nocbreak()
        curses.endwin()