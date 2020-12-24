import socket           # UDP通信用
import threading        # マルチスレッド用
import time             # ウェイト時間用
import numpy as np      # 画像データの配列用
#import libh264decoder   # H.264のデコード用(自分でビルドしたlibh264decoder.so)

class Tello:
    """Telloドローンと通信するラッパークラス"""

    def __init__(self, local_ip, local_port, imperial=False, command_timeout=.3, tello_ip='192.168.10.1', tello_port=8889):
        """
        クラスの初期化．ローカルのIP/ポートをバインドし，Telloをコマンドモードにする．

        :param local_ip (str): バインドする(UDPサーバにする)ローカルのIPアドレス
        :param local_port (int): バインドするローカルのポート番号
        :param imperial (bool): Trueの場合，速度の単位はマイル/時，距離の単位はフィート．
                                Falseの場合, 速度の単位はkm/h，距離はメートル．デフォルトはFalse
        :param command_timeout (int|float): コマンドの応答を待つ時間．デフォルトは0.3秒．
        :param tello_ip (str): TelloのIPアドレス．EDUでなければ192.168.10.1
        :param tello_port (int): Telloのポート.普通は8889
        """

        self.abort_flag = False     # 中断フラグ
        #self.decoder = libh264decoder.H264Decoder() # H.264のデコード関数を登録
        self.command_timeout = command_timeout      # タイムアウトまでの時間
        self.imperial = imperial    # 速度と距離の単位を選択
        self.response = None    # Telloが応答したデータが入る
        self.frame = None       # BGR並びのnumpy配列 -- カメラの出力した現在の画像
        self.is_freeze = False  # カメラ出力を一時停止(フリーズ)するかどうかのフラグ
        self.last_frame = None  # 一時停止時に出力する画像
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)        # コマンド送受信のソケット
        #self.socket_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # ビデオストリーム受信用のソケット
        self.tello_address = (tello_ip, tello_port)     # IPアドレスとポート番号のタプル(変更不可能)
        self.local_video_port = 11111                   # ビデオ受信のポート番号
        self.last_height = 0                            # get_heightで確認した最終の高度
        self.socket.bind((local_ip, local_port))        # コマンド受信のUDPサーバのスタート(バインド)

        # コマンドに対する応答の受信スレッド
        self.receive_thread = threading.Thread(target=self._receive_thread)     # スレッドの作成
        self.receive_thread.daemon = True   # メインプロセスの終了と一緒にスレッドが死ぬように設定

        self.receive_thread.start()         # スレッドスタート

        # ビデオ受信の開始 -- コマンド送信: command, streamon
        self.socket.sendto(b'command', self.tello_address)          # 'command'を送信し，TelloをSDKモードに
        print ('sent: command')
        self.socket.sendto(b'streamon', self.tello_address)         # 'streamon'を送信し，ビデオのストリーミングを開始
        print ('sent: streamon')

        #self.socket_video.bind((local_ip, self.local_video_port))   # ビデオ受信のUDPサーバのスタート(バインド)

        # ビデオ受信のスレッド
        #self.receive_video_thread = threading.Thread(target=self._receive_video_thread)     # スレッドの作成
        #self.receive_video_thread.daemon = True # メインプロセスの終了と一緒にスレッドが死ぬように設定

        #self.receive_video_thread.start()       # スレッドスタート



    def _receive_thread(self):
        """
        Telloからの応答を監視する

        スレッドとして走らせる．Telloが最後に返した応答をself.responseに格納する

        """
        while True:
            try:
                self.response, ip = self.socket.recvfrom(3000)      # Telloからの応答を受信（最大3000バイトまで一度に受け取れる）
                #print(self.response)
            except socket.error as exc:     # エラー時の処理
                print ("Caught exception socket.error : %s" % exc)


    def send_command(self, command):
        """
        Telloへコマンドを送信し，応答を待つ

        :param command: 送信するコマンド
        :return (str): Telloの応答

        """

        print (">> send cmd: {}".format(command))
        self.abort_flag = False     # 中断フラグを倒す
        timer = threading.Timer(self.command_timeout, self.set_abort_flag)      # タイムアウト時間が立ったらフラグを立てるタイマースレッドを作成

        self.socket.sendto(command.encode('utf-8'), self.tello_address)     # コマンドを送信

        timer.start()   # スレッドスタート
        while self.response is None:        # タイムアウト前に応答が来たらwhile終了
            if self.abort_flag is True:     # タイムアウト時刻になったらブレイク
                break
        timer.cancel()  # スレッド中断

        if self.response is None:       # 応答データが無い時
            response = 'none_response'
        else:                           # 応答データがあるとき
            response = self.response.decode('utf-8')

        self.response = None    # _receive_threadスレッドが次の応答を入れてくれるので，ここでは空にしておく

        return response     # 今回の応答データを返す

    def set_abort_flag(self):
        """
        self.abort_flagのフラグをTrueにする

        send_command関数の中のタイマーで呼ばれる．

        この関数が呼ばれるということは，応答が来なくてタイムアウトした，ということ．

        """

        self.abort_flag = True

    def takeoff(self):
        """
        離陸開始

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """

        return self.send_command('takeoff')

    def set_speed(self, speed):
        """
        スピードを設定

        この関数の引数にはkm/hかマイル/hを使う．
        Tello APIは 1〜100 センチメートル/秒を使う

        Metric: .1 to 3.6 km/h
        Imperial: .1 to 2.2 Mile/h

        Args:
            speed (int|float): スピード

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """

        speed = float(speed)

        if self.imperial is True:       # 単位系に応じて計算
            speed = int(round(speed * 44.704))      # Mile/h -> cm/s
        else:
            speed = int(round(speed * 27.7778))     # km/h -> cm/s

        return self.send_command('speed %s' % speed)

    def rotate_cw(self, degrees):
        """
        時計回りの旋回

        Args:
            degrees (int): 旋回角度， 1〜360度

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """

        return self.send_command('cw %s' % degrees)

    def rotate_ccw(self, degrees):
        """
        反時計回りの旋回

        Args:
            degrees (int): 旋回角度， 1〜360度.

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """
        return self.send_command('ccw %s' % degrees)

    def flip(self, direction):
        """
        宙返り

        Args:
            direction (str): 宙返りする方向の文字， 'l', 'r', 'f', 'b'.

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.
        """

        return self.send_command('flip %s' % direction)

    def get_response(self):
        """
        Telloの応答を返す

        Returns:
            int: Telloの応答

        """
        response = self.response
        return response

    def get_height(self):
        """
        Telloの高度(dm)を返す

        Returns:
            int: Telloの高度(dm)

        """
        height = self.send_command('height?')
        height = str(height)
        height = filter(str.isdigit, height)
        try:
            height = int(height)
            #self.last_height = height
        except:
            height = self.last_height
            pass
        return height

    def get_tof(self):
            """
        Telloの高度(dm)を返す

        Returns:
            int: Telloの高度(dm)

        """
        tof = self.send_command('tof?')
        tof = str(tof)
        tof = filter(str.isdigit, tof)
        try:
            tof = int(tof)
            self.last_height = tof
        except:
            tof = self.last_height
            pass
        return tof

    def get_battery(self):
        """
        バッテリー残量をパーセンテージで返す

        Returns:
            int: バッテリー残量のパーセンテージ

        """

        battery = self.send_command('battery?')

        try:
            battery = int(battery)
        except:
            pass

        return battery

    def get_flight_time(self):
        """
        飛行時間を秒数で返す

        Returns:
            int: 飛行の経過時間

        """

        flight_time = self.send_command('time?')

        try:
            flight_time = int(flight_time)
        except:
            pass

        return flight_time

    def get_speed(self):
        """
        現在のスピードを返す

        Returns:
            int: 現在スピード， km/h または Mile/h

        """

        speed = self.send_command('speed?')

        try:
            speed = float(speed)

            if self.imperial is True:
                speed = round((speed / 44.704), 1)      # cm/s -> mile/h
            else:
                speed = round((speed / 27.7778), 1)     # cm/s -> km/h
        except:
            pass

        return speed

    def land(self):
        """
        着陸を開始

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """

        return self.send_command('land')

    def move(self, direction, distance):
        """
        direction の方向へ distance の距離だけ移動する．

        この引数にはメートルまたはフィートを使う．
        Tello API は 20〜500センチメートルを使う．

        Metric: .02 〜 5 メートル
        Imperial: .7 〜 16.4 フィート

        Args:
            direction (str): 移動する方向の文字列，'forward', 'back', 'right' or 'left'．
            distance (int|float): 移動する距離．(メートルまたはフィート)

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """

        distance = float(distance)

        if self.imperial is True:
            distance = int(round(distance * 30.48))     # feet -> cm
        else:
            distance = int(round(distance * 100))       # m -> cm

        return self.send_command('%s %s' % (direction, distance))

    def move_backward(self, distance):
        """
        distance の距離だけ後進する．

        Tello.move()のコメントを見ること．

        Args:
            distance (int): 移動する距離

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """

        return self.move('back', distance)

    def move_down(self, distance):
        """
        distance の距離だけ降下する．

        Tello.move()のコメントを見ること．

        Args:
            distance (int): 移動する距離

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """

        return self.move('down', distance)

    def move_forward(self, distance):
        """
        distance の距離だけ前進する．

        Tello.move()のコメントを見ること．

        Args:
            distance (int): 移動する距離

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """
        return self.move('forward', distance)

    def move_left(self, distance):
        """
        distance の距離だけ左移動する．

        Tello.move()のコメントを見ること．

        Args:
            distance (int): 移動する距離

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """
        return self.move('left', distance)

    def move_right(self, distance):
        """
        distance の距離だけ右移動する．

        Tello.move()のコメントを見ること．

        Args:
            distance (int): 移動する距離

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """
        return self.move('right', distance)

    def move_up(self, distance):
        """
        distance の距離だけ上昇する．

        Tello.move()のコメントを見ること．

        Args:
            distance (int): 移動する距離

        Returns:
            str: Telloからの応答．'OK'または'FALSE'.

        """

        return self.move('up', distance)