# https://midoriit.com/2018/05/python%E3%81%AB%E3%82%88%E3%82%8B%E3%83%89%E3%83%AD%E3%83%BC%E3%83%B3%E3%80%8Ctello%E3%80%8D%E3%81%AE%E5%88%B6%E5%BE%A1.html
# https://qiita.com/hsgucci/items/d1df1122853ee6dd4bf7
# https://flxy.jp/article/2963
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import threading 
import socket
import time
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class TelloController1(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.initConnection()
        self.initUI()

        # 最初にcommandコマンドを送信
        try:
            sent = self.sock.sendto('command'.encode(encoding="utf-8"), self.tello)
        except:
            pass
        # 速度を遅めに設定
        try:
            sent = self.sock.sendto('speed 50'.encode(encoding="utf-8"), self.tello)
        except:
            pass

        #問い合わせスレッド起動
        askThread = threading.Thread(target=self.askTello)
        askThread.setDaemon(True)
        askThread.start()

    # 通信の設定
    def initConnection(self):
        host = ''
        port = 9000
        #port = 8890
        locaddr = (host,port) 
        self.tello = ('192.168.10.1', 8889)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(locaddr)

        # 受信スレッド起動
        recvThread = threading.Thread(target=self.recvSocket)
        recvThread.setDaemon(True)
        recvThread.start()

    # UIの作成
    def initUI(self):
        # 情報表示用ラベル
        self.label = QLabel('')
        self.label.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.batteryLabel = QLabel('100%')
        self.batteryLabel.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.batteryLabel.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.timeLabel = QLabel('0s')
        self.timeLabel.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.timeLabel.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.tofLabel = QLabel('0m')
        self.tofLabel.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.tofLabel.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        #self.heightLabel = QLabel('0m')
        #self.heightLabel.setFrameStyle(QFrame.Box | QFrame.Plain)
        #self.heightLabel.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.attitudeLabel = QLabel('0')
        self.attitudeLabel.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.attitudeLabel.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.accelerationLabel = QLabel('0')
        self.accelerationLabel.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.accelerationLabel.setAlignment(Qt.AlignBottom | Qt.AlignRight)

        # 終了ボタン
        endBtn = QPushButton("End")
        endBtn.clicked.connect(self.endBtnClicked)

        # 離着陸ボタン
        takeoffBtn = QPushButton("Takeoff")
        takeoffBtn.clicked.connect(self.takeoffBtnClicked)
        landBtn = QPushButton("Land")
        landBtn.clicked.connect(self.landBtnClicked)

        # 上昇下降回転ボタン
        upBtn = QPushButton("Up")
        upBtn.clicked.connect(self.upBtnClicked)
        downBtn = QPushButton("Down")
        downBtn.clicked.connect(self.downBtnClicked)
        cwBtn = QPushButton("CW")
        cwBtn.clicked.connect(self.cwBtnClicked)
        ccwBtn = QPushButton("CCW")
        ccwBtn.clicked.connect(self.ccwBtnClicked)

        # 前後左右ボタン
        forwardBtn = QPushButton("Forward")
        forwardBtn.clicked.connect(self.forwardBtnClicked)
        backBtn = QPushButton("Back")
        backBtn.clicked.connect(self.backBtnClicked)
        rightBtn = QPushButton("Right")
        rightBtn.clicked.connect(self.rightBtnClicked)
        leftBtn = QPushButton("Left")
        leftBtn.clicked.connect(self.leftBtnClicked)

        # UIのレイアウト
        layout = QGridLayout()
        layout.addWidget(self.label,0,0)
        layout.addWidget(self.batteryLabel,0,1)
        layout.addWidget(self.timeLabel,0,2)
        layout.addWidget(self.tofLabel,0,3)
        #layout.addWidget(self.heightLabel,0,3)
        layout.addWidget(self.attitudeLabel,0,5)
        layout.addWidget(self.accelerationLabel,0,6)

        layout.addWidget(endBtn,0,8)
        layout.addWidget(takeoffBtn,0,4)
        layout.addWidget(landBtn,1,4)

        layout.addWidget(upBtn,3,1)
        layout.addWidget(downBtn,5,1)
        layout.addWidget(cwBtn,4,2)
        layout.addWidget(ccwBtn,4,0)

        layout.addWidget(forwardBtn,3,6)
        layout.addWidget(backBtn,5,6)
        layout.addWidget(rightBtn,4,7)
        layout.addWidget(leftBtn,4,5)

        self.setLayout(layout)

    # 終了処理
    def endBtnClicked(self):
        sys.exit()

    # 各種コマンド送信
    def takeoffBtnClicked(self):
        try:
            sent = self.sock.sendto('takeoff'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def landBtnClicked(self):
        try:
            sent = self.sock.sendto('land'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def upBtnClicked(self):
        try:
            sent = self.sock.sendto('up 20'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def downBtnClicked(self):
        try:
            sent = self.sock.sendto('down 20'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def cwBtnClicked(self):
        try:
            sent = self.sock.sendto('cw 45'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def ccwBtnClicked(self):
        try:
            sent = self.sock.sendto('ccw 45'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def forwardBtnClicked(self):
        try:
            sent = self.sock.sendto('forward 20'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def backBtnClicked(self):
        try:
            sent = self.sock.sendto('back 20'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def rightBtnClicked(self):
        try:
            sent = self.sock.sendto('right 20'.encode(encoding="utf-8"), self.tello)
        except:
            pass
    def leftBtnClicked(self):
        try:
            sent = self.sock.sendto('left 20'.encode(encoding="utf-8"), self.tello)
        except:
            pass

    # Telloからのレスポンス受信
    def recvSocket(self):
        while True: 
            try:
                data, server = self.sock.recvfrom()
                resp = data.decode(encoding="utf-8").strip()
                if resp.isdecimal():    # 数字だけなら充電量
                    #self.batteryLabel.setText(resp + "%")
                    print(resp)
                elif resp[-1:] == "s":  # 最後の文字がsなら飛行時間
                    #self.timeLabel.setText(resp)
                    print(resp)
                    f = open("time_2020_11_07_delivery_1support.txt", "a")
                    f.write(resp+"\n")
                    f.close() # リストの各要素をファイルに書き込み
                elif resp[-1:] == "m":  # 最後の文字がdmなら高度
                    #self.tofLabel.setText(resp)
                    print(resp)
                    f = open("tof_2020_11_07_delivery_1support.txt", "a")
                    f.write(resp+"\n")
                    f.close()
                    print(resp) # リストの各要素をファイルに書き込み
                #elif resp[-1:] == "m":  # 最後の文字がdmなら高度
                    #self.heightLabel.setText(resp)
                    #f = open("height_2020_11_07.txt", "a")
                    #f.write(resp+"\n")
                    #f.close() # リストの各要素をファイルに書き込み
                elif resp[0] == "p":  
                    #self.attitudeLabel.setText(resp)
                    print(resp)
                    f = open("attitude_2020_11_07_delivery_1support.txt", "a")
                    f.write(resp+"\n")
                    f.close() # リストの各要素をファイルに書き込み
                elif resp[0] == "a":  
                    #self.accelerationLabel.setText(resp)
                    print(resp)
                    f = open("acceleration_2020_11_07_delivery_1support.txt", "a")
                    f.write(resp+"\n")
                    f.close() # リストの各要素をファイルに書き込み
                elif resp == "OK":      # OKは黒
                    self.label.setStyleSheet("color:black;")
                    self.label.setText(resp)
                else:                   # それ以外は赤
                    self.label.setStyleSheet("color:red;")
                    self.label.setText(resp)
            except:
                pass

    # 問い合わせ
    def askTello(self):
        while True:
            try:
                sent = self.sock.sendto('battery?'.encode(encoding="utf-8"), self.tello)
            except:
                pass
            time.sleep(0.25)
            try:
                sent = self.sock.sendto('time?'.encode(encoding="utf-8"), self.tello)
            except:
                pass
            time.sleep(0.25)
            try:
                sent = self.sock.sendto('tof?'.encode(encoding="utf-8"), self.tello)
            except:
                pass
            #try:
                #sent = self.sock.sendto('height?'.encode(encoding="utf-8"), self.tello)
            #except:
                #pass
            #time.sleep(0.5)
            try:
                sent = self.sock.sendto('attitude?'.encode(encoding="utf-8"), self.tello)
            except:
                pass
            time.sleep(0.25)
            try:
                sent = self.sock.sendto('acceleration?'.encode(encoding="utf-8"), self.tello)
            except:
                pass
            time.sleep(0.25)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TelloController1()
    window.show()
    sys.exit(app.exec_())
