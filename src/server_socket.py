import socket
from predictor_realtime import Predictor
import numpy as np
import threading
import time
import wave
import math
import audioop

class Sender():
    def __init__(self,conf_path):
       
        self.predictor = Predictor(conf_path)
        self.chunk = 512
        sk1 = socket.socket()
        sk1.bind(("192.168.16.166",8080))
        sk1.listen(5)
        self.cs1,addr1 = sk1.accept()

        sk = socket.socket()
        sk.bind(("192.168.16.166",20003))
        sk.listen(3)
        self.cs,addr2 = sk.accept()
        print('sk2 lined to %s'% str(addr2))

        self.wav_data=bytes()
        self.sample_width = 2

        #increase pause_seconds will harder to cut
        pause_seconds = 2.2
        bytes_len_per_seconds=32000
        self.pause_buffer_count = pause_seconds* bytes_len_per_seconds
        #increase energy_threshold will easier to cut
        self.energy_threshold = 0.83
        
        #self.wname = 0
        #self.wav_create()
    def wav_create(self):
        self.wf=wave.open('wav_tmp/'+str(self.wname)+'.wav','wb')
        self.wf.setnchannels(1)
        self.wf.setsampwidth(2)
        self.wf.setframerate(16000)

    def _wav_byte2np(self):
        align_point = len(self.wav_data) - len(self.wav_data)%8
        x = np.fromstring(self.wav_data[:align_point],np.int32)
        x = np.float32(x)
        self.wav_data = self.wav_data[align_point:]
        return x

    def process(self, sen_end=False):
        wav_bytes_len = len(self.wav_data)
        if not sen_end:
            if self.predictor._realtime_info[-1] is None:
                min_wav_len = 48000
            else:
                min_wav_len = 10000
            if wav_bytes_len > min_wav_len:
                #self.wf.writeframes(self.wav_data)
                self.predictor.predict_realtime(self._wav_byte2np(),False)

                predict_text = self.predictor.realtime_res()
                self.cs.send(predict_text.encode('utf-8'))

        else:
            if self.predictor._realtime_info[-1] is None:
                min_wav_len = 30000
            else:
                min_wav_len = 5000
            if wav_bytes_len > min_wav_len:
                #self.wf.writeframes(self.wav_data)
                self.predictor.predict_realtime(self._wav_byte2np(),True)
            else:
                print('warning sen end ,data insufficient')

            predict_text = self.predictor.realtime_res()
            self.cs.send(predict_text.encode('utf-8'))
            self.cs.send('next'.encode('utf-8'))

            self.predictor.flush_realtime()

            #self.wf.close()
            #self.wname += 1
            #self.wav_create()

    def run(self):
        pause_count = -100000000
        while True:
            received = self.cs1.recv(3000)
            received_len = len(received)
            if not received:
                self.process(True)
                break

            energy = float(audioop.rms(received,self.sample_width))/received_len
            if energy > self.energy_threshold:
                pause_count = received_len 
            else:
                pause_count += received_len 

            self.wav_data += received

            if pause_count > self.pause_buffer_count:
                t1=time.time() 
                self.process(True)
                #print('process ',time.time()-t1)
                pause_count = -100000000
            else:
                self.process()

        self.cs1.close()
        self.cs.close()

if __name__=='__main__':
    import sys
    conf_path = sys.argv[1]
    sd = Sender(conf_path)
    t1=time.time()
    print('clock on')
    sd.run()
    print('consumed',time.time()-t1)
