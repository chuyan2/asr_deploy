import error_rate
import wave
from predictor_realtime import Predictor

class Demo():
    def __init__(self,conf_path='configs/test.config'):
        self.predictor = Predictor(conf_path)
    def test(self,audio_path):
        return self.predictor.predict(audio_path)

    def check_conv(self,x):
        print('in check conv')
        print(x.shape)
        focus = x[0][1][1]
        print(focus.shape)
        print(focus)
    def test_realtime(self,audio_path):
        full = self.predictor.predict(audio_path)   
        print(full)
        """
        self.check_conv(full)
        print('sliced')
        sli_num = 4
        full_len = len(audio_data)
        assert full_len % sli_num == 0
        sli_len = int(full_len / sli_num)
        for i in range(sli_num):
            sli_len = 161* 20
            sli = self.predictor.predict(audio_data[i*sli_len:i*sli_len+sli_len])   
            self.check_conv(sli)
        """
    def test_result(self,audio_path):
        assert audio_path.endswith("csv")
        total_cer ,case_num = 0,0
        import time
        with open(audio_path,'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                if ',' in line:
                    wav,txt = line.split(',')
                    t1=time.time()
                    p = self.predictor.predict(wav)
                    
                    print(time.time()-t1)
                    with open(txt) as t:
                        label = t.read()
                        if label[-1] == '\n':
                            label = label[:-1]
                        one_cer = error_rate.cer(label,p)
                        total_cer += one_cer
                        print("label",label)
                        print("predict",p)
                        print("cer",one_cer)
                        case_num += 1
        average_cer = total_cer/case_num
        return average_cer

        
if __name__ == "__main__":
    demo = Demo()
    import sys
    data = sys.argv[1]
    if data.endswith("wav"):
        demo.test_realtime(data)
    elif data.endswith("csv"):
        print("average cer:",demo.test_result(data))
    else:
        print("input wrong")
