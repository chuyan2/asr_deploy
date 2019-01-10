export PYTHONPATH=$PWD'/src/'
mkdir wav_tmp
rm wav_tmp/*
#python src/server2.py configs/test.config
python src/server_socket.py configs/test.config
