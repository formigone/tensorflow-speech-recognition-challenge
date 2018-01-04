import os

base = '/Users/rsilveira/rnd/tensorflow-speech-recognition-challenge/data_speech_commands_v0.01/test/audio'
files = sorted(os.listdir(base))

i = 0

for file in files:
  print('audio {} 10 audio'.format(file))
