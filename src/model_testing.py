import numpy as np
from numpy.core.defchararray import title
import tensorflow.keras as keras
from scipy.io import wavfile as wav
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.signal.windows.windows import cosine

filter_order=8 #orden del filtro
max_ripple=5 #Rizado máximo
freq_sample=22050 #Frecuencia de muestreo
filter_type='highpass' #Tipo de filtro

def apply_filter_low(x, crit_freq):
    sos = signal.cheby1(filter_order, max_ripple, crit_freq-1, 'lowpass', fs=freq_sample, output='sos')
    return signal.sosfilt(sos, x)

def apply_filter_high(x, crit_freq):
    sos = signal.cheby1(filter_order, max_ripple, crit_freq, 'highpass', fs=freq_sample, output='sos')
    return signal.sosfilt(sos, x)

def check_div(data, div):
  if data.shape[0] % div != 0:
    data = np.append(data, np.zeros(div - data.shape[0] % div))
  return data

def load_data(path):
  fs, data = wav.read(path)
  if len(data.shape)>1:
    if data.shape[1] > 1:
      data = (data[:,0] + data[:,1]) / 2
  data = data/np.max(np.abs(data))
  data = np.clip(data, -1, 1)
  return data.astype(np.float32), fs

def process_data(data, seg_size, normalize=False):
  data = check_div(data, seg_size)
  num_batches = data.shape[0]//seg_size
  if normalize:
    data = (data + 1) / 2
  data = np.reshape(data, newshape=(num_batches, seg_size, 1))
  return data

def save_data(data, fs, name_file, normalize = False):
  if len(data.shape)>1:
    data = np.reshape(data, newshape=(data.shape[0]*data.shape[1],))
  if normalize:
    data = (data * 2) - 1
  data = data.astype(np.float32)
  wav.write(name_file, fs, data)

scaling=4
freq_sample=22050
critical_freq=freq_sample/(2*scaling)

def preprocessing(x):
  filtered = apply_filter_low(x, 11025/2)
  resampled = signal.resample(filtered, int(len(x)//2))
  resampled = signal.resample(resampled, 2*int(len(x)))
  return resampled
###############################################################################


audio_path = 'dataset/voice_test.wav'
save_path = 'Results'
print('1.- Cargando modelo...')
generator = keras.models.load_model('Results/Models/generator_4.tf')
print('2.- Cargando audio...')
audio, fs = load_data(audio_path)
chunk_size = 4096

data_low = []

print('3.- Preparando audio...')
low_input = preprocessing(audio)

audio_fft = fft.fft(audio)
audio_fft = np.abs(audio_fft/len(audio_fft))
t = np.arange(0, len(audio_fft))/freq_sample
audio_freq = fft.fftfreq(len(audio_fft), 1/fs)

batch_low = process_data(low_input, seg_size=chunk_size*2, normalize=True)

for low in batch_low:
  data_low.append(low)

data_low = np.array(data_low)

counter_full = np.ones_like(data_low)
counter_half = np.ones_like(data_low)/2
counter_null = np.zeros_like(data_low)

print('4.- Generando audio...')
prediction = generator.predict(data_low, verbose=1)
print('5.- Creando perfil de ruido...')
counter_full = generator.predict(counter_full, verbose=1)
counter_half = generator.predict(counter_half, verbose=1)
counter_null = generator.predict(counter_null, verbose=1)

save_data(prediction, fs, save_path + '/voice_reconstructed_x2.wav', normalize=True)
save_data(counter_full, fs, save_path + '/filt_full_6_x2.wav', normalize=True)
save_data(counter_half, fs, save_path + '/filt_half_6_x2.wav', normalize=True)
save_data(counter_null, fs, save_path + '/filt_null_6_x2.wav', normalize=True)
print('5.- Audio generado')


