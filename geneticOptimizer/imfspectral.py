import emd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sample_rate = 0.2
df = pd.read_csv('./BKDataCleaned/Floor2Z1.csv', parse_dates=True, index_col=["Date"])
channel = "temperature"
x=df[channel].iloc[:1000].values
imf = emd.sift.sift(x)
time_vect = [i for i in range(len(x))]
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')
# import pdb; pdb.set_trace()
freq_edges, freq_bins = emd.spectra.define_hist_bins(0, 10, 100)
hht = emd.spectra.hilberthuang(IF, IA, freq_edges)


plt.figure(figsize=(16, 8))
plt.subplot(211, frameon=False)
plt.plot(time_vect, x, 'k')
plt.plot(time_vect, imf[:, 0]-4, 'r')
plt.plot(time_vect, imf[:, 1]-8, 'g')
plt.plot(time_vect, imf[:, 2]-12, 'b')
plt.xlim(time_vect[0], time_vect[-1])
plt.grid(True)
plt.subplot(212)
plt.pcolormesh(time_vect, freq_bins, hht, cmap='ocean_r')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (secs)')
plt.grid(True)
plt.show()