import mne
import numpy as np
from mne_bids import BIDSPath
from nilearn.plotting import plot_markers
import matplotlib as plt

file_path = BIDSPath(root="/home/keitaro-sunagawa/Master-Research/ds005574/derivatives/ecogprep",
                     subject="02",
                     task="podcast",
                     datatype="ieeg",
                     description="highgamma",
                     suffix="ieeg",
                     extension="fif")
# print(f"File path within the dataset: {file_path}")

raw = mne.io.read_raw_fif(file_path, verbose=False)
fig = raw.plot_sensors()
fig.savefig('./tutorial.png')

ch2loc = {ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}
coords = np.vstack([ch2loc[ch] for ch in raw.info['ch_names']])
coords *= 1000  # nilearn likes to plot in meters, not mm
print("Coordinate matrix shape: ", coords.shape)

values = np.ones(len(coords))
fig = plot_markers(values, coords,
             node_size=30, display_mode='lzr', alpha=1,
             node_cmap='Grays', colorbar=False, node_vmin=0, node_vmax=1,
             output_file='plt_brain.png')
print(fig)