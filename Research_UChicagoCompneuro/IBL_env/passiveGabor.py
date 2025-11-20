from one.api import ONE
import numpy as np
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from brainbox.population.decode import get_spike_counts_in_bins
from sklearn.decomposition import PCA
from iblatlas.atlas import AllenAtlas
from iblatlas import atlas

ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')
sessions=one.search()
eids=one.search(project='brainwide',subject='SWC_043')


def load_trials(eids,valid_eids):
    for eid in eids:
        datasets = one.list_datasets(eid)
        if not any('passiveGabor' in d for d in datasets):
            print(f"No passiveGabor dataset for {eid}.")
            continue
    
        passive_Gabor = one.load_dataset(eid, '*passiveGabor*', collection='alf')
        start = passive_Gabor['start']
        stop = passive_Gabor['stop']
        valid_eids.append(eid)
    return valid_eids

# check the repetition of stimulus
def run_PCA(eid):
    passive_Gabor = one.load_dataset(eid, '*passiveGabor*', collection='alf')
    start = passive_Gabor['start']
    end = passive_Gabor['stop']
    trials_list=[]
    for i in range (len(start)):
        trial_i = {
        'start': passive_Gabor['start'][i],
        'stop': passive_Gabor['stop'][i],
        'contrast': passive_Gabor['contrast'][i],
        'position': passive_Gabor['position'][i],
        'phase': passive_Gabor['phase'][i],
        }
        trials_list.append(trial_i)
    contrast=passive_Gabor["contrast"]
    contrast_unique= np.unique(contrast)

    pid = one.alyx.rest('insertions', 'list', session=eid)[0]['id']
    ba = AllenAtlas()
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    trials=np.c_[start, end]
    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'], spikes['clusters'],trials)
    start = np.array(start)
    end = np.array(end)
    fr = counts.T / (end - start)[:, None]
    mean_rate = []
    for val in (contrast_unique):
        mean_rate.append(fr[contrast == val].mean(axis=0))  # mean across trials for each neuron

    mean_rate = np.array(mean_rate)  # shape: (n_contrasts, n_neurons)

    residuals=np.zeros_like(fr)

    for i, val in enumerate(contrast_unique):
        residuals[contrast == val] = fr[contrast == val] - mean_rate[i]
        # mean center residuals
    residuals=residuals-residuals.mean(axis=0)
    # Signal PCA
    pca_signal_model = PCA(n_components=2)
    pca_signal_model.fit(mean_rate)
    signal_vec = pca_signal_model.components_[0]

    # Noise PCA
    pca_noise_model = PCA(n_components=2)
    pca_noise_model.fit(residuals)
    noise_vec = pca_noise_model.components_[0]

    # Compare angle between them (in neuron space)
    cos_angle = np.dot(signal_vec, noise_vec) / (np.linalg.norm(signal_vec)*np.linalg.norm(noise_vec))
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    angle_deg=float(angle_deg)
    return angle_deg
def load_trial_data (eid):
    passive_Gabor = one.load_dataset(eid, '*passiveGabor*', collection='alf')
    start = passive_Gabor['start']
    end = passive_Gabor['stop']
    trials_list=[]
    for i in range (len(start)):
        trial_i = {
        'start': start[i],
        'stop': end[i],
        'contrast': passive_Gabor['contrast'][i],
        'position': passive_Gabor['position'][i],
        'phase': passive_Gabor['phase'][i],
        }
        trials_list.append(trial_i)
    return trials_list
def load_probe_regions(eid):
    print(f"Loading probe regions for eid: {eid}")
    pid = one.alyx.rest('insertions', 'list', session=eid)[0]['id']
    ba = AllenAtlas()
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    channels = sl.load_channels()  
    ba = atlas.AllenAtlas()        
    xyz = np.c_[channels['x'], channels['y'], channels['z']]
    region_ids = ba.get_labels(xyz)
    acronyms=channels['acronym']
    print(np.unique(acronyms))
    print(np.unique(region_ids))


angles = []
valid_eids=[]
valid_eids=load_trials(eids, valid_eids)

for eid in valid_eids:
    load_probe_regions(eid)
    angle_deg=run_PCA(eid)
    angles.append(angle_deg)
for angle in angles:
    print(f"Angle between signal and noise PCs: {angle:.2f} degrees")

