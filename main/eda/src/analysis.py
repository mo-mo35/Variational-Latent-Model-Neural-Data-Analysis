import os
import json
import numpy as np
import matplotlib.pyplot as plt
from one.api import ONE
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.singlecell import bin_spikes
from iblatlas.atlas import AllenAtlas
from statsmodels.stats.multitest import multipletests

# --- Load configuration ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

BIN_SIZE = config["bin_size"]
PRE_TIME = config["pre_time"]
POST_TIME = config["post_time"]

# --- ONE API and Allen Atlas ---
ba = AllenAtlas()
one = ONE()

def search_sessions_by_region():
    print("Regions in config:", config["regions"])
    region_sessions = {}
    for region in config["regions"]:
        print(f"Searching sessions for region: {region}...")
        try:
            sessions = one.search_insertions(atlas_acronym=[region], project="brainwide")
            region_sessions[region] = set(sessions)
            print(f"Region {region}: {len(sessions)} sessions found.")
            if len(sessions) < 3:
                print(f"Warning: fewer than 3 sessions for region {region}!")
        except Exception as e:
            print(f"Error searching for sessions in region {region}: {e}")
    # Count sessions across regions.
    session_counts = {}
    for sessions in region_sessions.values():
        for session in sessions:
            session_counts[session] = session_counts.get(session, 0) + 1

    common_sessions = {session for session, count in session_counts.items() if count > 1}
    print("\nSessions appearing in multiple regions:")
    for session, count in session_counts.items():
        if count > 1:
            print(f"{session} appears in {count} regions.")
    return region_sessions, common_sessions

def load_data(session):
    """
    Loads the session data. Returns a tuple or None if an error occurs.
    """
    print(f"\nLoading data for session: {session}")
    try:
        pid = session  # session is the PID.
        eid, _ = one.pid2eid(pid)
        sl = SessionLoader(eid=eid, one=one)
        sl.load_trials()

        ssl = SpikeSortingLoader(one=one, pid=pid, atlas=ba)
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels)

        stimulus_events = np.array(sl.trials['stimOn_times'])
        movement_events = np.array(sl.trials['firstMovement_times'])
        reward_events = np.array(sl.trials['feedback_times'])

        print(f"Data loaded for session {session}.")
        return sl, spikes, clusters, channels, stimulus_events, movement_events, reward_events
    except Exception as e:
        print(f"Error loading data for session {session}: {e}")
        return None


def permutation_test(observed_diff, shuffled_diffs, alpha=0.005):
    p_values = np.mean(np.abs(shuffled_diffs) >= np.abs(observed_diff), axis=0)
    bonferroni_threshold = alpha / len(observed_diff)
    reject_bonferroni = p_values < bonferroni_threshold
    
    if np.any(reject_bonferroni):
        _, p_fdr_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        reject_fdr = p_fdr_corrected < alpha
    else:
        reject_fdr = np.zeros_like(p_values, dtype=bool)
    
    return reject_fdr

def run_full_analysis(sessions):
    if not sessions:
        print("No sessions to process. Exiting analysis.")
        return

    print(f"\nStarting analysis on {len(sessions)} sessions...")

    for session in sessions:
        print(f"\n=== Processing session: {session} ===")
        data = load_data(session)
        if data is None:
            print(f"Skipping session {session} due to data loading issues.")
            continue

        sl, spikes, clusters, channels, stimulus_events, movement_events, reward_events = data

        clusters_acronyms = clusters['acronym'].astype(str)
        valid_mask = (clusters['label'] == 1) & np.isin(clusters_acronyms, config["regions"])
        valid_cluster_ids = np.where(valid_mask)[0]

        if len(valid_cluster_ids) == 0:
            print(f"No valid clusters in session {session}. Skipping.")
            continue

        valid_regions = np.unique(clusters_acronyms[valid_cluster_ids])
        if len(valid_regions) < 2:
            print(f"Session {session} has clusters only from {valid_regions}. Skipping analysis.")
            continue
        print(f"Session {session} has clusters from {valid_regions}.")

        for cluster in valid_cluster_ids:
            cluster_region = clusters_acronyms[cluster]
            print(f"\n--> Processing cluster {cluster} from region {cluster_region} in session {session}")

            spike_times = spikes.times[spikes.clusters == cluster]

            stim_raster, stim_times = bin_spikes(spike_times, stimulus_events, PRE_TIME, POST_TIME, BIN_SIZE)
            move_raster, move_times = bin_spikes(spike_times, movement_events, PRE_TIME, POST_TIME, BIN_SIZE)
            reward_raster, reward_times = bin_spikes(spike_times, reward_events, PRE_TIME, POST_TIME, BIN_SIZE)

            stim_raster /= BIN_SIZE
            move_raster /= BIN_SIZE
            reward_raster /= BIN_SIZE

            left_idx = ~np.isnan(sl.trials['contrastLeft'])
            right_idx = ~np.isnan(sl.trials['contrastRight'])

            stim_diff = np.nanmean(stim_raster[right_idx], axis=0) - np.nanmean(stim_raster[left_idx], axis=0)
            shuffled_stim_diff = np.array([
                np.nanmean(stim_raster[np.random.permutation(right_idx)], axis=0) - 
                np.nanmean(stim_raster[np.random.permutation(left_idx)], axis=0)
                for _ in range(1000)
            ])
            stim_significant = permutation_test(stim_diff, shuffled_stim_diff)

            pre_movement = move_times < 0
            post_movement = move_times >= 0

            move_diff = np.nanmean(move_raster[:, post_movement], axis=1) - np.nanmean(move_raster[:, pre_movement], axis=1)
            shuffled_move_diff = np.array([
                np.nanmean(move_raster[:, np.random.permutation(post_movement)], axis=1) -
                np.nanmean(move_raster[:, np.random.permutation(pre_movement)], axis=1)
                for _ in range(1000)
            ])
            move_significant = permutation_test(move_diff, shuffled_move_diff)

            pre_reward = reward_times < 0
            post_reward = reward_times >= 0

            reward_diff = np.nanmean(reward_raster[:, post_reward], axis=1) - np.nanmean(reward_raster[:, pre_reward], axis=1)
            shuffled_reward_diff = np.array([
                np.nanmean(reward_raster[:, np.random.permutation(post_reward)], axis=1) - 
                np.nanmean(reward_raster[:, np.random.permutation(pre_reward)], axis=1)
                for _ in range(1000)
            ])
            reward_significant = permutation_test(reward_diff, shuffled_reward_diff)

            plot_original_graphs(stim_times, stim_diff, stim_significant, "Stimulus Response")
            plot_original_graphs(move_times, move_diff, move_significant, "Movement Response")
            plot_original_graphs(reward_times, reward_diff, reward_significant, "Reward Response")

    print("\nAnalysis complete.")
