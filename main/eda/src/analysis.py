import os
import json
import numpy as np
import matplotlib.pyplot as plt
from one.api import ONE
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.singlecell import bin_spikes
from iblatlas.atlas import AllenAtlas
from statsmodels.stats.multitest import multipletests

# Enable interactive mode so that figures update without blocking.
plt.ion()

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

    # Filter sessions that appear in more than one region.
    common_sessions = {session for session, count in session_counts.items() if count > 1}
    print("\nSessions appearing in multiple regions (at least 2):")
    for session in common_sessions:
        print(f"  {session} appears in {session_counts[session]} regions.")
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

    print(f"\nStarting analysis on {len(sessions)} sessions (each appearing in at least 2 regions)...")

    # Containers to accumulate significant clusters for each event type over all sessions.
    significant_stim_clusters = []
    significant_movement_clusters = []
    significant_reward_clusters = []

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

        # Check that this session's clusters come from at least 2 regions.
        valid_regions = np.unique(clusters_acronyms[valid_cluster_ids])
        if len(valid_regions) < 2:
            print(f"Session {session} has clusters only from {valid_regions}. Skipping analysis.")
            continue
        print(f"Session {session} has clusters from {valid_regions}.")

        for cluster in valid_cluster_ids:
            spikes_idx = (spikes.clusters == cluster)
            spike_times = spikes.times[spikes_idx]
            
            # Bin spikes for each event type.
            stim_spike_raster, stim_times = bin_spikes(spike_times, stimulus_events,
                                                       pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
            stim_spike_raster = stim_spike_raster / BIN_SIZE

            move_spike_raster, move_times = bin_spikes(spike_times, movement_events,
                                                       pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
            move_spike_raster = move_spike_raster / BIN_SIZE

            reward_spike_raster, reward_times = bin_spikes(spike_times, reward_events,
                                                           pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
            reward_spike_raster = reward_spike_raster / BIN_SIZE

            print(f"\nProcessing cluster #{cluster}")
            # Compute PSTHs for stimulus-aligned trials.
            left_idx = ~np.isnan(sl.trials['contrastLeft'])
            right_idx = ~np.isnan(sl.trials['contrastRight'])
            psth_left = np.nanmean(stim_spike_raster[left_idx], axis=0)
            psth_right = np.nanmean(stim_spike_raster[right_idx], axis=0)
            
            correct_idx = sl.trials['feedbackType'] == 1
            incorrect_idx = sl.trials['feedbackType'] == -1
            psth_correct = np.nanmean(stim_spike_raster[correct_idx], axis=0)
            psth_incorrect = np.nanmean(stim_spike_raster[incorrect_idx], axis=0)

            # ------ Permutation testing for stimulus onset event --------------
            n_shuffles = 1000
            stim_d = np.zeros(stim_spike_raster.shape[1])
            stim_shuffled_d = np.zeros((n_shuffles, stim_spike_raster.shape[1]))
            for i in range(stim_spike_raster.shape[1]):
                stim_d[i] = np.nanmean(stim_spike_raster[right_idx, i]) - np.nanmean(stim_spike_raster[left_idx, i])
            for shuff in range(n_shuffles):
                shuffled_left = np.random.permutation(left_idx)
                shuffled_right = np.random.permutation(right_idx)
                for i in range(stim_spike_raster.shape[1]):
                    stim_shuffled_d[shuff, i] = np.nanmean(stim_spike_raster[shuffled_right, i]) - np.nanmean(stim_spike_raster[shuffled_left, i])
            stim_p_values = np.mean(np.abs(stim_shuffled_d) >= np.abs(stim_d), axis=0)
            alpha = 0.005
            bonferroni_threshold = alpha / stim_spike_raster.shape[1]
            stim_bonferroni_reject = stim_p_values < bonferroni_threshold
            remaining_p_values = stim_p_values[stim_bonferroni_reject]
            if len(remaining_p_values) > 0:
                _, stim_p_fdr_corrected, _, _ = multipletests(remaining_p_values, alpha=alpha, method='fdr_bh')
                stim_final_reject = np.copy(stim_bonferroni_reject)
                stim_final_reject[stim_bonferroni_reject] = stim_p_fdr_corrected < alpha
            else:
                stim_final_reject = np.zeros_like(stim_p_values, dtype=bool)
            if np.count_nonzero(stim_final_reject) > 5:
                significant_stim_clusters.append(cluster)
            
            # Plot stimulus event results without blocking:
            plt.figure(figsize=(10, 5))
            plt.plot(stim_times, stim_d, label="Observed Δ Firing Rate", color='blue')
            plt.title("Change in Δd firing rate by bin, rejection level 0.5%, event: stimOn")
            plt.axvline(0, color='black', linestyle='--', linewidth=2, label="Event Onset")
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            lower_bound = np.percentile(stim_shuffled_d, 0.25, axis=0)
            upper_bound = np.percentile(stim_shuffled_d, 99.75, axis=0)
            plt.fill_between(stim_times, lower_bound, upper_bound, color='gray', alpha=0.3, label="Null Distribution (99.5% CI)")
            plt.fill_between(stim_times, stim_d, where=stim_final_reject, color='red', alpha=0.3, label="Significant (FDR < 0.005)")
            plt.legend()
            plt.show(block=False)
            plt.pause(0.5)  # brief pause to render the plot
            plt.close()

            # ------ Permutation testing for movement event --------------
            movement_d = np.zeros(move_spike_raster.shape[1])
            shuffled_movement_d = np.zeros((n_shuffles, move_spike_raster.shape[1]))
            for i in range(move_spike_raster.shape[1]):
                movement_d[i] = np.nanmean(move_spike_raster[right_idx, i]) - np.nanmean(move_spike_raster[left_idx, i])
            for shuff in range(n_shuffles):
                shuffled_left = np.random.permutation(left_idx)
                shuffled_right = np.random.permutation(right_idx)
                for i in range(move_spike_raster.shape[1]):
                    shuffled_movement_d[shuff, i] = np.nanmean(move_spike_raster[shuffled_right, i]) - np.nanmean(move_spike_raster[shuffled_left, i])
            movement_p_values = np.mean(np.abs(shuffled_movement_d) >= np.abs(movement_d), axis=0)
            bonferroni_threshold = alpha / move_spike_raster.shape[1]
            movement_bonferroni_reject = movement_p_values < bonferroni_threshold
            remaining_p_values = movement_p_values[movement_bonferroni_reject]
            if len(remaining_p_values) > 0:
                _, movement_p_fdr_corrected, _, _ = multipletests(remaining_p_values, alpha=alpha, method='fdr_bh')
                movement_final_reject = np.copy(movement_bonferroni_reject)
                movement_final_reject[movement_bonferroni_reject] = movement_p_fdr_corrected < alpha
            else:
                movement_final_reject = np.zeros_like(movement_p_values, dtype=bool)
            if np.count_nonzero(movement_final_reject) > 5:
                significant_movement_clusters.append(cluster)
            
            plt.figure(figsize=(10, 5))
            plt.plot(move_times, movement_d, label="Observed Δ Firing Rate", color='blue')
            plt.title("Change in Δd firing rate by bin, rejection level 0.5%, event: movement")
            plt.axvline(0, color='black', linestyle='--', linewidth=2, label="Event Onset")
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            lower_bound = np.percentile(shuffled_movement_d, 0.25, axis=0)
            upper_bound = np.percentile(shuffled_movement_d, 99.75, axis=0)
            plt.fill_between(move_times, lower_bound, upper_bound, color='gray', alpha=0.3, label="Null Distribution (99.5% CI)")
            plt.fill_between(move_times, movement_d, where=movement_final_reject, color='red', alpha=0.3, label="Significant (FDR < 0.005)")
            plt.legend()
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

            # ------ Permutation testing for reward event --------------
            reward_d = np.zeros(reward_spike_raster.shape[1])
            shuffled_reward_d = np.zeros((n_shuffles, reward_spike_raster.shape[1]))
            for i in range(reward_spike_raster.shape[1]):
                reward_d[i] = np.nanmean(reward_spike_raster[right_idx, i]) - np.nanmean(reward_spike_raster[left_idx, i])
            for shuff in range(n_shuffles):
                shuffled_left = np.random.permutation(left_idx)
                shuffled_right = np.random.permutation(right_idx)
                for i in range(reward_spike_raster.shape[1]):
                    shuffled_reward_d[shuff, i] = np.nanmean(reward_spike_raster[shuffled_right, i]) - np.nanmean(reward_spike_raster[shuffled_left, i])
            reward_p_values = np.mean(np.abs(shuffled_reward_d) >= np.abs(reward_d), axis=0)
            bonferroni_threshold = alpha / reward_spike_raster.shape[1]
            reward_bonferroni_reject = reward_p_values < bonferroni_threshold
            remaining_p_values = reward_p_values[reward_bonferroni_reject]
            if len(remaining_p_values) > 0:
                _, reward_p_fdr_corrected, _, _ = multipletests(remaining_p_values, alpha=alpha, method='fdr_bh')
                reward_final_reject = np.copy(reward_bonferroni_reject)
                reward_final_reject[reward_bonferroni_reject] = reward_p_fdr_corrected < alpha
            else:
                reward_final_reject = np.zeros_like(reward_p_values, dtype=bool)
            if np.count_nonzero(reward_final_reject) > 5:
                significant_reward_clusters.append(cluster)
            
            plt.figure(figsize=(10, 5))
            plt.plot(reward_times, reward_d, label="Observed Δ Firing Rate", color='blue')
            plt.title("Change in Δd firing rate by bin, rejection level 0.5%, event: reward")
            plt.axvline(0, color='black', linestyle='--', linewidth=2, label="Event Onset")
            plt.axhline(0, color='gray', linestyle='--', linewidth=1)
            lower_bound = np.percentile(shuffled_reward_d, 0.25, axis=0)
            upper_bound = np.percentile(shuffled_reward_d, 99.75, axis=0)
            plt.fill_between(reward_times, lower_bound, upper_bound, color='gray', alpha=0.3, label="Null Distribution (99.5% CI)")
            plt.fill_between(reward_times, reward_d, where=reward_final_reject, color='red', alpha=0.3, label="Significant (FDR < 0.005)")
            plt.legend()
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

            ## ----- Sub-graphs: PSTHs and Contrast Comparisons --------------
            # For brevity, the subsequent plotting sections are similarly modified:
            # (Replace plt.show() with non-blocking calls and add a brief pause if needed.)
            # --- Example for PSTHs for stimulus events:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(stim_times, psth_left, c='green')
            axs[0].plot(stim_times, psth_right, c='yellow')
            axs[0].legend(['left', 'right'])
            axs[0].axvline(0, c='k', linestyle='--')
            axs[0].set_xlabel('Time from stimulus (s)')
            axs[0].set_ylabel('Firing rate (Hz)')
            
            axs[1].plot(stim_times, psth_correct, c='blue')
            axs[1].plot(stim_times, psth_incorrect, c='red')
            axs[1].legend(['correct', 'incorrect'])
            axs[1].axvline(0, c='k', linestyle='--')
            axs[1].set_xlabel('Time from stimulus (s)')
        
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

            # (Repeat similar non-blocking modifications for remaining figures ...)
    
    # At the end, print and return a dictionary with the sensitive clusters for each event type.
    result = {
       "stimulus": significant_stim_clusters,
       "movement": significant_movement_clusters,
       "reward": significant_reward_clusters
    }
    print("\nSignificant clusters by event type:")
    print(result)
    return result
def select_diverse_sessions(region_sessions, common_sessions, max_sessions=30):
    """
    Given the dictionary mapping region -> sessions and the set of common sessions,
    create a mapping from session -> set of regions, then sort sessions by the number of
    regions (in descending order) and return at most max_sessions.
    """
    session_to_regions = {}
    for region, sessions in region_sessions.items():
        for session in sessions:
            if session in common_sessions:
                session_to_regions.setdefault(session, set()).add(region)
                
    # Sort sessions by how many regions they have (highest diversity first)
    sorted_sessions = sorted(session_to_regions.keys(),
                             key=lambda s: len(session_to_regions[s]),
                             reverse=True)
    
    # Select the top sessions (or fewer if not enough sessions)
    selected_sessions = sorted_sessions[:max_sessions]
    
    print("\nSelected sessions for diversity:")
    for session in selected_sessions:
        regions = session_to_regions[session]
        print(f"  {session}: Regions = {regions}")
        
    return selected_sessions