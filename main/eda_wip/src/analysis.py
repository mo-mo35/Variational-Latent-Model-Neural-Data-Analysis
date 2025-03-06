import os
import json
import numpy as np
import matplotlib.pyplot as plt
from one.api import ONE
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.singlecell import bin_spikes
from iblatlas.atlas import AllenAtlas
from statsmodels.stats.multitest import multipletests
import vlgp
import pickle

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

def run_full_analysis(sessions):
    """
    For each session and for each valid cluster:
      (1) For each event type (stimulus, movement, reward):
            - Bin the spikes.
            - Run a permutation test with FDR correction across time bins (per cell).
            - Plot the permutation test results using the FDR-corrected p-values,
              and include the null distribution (gray shading) as well.
            - Plot PSTH comparisons (left/right and correct/incorrect).
      (2) After processing all event types for the cluster, perform contrast comparisons 
          (bar graphs) once using stimulus event data.
    
    Returns:
      result_clusters[session] = {
           "stimulus": [(cluster_id, region), ...],
           "movement": [(cluster_id, region), ...],
           "reward": [(cluster_id, region), ...]
      }
    """
    if not sessions:
        print("No sessions to process. Exiting analysis.")
        return {}

    print(f"\nStarting analysis on {len(sessions)} sessions (each appearing in at least 2 regions)...")
    result_clusters = {}
    alpha = 0.005  # significance threshold

    for session in sessions:
        print(f"\n=== Processing session: {session} ===")
        data = load_data(session)
        if data is None:
            print(f"Skipping session {session} due to data loading issues.")
            continue

        sl, spikes, clusters, channels, stimulus_events, movement_events, reward_events = data

        # Define event times.
        event_times = {
            "stimulus": stimulus_events,
            "movement": movement_events,
            "reward": reward_events
        }

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
        print(f"Session {session} has clusters from regions: {valid_regions}")

        # Containers for plotting-based significance (for visual feedback).
        significant_stim_clusters = []
        significant_movement_clusters = []
        significant_reward_clusters = []

        # Precompute condition indices (per session).
        left_idx = ~np.isnan(sl.trials['contrastLeft'])
        right_idx = ~np.isnan(sl.trials['contrastRight'])
        correct_idx = sl.trials['feedbackType'] == 1
        incorrect_idx = sl.trials['feedbackType'] == -1

        # Loop over each valid cluster.
        for cluster in valid_cluster_ids:
            spikes_idx = (spikes.clusters == cluster)
            spike_times = spikes.times[spikes_idx]
            region_label = clusters['acronym'][cluster]

            # Process every event type for permutation test and PSTH.
            for event_type in ["stimulus", "movement", "reward"]:
                events = event_times[event_type]
                # Bin spikes for current event type.
                binned_spikes, trial_times = bin_spikes(spike_times,
                                                        events,
                                                        pre_time=PRE_TIME,
                                                        post_time=POST_TIME,
                                                        bin_size=BIN_SIZE)
                # Convert counts to firing rate.
                spike_raster = binned_spikes / BIN_SIZE
                n_bins = spike_raster.shape[1]

                # --------------------
                # New False Positive Pipeline (using FDR correction only):
                # --------------------
                obs_diff = np.zeros(n_bins)
                for i in range(n_bins):
                    obs_diff[i] = np.nanmean(spike_raster[left_idx, i]) - np.nanmean(spike_raster[right_idx, i])
                n_shuffles = 1000
                null_diffs = np.zeros((n_shuffles, n_bins))
                for shuff in range(n_shuffles):
                    shuffled_left = np.random.permutation(left_idx)
                    shuffled_right = np.random.permutation(right_idx)
                    for i in range(n_bins):
                        null_diffs[shuff, i] = np.nanmean(spike_raster[shuffled_left, i]) - \
                                               np.nanmean(spike_raster[shuffled_right, i])
                pvals = np.mean(np.abs(null_diffs) >= np.abs(obs_diff), axis=0)
                _, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
                sig_bins = pvals_fdr < alpha
                '''# Compute null distribution percentiles for shading.
                lower_bound = np.percentile(null_diffs, 0.25, axis=0)
                upper_bound = np.percentile(null_diffs, 99.75, axis=0)

                # --------------------
                # Permutation Test Plot using FDR-corrected p-values:
                # --------------------
                
                plt.figure(figsize=(10, 5))
                plt.plot(trial_times, obs_diff, label="Observed Δ Firing Rate", color='blue')
                plt.fill_between(trial_times, lower_bound, upper_bound, color='gray', alpha=0.3, label="Null Distribution (99.5% CI)")
                plt.fill_between(trial_times, obs_diff, where=sig_bins, color='red', alpha=0.3, label="Significant (FDR < 0.005)")
                plt.axvline(0, color='black', linestyle='--', linewidth=2, label="Event Onset")
                plt.axhline(0, color='gray', linestyle='--', linewidth=1)
                plt.legend()
                plt.title(f"Δ Firing Rate by Bin (FDR corrected)\nEvent: {event_type}, Cluster: {cluster}, Region: {region_label}")
                plt.show(block=False)
                plt.pause(2.5)
                plt.close()'''

                # Check if this event for the cluster meets the minimum of 5 significant bins.
                if np.count_nonzero(sig_bins) > 5:
                    if event_type == "stimulus":
                        significant_stim_clusters.append((cluster, region_label))
                    elif event_type == "movement":
                        significant_movement_clusters.append((cluster, region_label))
                    elif event_type == "reward":
                        significant_reward_clusters.append((cluster, region_label))

                '''# --------------------
                # PSTH Plots for current event type.
                # --------------------
                psth_left = np.nanmean(spike_raster[left_idx], axis=0)
                psth_right = np.nanmean(spike_raster[right_idx], axis=0)
                psth_correct = np.nanmean(spike_raster[correct_idx], axis=0)
                psth_incorrect = np.nanmean(spike_raster[incorrect_idx], axis=0)

                fig, axs = plt.subplots(1, 2)
                axs[0].plot(trial_times, psth_left, c='green')
                axs[0].plot(trial_times, psth_right, c='yellow')
                axs[0].legend(['left', 'right'])
                axs[0].axvline(0, c='k', linestyle='--')
                axs[0].set_xlabel(f"Time from {event_type} (s)")
                axs[0].set_ylabel('Firing rate (Hz)')
                
                axs[1].plot(trial_times, psth_correct, c='blue')
                axs[1].plot(trial_times, psth_incorrect, c='red')
                axs[1].legend(['correct', 'incorrect'])
                axs[1].axvline(0, c='k', linestyle='--')
                axs[1].set_xlabel(f"Time from {event_type} (s)")
                fig.suptitle(f"Firing Rate after {event_type} Event\nCluster: {cluster}, Region: {region_label}")
                plt.show(block=False)
                plt.pause(2.5)
                plt.close()

            # End event type loop for this cluster.

            # --------------------
            # Contrast Comparisons (run once per cluster, independent of event loop)
            # --------------------
            # Bin spikes for the stimulus event separately.
            binned_stim, stim_times = bin_spikes(spike_times,
                                                 stimulus_events,
                                                 pre_time=PRE_TIME,
                                                 post_time=POST_TIME,
                                                 bin_size=BIN_SIZE)
            stim_spike_raster = binned_stim / BIN_SIZE

            contrast_levels_left = np.unique(sl.trials['contrastLeft'][~np.isnan(sl.trials['contrastLeft'])])
            contrast_levels_right = np.unique(sl.trials['contrastRight'][~np.isnan(sl.trials['contrastRight'])])
            
            # Initialize lists for firing rates.
            firing_rates_correct_left = []
            firing_rates_incorrect_left = []
            firing_rates_correct_right = []
            firing_rates_incorrect_right = []
            
            for contrast in contrast_levels_left:
                idx = sl.trials['contrastLeft'] == contrast
                firing_rates_correct_left.append(np.nanmean(stim_spike_raster[idx & correct_idx]))
                firing_rates_incorrect_left.append(np.nanmean(stim_spike_raster[idx & incorrect_idx]))
            
            for contrast in contrast_levels_right:
                idx = sl.trials['contrastRight'] == contrast
                firing_rates_correct_right.append(np.nanmean(stim_spike_raster[idx & correct_idx]))
                firing_rates_incorrect_right.append(np.nanmean(stim_spike_raster[idx & incorrect_idx]))
            
            # Convert contrast values to string labels.
            contrast_labels_left = [str(c) for c in contrast_levels_left]
            contrast_labels_right = [str(c) for c in contrast_levels_right]
            
            # Plot contrast comparison bar graphs.
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            x = np.arange(len(contrast_levels_left))
            axs[0].bar(x - 0.2, firing_rates_correct_left, width=0.4, label='Correct', color='green')
            axs[0].bar(x + 0.2, firing_rates_incorrect_left, width=0.4, label='Incorrect', color='orange')
            axs[0].set_xticks(x)
            axs[0].set_xticklabels(contrast_labels_left)
            axs[0].set_xlabel('Contrast Left')
            axs[0].set_ylabel('Firing Rate (Hz)')
            axs[0].set_title(f"Firing Rate by Contrast Left Cluster: {cluster}, Region: {region_label}")
            axs[0].legend()
    
            x = np.arange(len(contrast_levels_right))
            axs[1].bar(x - 0.2, firing_rates_correct_right, width=0.4, label='Correct', color='green')
            axs[1].bar(x + 0.2, firing_rates_incorrect_right, width=0.4, label='Incorrect', color='orange')
            axs[1].set_xticks(x)
            axs[1].set_xticklabels(contrast_labels_right)
            axs[1].set_xlabel('Contrast Right')
            axs[1].set_ylabel('Firing Rate (Hz)')
            axs[1].set_title(f"Firing Rate by Contrast Right Cluster: {cluster}, Region: {region_label}")
            axs[1].legend()
    
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(2.5)
            plt.close()'''

        # End cluster loop.
        # Use the FDR-corrected p-values to determine significant clusters (if any event has >5 significant bins).
        sig_clusters = {"stimulus": significant_stim_clusters,
                        "movement": significant_movement_clusters,
                        "reward": significant_reward_clusters}
        result_clusters[session] = {"stimulus": sig_clusters["stimulus"],
                                    "movement": sig_clusters["movement"],
                                    "reward": sig_clusters["reward"]}
        print(f"\nSession {session} significant clusters by event type:")
        print(result_clusters[session])
    print("\nOverall significant clusters by event type:")
    print(result_clusters)
    return result_clusters




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


def run_vlgp_model(events, spikes, clusters, BIN_SIZE, PRE_TIME, POST_TIME, config, sensitive_cluster_ids):
    print("Using sensitive clusters (IDs):", sensitive_cluster_ids)
    n_trials = len(events)
    trials_vlgp = []
    
    # For each trial, create a trial matrix with one column per sensitive cluster.
    for trial_idx in range(n_trials):
        trial_neuron_data = []  # binned spike data for each cluster
        region_labels = []      # store corresponding region labels
        for cluster in sensitive_cluster_ids:
            spikes_idx = (spikes.clusters == cluster)
            spike_times = spikes.times[spikes_idx]
            binned_spikes, trial_times = bin_spikes(spike_times,
                                                    events,
                                                    pre_time=PRE_TIME,
                                                    post_time=POST_TIME,
                                                    bin_size=BIN_SIZE)
            # Convert counts to firing rate.
            binned_spikes = binned_spikes / BIN_SIZE
            trial_neuron_data.append(binned_spikes[0])
            # Get region info from clusters.
            region = clusters['acronym'][cluster]
            region_labels.append(region)
            
        trial_matrix = np.column_stack(trial_neuron_data)
        
        trials_vlgp.append({'ID': trial_idx, 'y': trial_matrix, 'regions': region_labels})
    
    print("Created", len(trials_vlgp), "trials for vLGP model.")
    
    # Adjust the number of latent factors: it must not exceed the number of neurons.
    n_neurons = trials_vlgp[0]['y'].shape[1] if trials_vlgp else 0
    n_factors = 3 if n_neurons >= 3 else n_neurons
    print(f"Fitting vLGP model with n_factors = {n_factors} (n_neurons = {n_neurons})")
    
    fit = vlgp.fit(trials_vlgp, n_factors=n_factors, max_iter=20, min_iter=10)
    # Save the fitted model (or just the trials dictionary)
    with open("vlgp_model.pkl", "wb") as f:
        pickle.dump(fit, f)
    return fit


def fit_vlgp_models_by_region(result_clusters, top_regions=2):
    """
    From the sensitive_clusters dictionary (structured as:
      { session: { event_type: [(cluster, region), ...] } }
    ),
    this function selects the "best" session (with the most sensitive clusters overall)
    that has sensitive clusters from at least 2 different regions. For each event type in that session,
    it keeps only the top `top_regions` regions with the highest number of sensitive clusters,
    then fits the vLGP models for those regions.
    
    Parameters:
      result_clusters (dict): Dictionary containing sensitive clusters information.
      top_regions (int): Number of top regions to select based on the count of sensitive clusters.
      
    Returns:
      fitted_models (dict): Dictionary with fitted vLGP models for the best session.
    """
    # Filter out sessions that have sensitive clusters in only one region.
    valid_sessions = {}
    for session, events in result_clusters.items():
        session_regions = set()
        for event_type, clusters in events.items():
            for cluster, region in clusters:
                session_regions.add(region)
        if len(session_regions) >= 2:
            valid_sessions[session] = events
        else:
            print(f"Skipping session {session} because it only has sensitive clusters from one region: {session_regions}")
    
    if not valid_sessions:
        print("No sessions with sensitive clusters in at least 2 regions. Exiting model fitting.")
        return {}

    # Identify the best session by total count of sensitive clusters across event types.
    best_session = max(valid_sessions.keys(), key=lambda s: sum(len(valid_sessions[s][et]) for et in valid_sessions[s]))
    print(f"Best session selected: {best_session}")
    
    # Filter sensitive clusters for the best session: for each event type, keep only top `top_regions` regions.
    filtered_sensitive = {}
    for event_type, clusters_list in valid_sessions[best_session].items():
        # Count the number of clusters per region.
        region_counts = {}
        for cluster, region in clusters_list:
            region_counts[region] = region_counts.get(region, 0) + 1
        
        # Sort regions by number of clusters in descending order and keep top `top_regions`.
        top_region_list = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)[:top_regions]
        top_region_names = [region for region, count in top_region_list]
        print(f"Event {event_type}: Top regions selected: {top_region_names}")
        
        # Filter the clusters: only keep those in the top regions.
        filtered_sensitive[event_type] = [(cluster, region) for cluster, region in clusters_list if region in top_region_names]
    
    # Load session data for the best session.
    data = load_data(best_session)
    if data is None:
        print("Data loading error for best session.")
        return {}
    sl, spikes, clusters, channels, stimulus_events, movement_events, reward_events = data
    event_times = {
        "stimulus": stimulus_events,
        "movement": movement_events,
        "reward": reward_events
    }
    
    # Fit vLGP models for each event type and for each top region.
    fitted_models = {best_session: {}}
    for event_type, clusters_list in filtered_sensitive.items():
        # Group clusters by region.
        region_groups = {}
        for cluster, region in clusters_list:
            region_groups.setdefault(region, []).append(cluster)
        
        fitted_models[best_session][event_type] = {}
        for region, cluster_list in region_groups.items():
            print(f"Fitting vLGP for Session {best_session}, Event {event_type}, Region {region} with clusters {cluster_list}")
            # Use the appropriate event times.
            events = event_times.get(event_type, stimulus_events)
            # Fit the model using your existing run_vlgp_model function.
            fitted_model = run_vlgp_model(events, spikes, clusters, BIN_SIZE, PRE_TIME, POST_TIME, config, cluster_list)
            fitted_models[best_session][event_type][region] = fitted_model
    
    return fitted_models





def filter_sensitive_clusters(sensitive_clusters, top_n_regions=2):
    """
    Given sensitive_clusters (a dict structured as:
      { session: { event_type: [(cluster, region), ...] } }
    ),
    filter the clusters for each event type in each session to keep only the top_n_regions
    (i.e. the regions with the highest number of sensitive clusters).
    Returns a new dictionary in the same structure but with filtered lists.
    """
    filtered_sensitive = {}
    for session, event_dict in sensitive_clusters.items():
        filtered_sensitive[session] = {}
        for event_type, clusters_list in event_dict.items():
            # Group clusters by region.
            region_groups = {}
            for (cluster, region) in clusters_list:
                region_groups.setdefault(region, []).append(cluster)
            # Sort regions by the number of sensitive clusters (descending).
            sorted_regions = sorted(region_groups.items(), key=lambda x: len(x[1]), reverse=True)
            # Keep only the top_n_regions.
            filtered_event = []
            for region, clusters in sorted_regions[:top_n_regions]:
                # Append each (cluster, region) pair.
                for cluster in clusters:
                    filtered_event.append((cluster, region))
            filtered_sensitive[session][event_type] = filtered_event
    return filtered_sensitive