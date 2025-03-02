import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src import analysis

# --- Load configuration ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
BIN_SIZE = config["bin_size"]
PRE_TIME = config["pre_time"]
POST_TIME = config["post_time"]

def main():
    # Run session search by region.
    print("Running session search by region...")
    region_sessions, common_sessions = analysis.search_sessions_by_region()
    
    # Report total sessions.
    all_sessions = set()
    for sessions in region_sessions.values():
        all_sessions.update(sessions)
    print(f"\nTotal unique sessions found (union): {len(all_sessions)}")
    for s in all_sessions:
        print(f"  {s}")
    
    # Select sessions with diverse region combinations.
    diverse_sessions = analysis.select_diverse_sessions(region_sessions, common_sessions, max_sessions=1)
    print(f"\nRunning full analysis on a subset of {len(diverse_sessions)} diverse sessions...")
    sensitive_clusters = analysis.run_full_analysis(diverse_sessions)
    
    # Now, pick one session from your sensitive_clusters results.
    session_id = list(sensitive_clusters.keys())[0]
    print(f"\nRunning vLGP model on session: {session_id}")
    
    # Load data for this session.
    data = analysis.load_data(session_id)
    if data is None:
        print(f"Could not load data for session {session_id}.")
        return
    
    sl, spikes, clusters, channels, stimulus_events, movement_events, reward_events = data
    
    # For example, use the sensitive clusters from the stimulus analysis.
    sensitive_cluster_list = sensitive_clusters[session_id]["stimulus"]
    # Extract only the cluster IDs.
    sensitive_cluster_ids = [item[0] for item in sensitive_cluster_list]
    
    # Run the vLGP model using these sensitive clusters.
    fit = analysis.run_vlgp_model(stimulus_events, spikes, clusters, BIN_SIZE, PRE_TIME, POST_TIME, config, sensitive_cluster_ids)
    if fit is None:
        print("vLGP model fitting failed.")
        return

if __name__ == "__main__":
    main()
