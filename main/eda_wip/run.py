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
    diverse_sessions = analysis.select_diverse_sessions(region_sessions, common_sessions, max_sessions=10)
    print(f"\nRunning full analysis on a subset of {len(diverse_sessions)} diverse sessions...")
    
     # Run full analysis to obtain sensitive clusters per session/event type.
    sensitive_clusters = analysis.run_full_analysis(diverse_sessions)
    
    best_session = analysis.select_best_session_by_sensitivity(sensitive_clusters)
    if best_session is None:
        print("No session met the criteria for sensitive clusters in at least two regions.")
    else:
        print(f"\nThe best session based on sensitivity is: {best_session}")
        # Create a dictionary with only the best session's sensitive clusters.
        best_sensitive_clusters = {best_session: sensitive_clusters[best_session]}
        
        # Fit vLGP models by region using the best session's sensitive clusters.
        fitted_models = analysis.fit_vlgp_models_for_best_session(best_sensitive_clusters, best_session)
        
        # Print summary of fitted models.
        for session, event_dict in fitted_models.items():
            for event_type, region_dict in event_dict.items():
                for region, model in region_dict.items():
                    print(f"Session {session} | Event {event_type} | Region {region} fitted model with {len(model['trials'])} trials.")
    
if __name__ == "__main__":
    main()