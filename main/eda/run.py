from src import analysis

def main():
    # Run the search first.
    print("Running session search by region...")
    region_sessions, common_sessions = analysis.search_sessions_by_region()
    
    # Report the total number of sessions in the union.
    all_sessions = set()
    for sessions in region_sessions.values():
        all_sessions.update(sessions)
    print(f"\nTotal unique sessions found (union): {len(all_sessions)}")
    for s in all_sessions:
        print(f"  {s}")
    
    # Filter common sessions to only those with the most variety in region combinations.
    diverse_sessions = analysis.select_diverse_sessions(region_sessions, common_sessions, max_sessions=30)
    print(f"\nRunning full analysis on a subset of {len(diverse_sessions)} diverse sessions...")
    sensitive_clusters = analysis.run_full_analysis(diverse_sessions)
    
    # Optionally, print the sensitive clusters by event type.
    print("\nFinal Sensitive Clusters:")
    print("Stimulus:", sensitive_clusters.get("stimulus", []))
    print("Movement:", sensitive_clusters.get("movement", []))
    print("Reward:", sensitive_clusters.get("reward", []))

if __name__ == "__main__":
    main()
