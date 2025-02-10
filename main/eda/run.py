import argparse
from src import analysis

def main():
    # Run the search first.
    print("Running session search by region...")
    region_sessions, common_sessions = analysis.search_sessions_by_region()
    
    # Build the union of sessions from all regions.
    all_sessions = set()
    for sessions in region_sessions.values():
        all_sessions.update(sessions)
    print(f"\nTotal unique sessions found (union): {len(all_sessions)}")
    for s in all_sessions:
        print(f"  {s}")
    
    # Run full analysis on the union of sessions.
    print("\nRunning full analysis on sessions with at least 2 shared regions...")
    analysis.run_full_analysis(all_sessions)

if __name__ == "__main__":
    main()
