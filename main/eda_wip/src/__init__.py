#eda/src/__init__.py

"""
This package contains modules for performing neural data analysis,
including session searches, data loading, and full analysis of spike data.
"""

# Import all functions from the analysis module so they can be accessed directly.
from .analysis import search_sessions_by_region, load_data, run_full_analysis
