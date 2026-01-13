"""
FishSense - Google Earth Engine Utilities
Reusable functions for GEE operations
"""

import ee
from datetime import datetime, timedelta

def initialize_gee():
    """
    Initialize Google Earth Engine
    Returns True if successful, False otherwise
    """
    try:
        ee.Initialize(project='fishsense-480120')
        return True
    except Exception as e:
        print(f"❌ Error initializing GEE: {e}")
        print("➤ Try running: earthengine authenticate")
        return False

def get_study_area():
    """
    Get the bounding box for Sri Lankan coastal waters
    Returns ee.Geometry.Rectangle
    """
    # Sri Lankan coastal waters
    # Format: [west, south, east, north]
    return ee.Geometry.Rectangle([79.5, 5.9, 81.9, 9.9])

def get_date_range(days_back=30):
    """
    Get date range for data retrieval
    
    Args:
        days_back: Number of days to look back from today
    
    Returns:
        tuple: (start_date_string, end_date_string)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    date_start = start_date.strftime('%Y-%m-%d')
    date_end = end_date.strftime('%Y-%m-%d')
    
    return date_start, date_end

def test_connection():
    """
    Quick test to verify GEE connection is working
    Returns True if working, False otherwise
    """
    try:
        if not initialize_gee():
            return False
        
        # Try a simple operation
        test = ee.Image(1).getInfo()
        print("✓ GEE connection is working!")
        return True
    except Exception as e:
        print(f"✗ GEE connection failed: {e}")
        return False

# If this file is run directly, test the connection
if __name__ == "__main__":
    print("Testing GEE utilities...")
    if test_connection():
        print("\n✓ All utilities are working!")
        
        # Test study area
        study_area = get_study_area()
        print(f"✓ Study area loaded: Sri Lankan coastal waters")
        
        # Test date range
        start, end = get_date_range(30)
        print(f"✓ Date range: {start} to {end}")
    else:
        print("\n✗ GEE utilities test failed")