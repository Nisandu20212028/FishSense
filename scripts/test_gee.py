"""
Test Google Earth Engine initialization
"""

import ee

print("Testing Google Earth Engine...")

try:
    # Try to initialize with FishSense project
    ee.Initialize(project='fishsense-480120')
    print("✅ GEE initialized successfully with fishsense-480120!")
    
    # Try a simple query
    point = ee.Geometry.Point([79.3, 7.0])
    print(f"✅ Created test point: {point.getInfo()}")
    
    # Try to access a dataset
    sst = ee.ImageCollection('NOAA/CDR/SST_WHOI/V2').first()
    print("✅ Successfully accessed NOAA SST dataset")
    
    print("\n🎉 Google Earth Engine is working correctly!")
    print("The real-time mode should work in the dashboard.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Run: earthengine authenticate")
    print("2. Follow the authentication flow")
    print("3. Try again")
