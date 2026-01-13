"""
Test Google Earth Engine Connection
Verifies GEE is properly set up and working
"""

import ee

print("=" * 70)
print("TESTING GOOGLE EARTH ENGINE CONNECTION")
print("=" * 70)

# Test 1: Initialize Earth Engine
print("\n1. Initializing Earth Engine...")
try:
    ee.Initialize(project='fishsense-480120')
    print("   ✓ Successfully initialized!")
except Exception as e:
    print(f"   ✗ Initialization failed!")
    print(f"   Error: {e}")
    print("\n   TROUBLESHOOTING:")
    print("   → Run: earthengine authenticate")
    print("   → Make sure you selected the FishSense project")
    exit()

# Test 2: Basic operations
print("\n2. Testing basic operations...")
try:
    test_image = ee.Image(1)
    value = test_image.getInfo()
    print(f"   ✓ Basic operations work!")
except Exception as e:
    print(f"   ✗ Operation failed: {e}")
    exit()

# Test 3: Access real satellite data
print("\n3. Testing satellite data access...")
try:
    # Try to access NOAA Sea Surface Temperature
    dataset = ee.ImageCollection('NOAA/CDR/SST_WHOI/V2')
    count = dataset.size().getInfo()
    print(f"   ✓ Can access satellite data!")
    print(f"   ✓ NOAA SST dataset has {count} images")
except Exception as e:
    print(f"   ✗ Data access failed: {e}")
    exit()

# Test 4: Geographic operations
print("\n4. Testing geographic operations...")
try:
    # Create a point in Colombo, Sri Lanka
    colombo = ee.Geometry.Point([79.8612, 6.9271])
    info = colombo.getInfo()
    print(f"   ✓ Geographic operations work!")
    print(f"   ✓ Test location: Colombo, Sri Lanka")
    print(f"   ✓ Coordinates: {info['coordinates']}")
except Exception as e:
    print(f"   ✗ Geographic operations failed: {e}")
    exit()

# Test 5: Access Sri Lankan coastal area
print("\n5. Testing study area access...")
try:
    # Define Sri Lankan coastal waters
    sri_lanka_bbox = ee.Geometry.Rectangle([79.5, 5.9, 81.9, 9.9])
    area = sri_lanka_bbox.area().getInfo()
    print(f"   ✓ Study area defined successfully!")
    print(f"   ✓ Study area size: {area/1e9:.2f} km²")
except Exception as e:
    print(f"   ✗ Study area test failed: {e}")

# All tests passed!
print("\n" + "=" * 70)
print("✓✓✓ ALL TESTS PASSED!")
print("Google Earth Engine is ready for FishSense!")
print("=" * 70)
print("\nYou can now:")
print("  → Access satellite data")
print("  → Process oceanographic parameters")
print("  → Build your machine learning models")
print("=" * 70)