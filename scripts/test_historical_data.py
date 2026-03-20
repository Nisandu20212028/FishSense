"""
Test real-time satellite data fetching with historical dates
Testing with 2023 data when we know satellite coverage existed
"""

import ee
import numpy as np
from datetime import datetime

# Initialize GEE
print("Initializing Google Earth Engine...")
ee.Initialize(project='fishsense-480120')
print("✅ GEE initialized\n")

# Test location (west of Sri Lanka)
lon = 79.3
lat = 7.0

print(f"Testing location: Lon {lon}, Lat {lat}")
print("=" * 60)

# Test with historical date (2023) when training data was collected
start_str = '2023-11-01'
end_str = '2023-11-30'

print(f"Date range: {start_str} to {end_str} (Historical - Nov 2023)\n")

# Create point geometry
point = ee.Geometry.Point([lon, lat])

# Test MODIS Aqua SST
print("1. Testing MODIS Aqua SST...")
try:
    sst_collection = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI') \
        .filterDate(start_str, end_str) \
        .filterBounds(point) \
        .select('sst')
    
    sst_count = sst_collection.size().getInfo()
    print(f"   Found {sst_count} images")
    
    if sst_count > 0:
        sst_image = sst_collection.sort('system:time_start', False).first()
        
        # Sample SST
        sst_sample = sst_image.select('sst').sample(
            region=point,
            scale=4000,
            numPixels=1
        ).getInfo()
        
        if sst_sample and sst_sample['features'] and len(sst_sample['features']) > 0:
            sst_celsius = sst_sample['features'][0]['properties'].get('sst')
            timestamp_ms = sst_image.get('system:time_start').getInfo()
            data_date = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')
            
            print(f"   ✅ SST: {sst_celsius:.2f}°C")
            print(f"   ✅ Date: {data_date}")
        else:
            print("   ⚠️ No data at sample point")
    else:
        print("   ⚠️ No images in collection")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test HYCOM Ocean Currents
print("2. Testing HYCOM Ocean Currents...")
try:
    current_collection = ee.ImageCollection('HYCOM/sea_water_velocity') \
        .filterDate(start_str, end_str) \
        .filterBounds(point)
    
    current_count = current_collection.size().getInfo()
    print(f"   Found {current_count} images")
    
    if current_count > 0:
        current_image = current_collection.sort('system:time_start', False).first()
        
        # Sample Currents
        current_sample = current_image.select(['velocity_u_0', 'velocity_v_0']).sample(
            region=point,
            scale=10000,
            numPixels=1
        ).getInfo()
        
        if current_sample and current_sample['features'] and len(current_sample['features']) > 0:
            u_cm_s = current_sample['features'][0]['properties'].get('velocity_u_0')
            v_cm_s = current_sample['features'][0]['properties'].get('velocity_v_0')
            
            if u_cm_s is not None and v_cm_s is not None:
                u_m_s = u_cm_s / 100.0
                v_m_s = v_cm_s / 100.0
                current_speed = np.sqrt(u_m_s**2 + v_m_s**2)
                
                print(f"   ✅ Current Speed: {current_speed:.2f} m/s")
                print(f"   ✅ U-component: {u_m_s:.2f} m/s")
                print(f"   ✅ V-component: {v_m_s:.2f} m/s")
            else:
                print("   ⚠️ Current values are null")
        else:
            print("   ⚠️ No data at sample point")
    else:
        print("   ⚠️ No images in collection")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()
print("=" * 60)
print("Test complete!")
print("\nConclusion:")
print("If historical data (2023) works but current data (2026) doesn't,")
print("it means the datasets may not have data for future dates yet.")
