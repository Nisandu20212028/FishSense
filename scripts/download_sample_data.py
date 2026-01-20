"""
FishSense: Download Sample Satellite Data (CORRECTED VERSION)
Downloads samples of SST, Chlorophyll proxy, and Current data
FIXES:
1. Relaxed chlorophyll quality mask
2. Fixed ocean current unit conversion (cm/s to m/s)
3. Expanded sampling grid
4. Better error handling
"""

import ee
import json
import os
import math
import numpy as np
from datetime import datetime
from gee_utils import initialize_gee, get_study_area

# Initialize Earth Engine
print("Initializing Google Earth Engine...")
if not initialize_gee():
    print("Failed to initialize GEE")
    exit()

print("✓ GEE Initialized!\n")

# Create data directory if it doesn't exist
data_dir = "data/samples"
os.makedirs(data_dir, exist_ok=True)

# Get study area
study_area = get_study_area()

# Use 2023 data (we know it's available)
start_date = "2023-11-01"
end_date = "2023-11-30"

print("=" * 70)
print("DOWNLOADING SAMPLE DATA (CORRECTED VERSION)")
print("=" * 70)
print(f"📍 Study Area: Sri Lankan Coastal Waters")
print(f"📅 Date Range: {start_date} to {end_date}")
print(f"💾 Save Location: {data_dir}/")
print("=" * 70)

# ============================================================================
# EXPANDED SAMPLE GRID - 25 points (5x5 grid)
# ============================================================================

print("\n📍 Creating sampling grid...")

# Create a 5x5 grid across Sri Lankan waters
lons = np.linspace(79.5, 82.0, 5)  # 5 longitude points
lats = np.linspace(6.0, 9.5, 5)    # 5 latitude points

sample_points = []
for lon in lons:
    for lat in lats:
        sample_points.append([float(lon), float(lat)])

print(f"✓ Created grid with {len(sample_points)} sampling points")

# ============================================================================
# Function to sample data at multiple points
# ============================================================================

def sample_at_points(image, points, bands, scale):
    """
    Sample an image at multiple points
    """
    results = []
    successful_samples = 0

    for lon, lat in points:
        try:
            point = ee.Geometry.Point([lon, lat])

            sample = image.select(bands).sample(
                region=point,
                scale=scale,
                numPixels=1
            ).getInfo()

            if sample and sample['features']:
                props = sample['features'][0]['properties']
                values = {band: props.get(band) for band in bands}
                
                # Check if we got valid data
                if any(v is not None for v in values.values()):
                    successful_samples += 1
            else:
                values = {band: None for band in bands}

            results.append({
                'longitude': lon,
                'latitude': lat,
                'values': values
            })

        except Exception as e:
            print(f"  ⚠ Error sampling at [{lon:.2f}, {lat:.2f}]: {str(e)[:80]}")
            results.append({
                'longitude': lon,
                'latitude': lat,
                'values': {band: None for band in bands}
            })

    print(f"  ✓ Successfully sampled {successful_samples}/{len(points)} points")
    return results

# ============================================================================
# 1. SEA SURFACE TEMPERATURE (NOAA OISST)
# ============================================================================

print("\n🌡️  Downloading Sea Surface Temperature data...")

try:
    sst_collection = ee.ImageCollection('NOAA/CDR/OISST/V2_1') \
        .filterDate(start_date, end_date) \
        .filterBounds(study_area)

    count = sst_collection.size().getInfo()

    if count > 0:
        sst_image = sst_collection.sort('system:time_start', False).first()
        date_millis = sst_image.get('system:time_start').getInfo()
        image_date = datetime.fromtimestamp(date_millis / 1000).strftime('%Y-%m-%d')

        print(f"  ✓ Found {count} images")
        print(f"  ✓ Using image from: {image_date}")

        sst_data = sample_at_points(
            sst_image,
            sample_points,
            ['sst'],
            25000
        )

        # FIX: Convert SST from Celsius * 100 to Celsius
        for s in sst_data:
            raw = s['values']['sst']
            if raw is not None:
                s['values']['sst_celsius'] = round(raw / 100.0, 2)
            else:
                s['values']['sst_celsius'] = None

        output_file = f"{data_dir}/sst_sample_{image_date}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'date': image_date,
                'dataset': 'NOAA/CDR/OISST/V2_1',
                'unit': 'degrees Celsius',
                'samples': sst_data
            }, f, indent=2)

        print(f"  ✓ Data saved to: {output_file}")

    else:
        print("  ✗ No SST images found for this date range")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# 2. CHLOROPHYLL PROXY (SENTINEL-3 OLCI) - RELAXED QUALITY MASK
# ============================================================================

print("\n🌿 Downloading Chlorophyll proxy data (Sentinel-3 OLCI)...")

try:
    chlor_collection = ee.ImageCollection('COPERNICUS/S3/OLCI') \
        .filterDate(start_date, end_date) \
        .filterBounds(study_area)

    # FIX: More permissive quality mask (0-3 instead of 0-1)
    def mask_olci(image):
        quality = image.select('quality_flags')
        # Accept quality flags 0-3 (was 0-1 before)
        mask = quality.lte(3)
        return image.updateMask(mask)

    chlor_collection = chlor_collection.map(mask_olci)

    count = chlor_collection.size().getInfo()
    print(f"  ✓ Found {count} images (after quality filtering)")

    if count > 0:
        chlor_image = chlor_collection.sort('system:time_start', False).first()
        date_millis = chlor_image.get('system:time_start').getInfo()
        image_date = datetime.fromtimestamp(date_millis / 1000).strftime('%Y-%m-%d')

        print(f"  ✓ Using image from: {image_date}")

        chlor_data = sample_at_points(
            chlor_image,
            sample_points,
            ['Oa05_radiance', 'Oa08_radiance', 'Oa10_radiance'],
            300
        )

        # Calculate a simple chlorophyll index (normalized difference)
        for s in chlor_data:
            oa05 = s['values']['Oa05_radiance']
            oa08 = s['values']['Oa08_radiance']
            
            if oa05 is not None and oa08 is not None and oa08 != 0:
                # Simple ratio as chlorophyll proxy
                chlor_index = (oa08 - oa05) / (oa08 + oa05)
                s['values']['chlorophyll_index'] = round(chlor_index, 4)
            else:
                s['values']['chlorophyll_index'] = None

        output_file = f"{data_dir}/chlorophyll_proxy_{image_date}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'date': image_date,
                'dataset': 'COPERNICUS/S3/OLCI',
                'unit': 'spectral radiance (chlorophyll proxy)',
                'note': 'chlorophyll_index = (Oa08-Oa05)/(Oa08+Oa05)',
                'samples': chlor_data
            }, f, indent=2)

        print(f"  ✓ Data saved to: {output_file}")

    else:
        print("  ✗ No chlorophyll images found after quality filtering")
        print("  ℹ️  This is common due to cloud cover - try a different date range")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# 3. OCEAN CURRENTS (HYCOM — SURFACE 0 m) - FIXED UNITS
# ============================================================================

print("\n🌊 Downloading Ocean Current data...")

try:
    current_collection = ee.ImageCollection('HYCOM/sea_water_velocity') \
        .filterDate(start_date, end_date)

    count = current_collection.size().getInfo()

    if count > 0:
        current_image = current_collection.sort('system:time_start', False).first()
        date_millis = current_image.get('system:time_start').getInfo()
        image_date = datetime.fromtimestamp(date_millis / 1000).strftime('%Y-%m-%d')

        print(f"  ✓ Found {count} images")
        print(f"  ✓ Using image from: {image_date}")

        # Use SURFACE current components (0 m depth)
        u_band = 'velocity_u_0'
        v_band = 'velocity_v_0'

        current_data = sample_at_points(
            current_image,
            sample_points,
            [u_band, v_band],
            10000
        )

        # FIX: Convert from cm/s to m/s and compute magnitude
        for s in current_data:
            u_raw = s['values'][u_band]  # in cm/s
            v_raw = s['values'][v_band]  # in cm/s
            
            if u_raw is not None and v_raw is not None:
                # Convert cm/s to m/s
                u_ms = u_raw / 100.0
                v_ms = v_raw / 100.0
                
                # Calculate speed magnitude
                speed = math.sqrt(u_ms**2 + v_ms**2)
                
                # Calculate direction (degrees from north)
                direction = math.degrees(math.atan2(u_ms, v_ms))
                if direction < 0:
                    direction += 360
                
                s['values']['velocity_u_m_s'] = round(u_ms, 3)
                s['values']['velocity_v_m_s'] = round(v_ms, 3)
                s['values']['current_speed_m_s'] = round(speed, 3)
                s['values']['current_direction_deg'] = round(direction, 1)
            else:
                s['values']['velocity_u_m_s'] = None
                s['values']['velocity_v_m_s'] = None
                s['values']['current_speed_m_s'] = None
                s['values']['current_direction_deg'] = None

        output_file = f"{data_dir}/currents_sample_{image_date}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'date': image_date,
                'dataset': 'HYCOM/sea_water_velocity',
                'unit': 'm/s (converted from cm/s)',
                'note': 'Surface currents at 0m depth. Typical range: 0.1-2.5 m/s',
                'samples': current_data
            }, f, indent=2)

        print(f"  ✓ Data saved to: {output_file}")
        
        # Calculate statistics
        speeds = [s['values']['current_speed_m_s'] for s in current_data 
                 if s['values']['current_speed_m_s'] is not None]
        if speeds:
            print(f"  ℹ️  Current speed range: {min(speeds):.3f} - {max(speeds):.3f} m/s")
            print(f"  ℹ️  Average speed: {sum(speeds)/len(speeds):.3f} m/s")

    else:
        print("  ✗ No current data found for this date range")

except Exception as e:
    print(f"  ✗ Error: {e}")

# ============================================================================
# SUMMARY & VALIDATION
# ============================================================================

print("\n" + "=" * 70)
print("✅ DOWNLOAD COMPLETE!")
print("=" * 70)

# Validate data quality
print("\n📊 DATA QUALITY CHECK:")

def count_valid_samples(filename):
    if not os.path.exists(filename):
        return 0, 0
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    total = len(data['samples'])
    valid = sum(1 for s in data['samples'] 
                if any(v is not None for v in s['values'].values()))
    
    return valid, total

# Check each dataset
for dtype, pattern in [('SST', 'sst_sample_*.json'),
                       ('Chlorophyll', 'chlorophyll_proxy_*.json'),
                       ('Currents', 'currents_sample_*.json')]:
    
    import glob
    files = glob.glob(f"{data_dir}/{pattern}")
    
    if files:
        valid, total = count_valid_samples(files[0])
        percentage = (valid/total)*100 if total > 0 else 0
        
        status = "✅" if percentage >= 50 else "⚠️" if percentage >= 25 else "❌"
        print(f"  {status} {dtype}: {valid}/{total} valid points ({percentage:.1f}%)")
    else:
        print(f"  ❌ {dtype}: No data file found")

print("\n💡 Next Steps:")
print("   1. Check the data quality percentages above")
print("   2. If < 50% valid data, try a different date range")
print("   3. If data looks good, proceed to visualization")
print("   4. For ML training, you'll need 500+ total data points")

print("\n📁 All data saved to: {}/".format(data_dir))
print("=" * 70)