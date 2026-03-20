import ee
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

# import from gee_utils 
try:
    from gee_utils import initialize_gee, get_study_area
    print("✓ Using gee_utils.py")
    USE_GEE_UTILS = True
except ImportError:
    print("⚠️  gee_utils.py not found, using fallback initialization")
    USE_GEE_UTILS = False
    
    def initialize_gee():
        """Fallback initialization"""
        try:
            ee.Initialize()
            return True
        except Exception:
            return False
    
    def get_study_area():
        """Define study area around Sri Lanka"""
        return ee.Geometry.Rectangle([79.5, 5.5, 82.0, 10.0])

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("FISHSENSE: BULK DATA COLLECTION FOR ML TRAINING")
print("=" * 80)

print("\nInitializing Google Earth Engine...")
if not initialize_gee():
    print("Failed to initialize GEE")
    exit()
print("✓ GEE Initialized!\n")

study_area = get_study_area()

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR 2-FEATURE MODEL
# ============================================================================

# Date range: 3 months of data (March-May 2023 - Dry season for better chlorophyll coverage)
start_date = "2023-03-01"
end_date = "2023-05-31"

# Sample every 7 days (gets us ~13 time points)
sample_interval_days = 7

# Spatial grid: 10x10 = 100 locations
num_lat_points = 10
num_lon_points = 10

print("\n🎯 3-FEATURE MODEL: SST + Chlorophyll-a + Ocean Currents")
print("   Using ESA Ocean Color CCI chlorophyll-a (multi-satellite merged)")

print("CONFIGURATION:")
print(f"  📅 Date range: {start_date} to {end_date}")
print(f"  📍 Spatial grid: {num_lon_points}x{num_lat_points} = {num_lon_points * num_lat_points} locations")
print(f"  ⏱️  Temporal sampling: Every {sample_interval_days} days")
print(f"  🎯 Expected total points: ~{(num_lon_points * num_lat_points * 13)}")
print("=" * 80)

# ============================================================================
# CREATE SAMPLING GRID
# ============================================================================

print("\n📍 Creating spatial sampling grid...")

lons = np.linspace(79.5, 82.0, num_lon_points)
lats = np.linspace(6.0, 9.5, num_lat_points)

sample_points = []
for lon in lons:
    for lat in lats:
        sample_points.append([float(lon), float(lat)])

print(f"✓ Created {len(sample_points)} sampling locations")

# ============================================================================
# CREATE DATE SEQUENCE
# ============================================================================

print("\n📅 Creating temporal sampling sequence...")

start = datetime.strptime(start_date, "%Y-%m-%d")
end = datetime.strptime(end_date, "%Y-%m-%d")

sample_dates = []
current = start
while current <= end:
    sample_dates.append(current.strftime("%Y-%m-%d"))
    current += timedelta(days=sample_interval_days)

print(f"✓ Created {len(sample_dates)} sampling dates")

# ============================================================================
# DATA COLLECTION
# ============================================================================

print("\n🌊 Collecting satellite data...")
print("=" * 80)

all_data = []
total_expected = len(sample_dates) * len(sample_points)
processed = 0

for date_str in sample_dates:
    print(f"\n📅 Processing {date_str}...")
    
    # Create date range for this sample
    date = datetime.strptime(date_str, "%Y-%m-%d")
    date_start = date.strftime("%Y-%m-%d")
    date_end = (date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # ========================================================================
    # 1. GET SEA SURFACE TEMPERATURE
    # ========================================================================
    
    try:
        sst_collection = ee.ImageCollection('NOAA/CDR/OISST/V2_1') \
            .filterDate(date_start, date_end) \
            .filterBounds(study_area)
        
        sst_count = sst_collection.size().getInfo()
        
        if sst_count > 0:
            sst_image = sst_collection.first()
        else:
            print(f"  ⚠️  No SST data for {date_str}, skipping...")
            continue
    
    except Exception as e:
        print(f"  ✗ SST Error: {str(e)[:80]}")
        continue
    
    # ========================================================================
    # 2. GET CHLOROPHYLL-A (MODIS Aqua)
    # ========================================================================
    
    try:
        # Copernicus Ocean Color CCI - Merges multiple satellites for better coverage
        # This dataset combines MODIS, VIIRS, SeaWiFS, MERIS for gap-filling
        # Coverage: 1997-2024
        chlor_collection = ee.ImageCollection('COPERNICUS/MARINE/SATELLITE_OCEAN_COLOR_V6') \
            .filterDate(date_start, date_end) \
            .filterBounds(study_area)
        
        chlor_count = chlor_collection.size().getInfo()
        
        if chlor_count > 0:
            chlor_image = chlor_collection.first()
        else:
            print(f"  ⚠️  No chlorophyll data for {date_str}, using None values...")
            chlor_image = None  # Will result in None values for this date
    
    except Exception as e:
        print(f"  ⚠️  Chlorophyll Error: {str(e)[:80]}, using None values...")
        chlor_image = None
    
    # ========================================================================
    # 3. GET OCEAN CURRENTS
    # ========================================================================
    
    try:
        current_collection = ee.ImageCollection('HYCOM/sea_water_velocity') \
            .filterDate(date_start, date_end)
        
        current_count = current_collection.size().getInfo()
        
        if current_count > 0:
            current_image = current_collection.first()
        else:
            print(f"  ⚠️  No current data for {date_str}, skipping...")
            continue
    
    except Exception as e:
        print(f"  ✗ Current Error: {str(e)[:80]}")
        continue
    
    # ========================================================================
    # 4. SAMPLE AT ALL POINTS
    # ========================================================================
    
    valid_samples = 0
    
    for lon, lat in sample_points:
        try:
            point = ee.Geometry.Point([lon, lat])
            
            # Sample SST
            sst_sample = sst_image.select('sst').sample(
                region=point,
                scale=25000,
                numPixels=1
            ).getInfo()
            
            # Sample Chlorophyll-a (if available)
            chlor_a = None
            if chlor_image is not None:
                try:
                    chlor_sample = chlor_image.select('chlor_a').sample(
                        region=point,
                        scale=4000,  # Ocean Color CCI resolution ~4km
                        numPixels=1
                    ).getInfo()
                    
                    if chlor_sample and chlor_sample['features']:
                        chlor_a = chlor_sample['features'][0]['properties'].get('chlor_a')
                except Exception:
                    chlor_a = None
            
            # Sample Currents
            current_sample = current_image.select(['velocity_u_0', 'velocity_v_0']).sample(
                region=point,
                scale=10000,
                numPixels=1
            ).getInfo()
            
            # Extract values
            sst_raw = None
            current_u = None
            current_v = None
            
            if sst_sample and sst_sample['features']:
                sst_raw = sst_sample['features'][0]['properties'].get('sst')
            
            if current_sample and current_sample['features']:
                props = current_sample['features'][0]['properties']
                current_u = props.get('velocity_u_0')
                current_v = props.get('velocity_v_0')
            
            # Convert units
            sst_celsius = sst_raw / 100.0 if sst_raw is not None else None
            
            if current_u is not None and current_v is not None:
                current_u_ms = current_u / 100.0
                current_v_ms = current_v / 100.0
                current_speed = math.sqrt(current_u_ms**2 + current_v_ms**2)
                current_dir = math.degrees(math.atan2(current_u_ms, current_v_ms))
                if current_dir < 0:
                    current_dir += 360
            else:
                current_u_ms = None
                current_v_ms = None
                current_speed = None
                current_dir = None
            
            # Only add if we have at least SST or current data
            if sst_celsius is not None or current_speed is not None:
                all_data.append({
                    'date': date_str,
                    'longitude': lon,
                    'latitude': lat,
                    'sst_celsius': sst_celsius,
                    'chlor_a_mg_m3': chlor_a,  # Chlorophyll-a concentration
                    'current_u_m_s': current_u_ms,
                    'current_v_m_s': current_v_ms,
                    'current_speed_m_s': current_speed,
                    'current_direction_deg': current_dir
                })
                valid_samples += 1
            
            processed += 1
            
        except Exception as e:
            processed += 1
            continue
    
    print(f"  ✓ Collected {valid_samples}/{len(sample_points)} valid samples")
    print(f"  📊 Progress: {processed}/{total_expected} points ({(processed/total_expected)*100:.1f}%)")

# ============================================================================
# SAVE TO CSV
# ============================================================================

print("\n" + "=" * 80)
print("💾 SAVING DATA...")
print("=" * 80)

if len(all_data) > 0:
    df = pd.DataFrame(all_data)
    
    # Create output filename
    output_file = f"data/fishsense_training_data_{start_date}_to_{end_date}.csv"
    df.to_csv(output_file, index=False)
    
    print("\n✅ SUCCESS!")
    print(f"📁 File saved: {output_file}")
    print(f"📊 Total data points: {len(df)}")
    print("\n📈 DATA SUMMARY:")
    print(f"  • Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  • Unique dates: {df['date'].nunique()}")
    print(f"  • Unique locations: {len(df.groupby(['longitude', 'latitude']))}")
    
    # Data quality stats
    sst_valid = df['sst_celsius'].notna().sum()
    chlor_valid = df['chlor_a_mg_m3'].notna().sum()
    current_valid = df['current_speed_m_s'].notna().sum()
    all_three_valid = ((df['sst_celsius'].notna()) & (df['chlor_a_mg_m3'].notna()) & (df['current_speed_m_s'].notna())).sum()
    
    print("\n🔍 DATA QUALITY:")
    print(f"  • SST available: {sst_valid}/{len(df)} ({(sst_valid/len(df)*100):.1f}%)")
    print(f"  • Chlorophyll-a available: {chlor_valid}/{len(df)} ({(chlor_valid/len(df)*100):.1f}%)")
    print(f"  • Current available: {current_valid}/{len(df)} ({(current_valid/len(df)*100):.1f}%)")
    print(f"  • All 3 features: {all_three_valid}/{len(df)} ({(all_three_valid/len(df)*100):.1f}%)")
    
    if all_three_valid >= 300:
        print(f"\n✅ EXCELLENT! {all_three_valid} complete samples - ready for ML training!")
    elif all_three_valid >= 150:
        print(f"\n✅ GOOD! {all_three_valid} complete samples - sufficient for prototype")
    elif chlor_valid >= 200:
        print(f"\n⚠️  NOTE: {chlor_valid} chlorophyll samples, {all_three_valid} with all features")
        print(f"   Chlorophyll often missing due to clouds - this is normal")
    else:
        print("\n⚠️  WARNING: Only {all_three_valid} complete samples - may need more data")
    
    # Show sample data
    print("\n📋 SAMPLE DATA (first 5 rows):")
    print(df.head().to_string())
    
    # Statistics
    print("\n📊 FEATURE STATISTICS:")
    print(df.describe())
    
else:
    print("\n❌ ERROR: No data collected!")
    print("Check your internet connection and GEE access")

print("\n" + "=" * 80)
print("🎯 NEXT STEPS:")
print("  1. Check the CSV file")
print("  2. Visualize the data (optional)")
print("  3. Build ML models with this data")
print("=" * 80)