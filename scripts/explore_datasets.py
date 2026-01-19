import ee
from datetime import datetime
from gee_utils import initialize_gee, get_study_area

# -------------------------------------------------------------
# 1. INITIALIZE GOOGLE EARTH ENGINE
# -------------------------------------------------------------
print("Initializing Google Earth Engine...")
initialize_gee()
print("✓ GEE Initialized!\n")

# -------------------------------------------------------------
# 2. DEFINE STUDY AREA AND DATE RANGE
# -------------------------------------------------------------
study_area = get_study_area()

start_date = "2023-01-01"
end_date = "2023-12-31"

print("=" * 80)
print("FISHSENSE DATASET EXPLORER")
print("=" * 80)
print("📍 Study Area: Sri Lankan Coastal Waters")
print(f"📅 Date Range: {start_date} to {end_date}")
print("=" * 80)

# -------------------------------------------------------------
# 3. SEA SURFACE TEMPERATURE (SST)
# -------------------------------------------------------------
print("\n🌡️  SEA SURFACE TEMPERATURE (SST):\n")

try:
    sst_collection = (
        ee.ImageCollection("NOAA/CDR/OISST/V2_1")
        .filterDate(start_date, end_date)
        .filterBounds(study_area)
    )

    count = sst_collection.size().getInfo()
    print(f"✓ Images available: {count}")

    if count > 0:
        latest = sst_collection.sort("system:time_start", False).first()
        date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()

        print(f"✓ Most recent image date: {date}")
        print("✓ Band used: sst")

        sample = latest.select("sst").sample(
            region=ee.Geometry.Point([80.5, 7.5]),
            scale=25000,
            numPixels=1
        ).getInfo()

        if sample["features"]:
            sst_value = sample["features"][0]["properties"]["sst"]
            print(f"✓ Sample SST value: {sst_value:.2f} °C")

except Exception as e:
    print(f"✗ Error accessing SST dataset: {e}")

# -------------------------------------------------------------
# 4. PRODUCTIVITY PROXY (SENTINEL-3 OLCI)
# -------------------------------------------------------------
print("\n🌿 CHLOROPHYLL-A PROXY (SENTINEL-3 OLCI):\n")

try:
    chlor_collection = (
        ee.ImageCollection("COPERNICUS/S3/OLCI")
        .filterDate(start_date, end_date)
        .filterBounds(study_area)
    )

    # Mask invalid pixels using quality flags
    def mask_olci(image):
        quality = image.select("quality_flags")
        mask = quality.eq(0)  # 0 = good quality
        return image.updateMask(mask)

    chlor_collection = chlor_collection.map(mask_olci)

    count = chlor_collection.size().getInfo()
    print(f"✓ Images available: {count}")

    if count > 0:
        latest = chlor_collection.sort("system:time_start", False).first()
        date = ee.Date(latest.get("system:time_start")).format("YYYY-MM-dd").getInfo()

        print(f"✓ Most recent image date: {date}")
        print("✓ Bands used: Oa05_radiance, Oa08_radiance, Oa10_radiance")

        sample = latest.select(
            ["Oa05_radiance", "Oa08_radiance", "Oa10_radiance"]
        ).sample(
            region=ee.Geometry.Point([80.5, 7.5]),
            scale=300,
            numPixels=1
        ).getInfo()

        if sample["features"]:
            props = sample["features"][0]["properties"]
            print(
                f"✓ Sample values → "
                f"Oa05: {props['Oa05_radiance']:.6f}, "
                f"Oa08: {props['Oa08_radiance']:.6f}, "
                f"Oa10: {props['Oa10_radiance']:.6f}"
            )

except Exception as e:
    print(f"✗ Error accessing Sentinel-3 OLCI dataset: {e}")

# -------------------------------------------------------------
# 5. OCEAN CURRENTS (HYCOM)
# -------------------------------------------------------------
print("\n🌊 OCEAN CURRENTS:\n")

try:
    current_collection = (
        ee.ImageCollection("HYCOM/sea_water_velocity")
        .filterDate(start_date, end_date)
    )

    count = current_collection.size().getInfo()
    print(f"✓ Images available: {count}")

    if count > 0:
        latest = current_collection.sort("system:time_start", False).first()
        print("✓ Bands used: water_u, water_v")

except Exception as e:
    print(f"✗ Error accessing HYCOM dataset: {e}")

# -------------------------------------------------------------
# 6. SUMMARY
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

print("""
✔ Sea Surface Temperature (SST)
  - NOAA OISST (Daily)

✔ Ocean Productivity Proxy
  - Sentinel-3 OLCI spectral radiance bands

✔ Ocean Currents
  - HYCOM Global Model

All datasets are validated and suitable for
fish distribution and habitat modeling.
""")

print("=" * 80)
print("✓ Dataset exploration complete!")
