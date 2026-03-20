# FishSense - Known Limitations and Considerations

## 🔴 Known Limitations

### 1. **Data Availability and Latency**
- **Satellite Data Delay**: Real-time oceanographic data may have 1-3 day latency depending on satellite overpass schedules and processing time
- **Cloud Cover Impact**: Optical sensors (SST, Chlorophyll-a) cannot capture data through clouds, leading to data gaps
- **Data Resolution**: Different datasets have varying spatial resolutions (4-28 km), which may miss fine-scale oceanographic features

### 2. **Geographic Coverage**
- **Primary Focus**: Model is trained specifically for Sri Lankan coastal waters (79.5°E-81.9°E, 5.9°N-9.9°N)
- **Limited Generalization**: Predictions outside this region may be unreliable
- **On-Land Detection**: System warns but does not prevent predictions for land coordinates

### 3. **Model Constraints**
- **Training Data**: Model accuracy depends on the quality and quantity of historical fishing ground data
- **Environmental Variables**: Currently uses limited oceanographic parameters (SST, currents, chlorophyll-a)
- **Missing Factors**: Does not account for:
  - Fish migration patterns
  - Seasonal variations
  - Weather conditions (wind, waves)
  - Fishing pressure in areas
  - Marine protected areas

### 4. **Technical Dependencies**
- **Google Earth Engine**: Requires active GEE account and authentication
- **Internet Connection**: Live data fetching requires stable internet
- **System Requirements**: Recommended 16GB RAM for optimal performance
- **Browser Compatibility**: Best viewed on modern browsers (Chrome, Edge, Firefox)

### 5. **User Interface**
- **CSS Caching Issues**: Some styling changes may require hard browser refresh (Ctrl+F5)
- **Mobile Responsiveness**: Dashboard is optimized for desktop/laptop screens
- **Input Validation**: Limited validation for extreme oceanographic values

### 6. **Prediction Accuracy**
- **Model Type**: Random Forest classifier provides probability-based predictions, not guarantees
- **No Real-Time Validation**: Predictions are not verified against actual fishing yield data
- **Environmental Variability**: Ocean conditions can change rapidly, affecting prediction reliability

---

## ⚠️ Important Considerations for Users

### For Fishermen
- **Use as Guidance Only**: Predictions should be combined with local knowledge and experience
- **Safety First**: Always check weather forecasts and sea conditions before going out
- **Verify Regulations**: Ensure fishing is allowed in predicted zones (check marine protected areas)

### For Researchers
- **Data Quality**: Verify satellite data quality metrics before using predictions
- **Model Validation**: Results should be validated against ground truth data
- **Continuous Improvement**: Model should be retrained periodically with new data

### For Policymakers
- **Decision Support Tool**: Use as one input among many for fisheries management
- **Not Replacement for Surveys**: Should complement, not replace, traditional fisheries assessment methods
- **Sustainability Considerations**: Monitor predicted zones to prevent overfishing

---

## 🔧 Future Improvements

### Planned Enhancements
1. **Expand Features**: Include bathymetry, moon phase, tidal data
2. **Temporal Predictions**: Add forecast capabilities (3-7 day predictions)
3. **Multi-Species Models**: Develop species-specific prediction models
4. **Mobile App**: Create mobile application for fishermen
5. **Feedback Loop**: Implement system to collect actual fishing data for model refinement
6. **Multi-Language Support**: Add Sinhala and Tamil language options
7. **Offline Mode**: Cache data for limited offline functionality

### Technical Improvements
- Optimize data processing for faster predictions
- Implement automated model retraining pipeline
- Add comprehensive error handling and logging
- Develop API for third-party integrations

---

## 📞 Support and Feedback

If you encounter issues or have suggestions, please contact:
- **Developer**: Nisandu Senanayake (w1871483)
- **Supervisor**: Kanishka Hewageegana
- **Institution**: University of Westminster

---

## 🛡️ Disclaimer

**This system is a prototype developed for academic purposes.**

- Predictions are based on historical patterns and may not reflect current fish abundance
- Users should always exercise caution and use their own judgment
- The system does not guarantee fishing success or safety at sea
- The developers are not liable for any losses or incidents resulting from the use of this system

**Always prioritize safety, comply with local fishing regulations, and practice sustainable fishing.**
