# Data Directory

This directory should contain the datasets used in the analysis.

## Required Datasets

### 1. Microclimate Sensors Data
- **File**: `microclimate-sensors-data.csv`
- **Source**: [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/explore/dataset/microclimate-sensors-data/)
- **Records**: 314,725 observations from IoT sensors
- **Features**: 16 columns including wind, temperature, humidity, PM2.5, PM10, noise

### 2. Obesity Dataset
- **File**: `ObesityDataSet_raw_and_data_sinthetic.csv`
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/)
- **Records**: 2,111 observations
- **Features**: 16 features on eating habits and physical condition
- **Target**: NObeyesdad (7 obesity level classes)

### 3. Gene Expression Cancer RNA-Seq
- **File**: `TCGA-PANCAN-HiSeq-801x20531/data.csv`
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/)
- **Records**: 801 cancer samples (542 used after cleaning)
- **Features**: 20,531 gene expression features
- **Cancer Types**: BRCA, KIRC, COAD, LUAD, PRAD

## Download Instructions

### Microclimate Data
```bash
# Download from City of Melbourne Open Data Portal
wget "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/microclimate-sensors-data/exports/csv" -O microclimate-sensors-data.csv
```

### Obesity Dataset
```bash
# Download from UCI Repository
wget https://archive.ics.uci.edu/static/public/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip
unzip estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip
```

### Gene Expression Data
```bash
# Download from UCI Repository
wget https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip
unzip gene+expression+cancer+rna+seq.zip
tar -xzf TCGA-PANCAN-HiSeq-801x20531.tar.gz
```

## Data Schema

### Microclimate Features
| Column | Type | Description |
|--------|------|-------------|
| Device_id | string | Sensor identifier |
| Time | datetime | Timestamp |
| SensorLocation | string | Physical location |
| LatLong | string | GPS coordinates |
| MinimumWindDirection | float | Min wind direction (degrees) |
| AverageWindDirection | float | Avg wind direction (degrees) |
| MaximumWindDirection | float | Max wind direction (degrees) |
| MinimumWindSpeed | float | Min wind speed (m/s) |
| AverageWindSpeed | float | Avg wind speed (m/s) |
| GustWindSpeed | float | Gust wind speed (m/s) |
| AirTemperature | float | Temperature (°C) |
| RelativeHumidity | float | Humidity (%) |
| AtmosphericPressure | float | Pressure (hPa) |
| PM25 | float | PM2.5 concentration |
| PM10 | float | PM10 concentration |
| Noise | float | Noise level (dB) |

### Obesity Features
| Column | Type | Description |
|--------|------|-------------|
| Gender | string | Male/Female |
| Age | float | Age in years |
| Height | float | Height in meters |
| Weight | float | Weight in kg |
| family_history_with_overweight | string | yes/no |
| FAVC | string | Frequent high caloric food |
| FCVC | float | Vegetable consumption frequency |
| NCP | float | Number of main meals |
| CAEC | string | Food consumption between meals |
| SMOKE | string | Smoking habit |
| CH2O | float | Daily water consumption |
| SCC | string | Calorie consumption monitoring |
| FAF | float | Physical activity frequency |
| TUE | float | Technology use time |
| CALC | string | Alcohol consumption |
| MTRANS | string | Transportation mode |
| NObeyesdad | string | Obesity level (target) |

## Notes

- Datasets are not included in this repository due to size
- Please download from official sources before running the analysis
- Ensure data files are placed in this directory with correct names
