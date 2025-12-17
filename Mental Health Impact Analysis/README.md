# Mental Health Impact Analysis - COVID-19 Effects Across US States

## Overview
A comprehensive data analysis project examining the relationship between COVID-19 pandemic impacts and mental health outcomes across United States, with focus on state-level variations, demographic differences, and temporal patterns.

## Architecture
- **Data Integration**: Multi-source dataset combination (COVID cases, population estimates, mental health surveys)
- **Temporal Analysis**: Monthly COVID-19 positivity rates and mental health trends
- **Demographic Segmentation**: Gender and age-based mental health impact analysis
- **Geospatial Patterns**: State-level variations in pandemic effects and mental health responses
- **Network Analysis**: Inter-state relationships and mental health correlation patterns

## Tech Stack
- **Python** - Core analysis language
- **pandas** - Data manipulation and integration
- **matplotlib/seaborn** - Data visualization and plotting
- **Jupyter Notebook** - Interactive analysis environment
- **CSV datasets** - COVID-19 cases, population estimates, mental health surveys

## Dataset Sources
- **COVID-19 Data**: State-level daily case/death counts (New York Times dataset)
- **Population Estimates**: US Census Bureau population data by state
- **Mental Health Surveys**: Gender and age-specific depression/anxiety data
- **Geographic Data**: State FIPS codes and regional classifications

## How to Run

1. **Install dependencies**:
   ```bash
   pip install pandas matplotlib seaborn jupyter
   ```

2. **Run analysis notebooks**:
   ```bash
   jupyter notebook
   # Open and run notebooks in Code/ directory in order
   ```

3. **Data processing workflow**:
   - COVID data aggregation by state/month
   - Population estimate integration
   - Positivity rate calculations
   - Mental health demographic analysis
   - Network correlation analysis

## Key Decisions
- **Monthly Aggregation**: COVID data grouped by state and month for temporal analysis
- **Population Normalization**: Positivity rates calculated using average population estimates
- **Multi-dimensional Analysis**: Combined COVID impact with mental health outcomes
- **Demographic Focus**: Separate analysis for gender and age group differences
- **Geospatial Context**: State-level variations highlight regional pandemic responses

## Analysis Components
- **Population Positivity Rates**: Monthly COVID-19 infection rates by state
- **Mental Health by Demographics**: Depression/anxiety patterns by gender and age
- **Temporal Correlations**: Relationship between COVID peaks and mental health declines
- **Network Analysis**: Inter-state mental health pattern correlations
- **Policy Implications**: Evidence-based recommendations for mental health support

## Results
The analysis reveals significant state-level variations in both COVID-19 impacts and mental health responses, with clear demographic differences in vulnerability patterns and recovery trajectories during the pandemic period.
