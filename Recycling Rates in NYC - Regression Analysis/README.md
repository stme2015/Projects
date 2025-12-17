# NYC Recycling Rates Regression Analysis

## Overview
A comprehensive statistical analysis examining the factors influencing recycling rates across New York City community districts, with particular focus on demographic, economic, and COVID-19 lockdown effects on waste diversion patterns.

## Architecture
- **Data Sources**: DSNY tonnage data, ACS census demographics, SNAP benefits data
- **Spatial Analysis**: Community district-level aggregation across NYC boroughs
- **Temporal Analysis**: Pre/post COVID-19 lockdown comparison
- **Statistical Modeling**: Multiple linear regression with socioeconomic predictors
- **Geospatial Integration**: Borough and community district mapping

## Tech Stack
- **Python** - Core analysis and modeling
- **pandas** - Data manipulation and cleaning
- **statsmodels** - Statistical regression modeling
- **matplotlib/seaborn** - Data visualization
- **geopandas** - Spatial data analysis
- **scipy** - Statistical testing and distributions
- **Jupyter Notebook** - Interactive analysis environment

## Dataset Sources
- **DSNY Monthly Tonnage**: NYC Department of Sanitation waste collection data
- **ACS Census Data**: Demographic and economic indicators by community district
- **SNAP Program Data**: Food assistance benefits as socioeconomic indicator
- **Furman Center**: Neighborhood indicators and housing data
- **NYC Open Data**: Geographic boundaries and administrative divisions

## Analysis Components
- **Waste Diversion Rates**: Recycling vs. refuse collection patterns
- **Demographic Correlations**: Age, race, income effects on recycling behavior
- **Economic Indicators**: Employment sectors and SNAP participation analysis
- **COVID-19 Impact**: Lockdown effects on waste management patterns
- **Spatial Variations**: Borough and community district differences

## How to Run

1. **Install dependencies**:
   ```bash
   pip install pandas statsmodels matplotlib seaborn geopandas scipy
   ```

2. **Run analysis pipeline**:
   ```bash
   jupyter notebook
   # Execute notebooks in order: 00_ACS_NTA_DataClean.ipynb → TargetVariableCreation_Clean.ipynb → modeling notebooks
   ```

3. **Data processing workflow**:
   - Census data cleaning and aggregation
   - DSNY tonnage data processing
   - Target variable creation (diversion rate changes)
   - Model training and evaluation
   - Results visualization

## Key Decisions
- **Community District Level**: Fine-grained spatial analysis for policy relevance
- **Pre/Post Lockdown Comparison**: COVID-19 impact assessment methodology
- **Multi-Model Approach**: Census-only vs. comprehensive feature models
- **SNAP Integration**: Socioeconomic vulnerability indicator inclusion
- **Binned Income Categories**: Handling income distribution for regression

## Modeling Techniques
- **Multiple Linear Regression**: Primary statistical modeling approach
- **Feature Engineering**: Income binning, employment sector categorization
- **Cross-Validation**: Train/test split validation for model robustness
- **Correlation Analysis**: Feature selection and multicollinearity assessment
- **Residual Analysis**: Model diagnostics and assumption checking

## Results Focus
- **Demographic Predictors**: Age, race, and income effects on recycling rates
- **Economic Factors**: Employment sectors and benefit program participation
- **Spatial Patterns**: Borough-level variations in waste management
- **COVID-19 Effects**: Lockdown impact on diversion rate changes
- **Policy Implications**: Targeted interventions for improving recycling rates
