# QF5214_G6_PORJECT
## Introduction
In commodity futures trading, the monitoring and prediction of crop growing conditions is crucial. Environmental factors such as precipitation, temperature, and solar radiation have a signiﬁcant impact on crop yields and quality. For instance, research suggests that roughly 20% of the increase in Iowa corn yields over recent decades can be attributed to enhanced solar radiation (solar brightening). Fluctuations in these variables alter the fundamental supply-demand balance of agricultural commodities, making weather one of the key determinants of commodity prices. Unpredictable weather often leads to volatile crop prices, as the agricultural sector is highly vulnerable to climate variability. Likewise, extreme weather events can cause surges in crop insurance claims: in severe drought years, insurance payouts have exceeded premiums, and when disasters aﬀect large swaths of farmland, yield losses drive up commodity prices and result in more frequent and costly insurance indemnities. Therefore, reliably monitoring and forecasting these environmental variables is of paramount importance for anticipating future market trends and informing agricultural insurance compensation strategies.  

NASA POWER project addresses this need by providing high-resolution meteorological and solar radiation data with global coverage. The POWER project aggregates NASAs Earth observation data on surface solar irradiance and various meteorological parameters, oﬀering free and convenient access to these data for users worldwide. NASA POWER datasets have been used to support applications in renewable energy, building energy eﬃciency, and agricultural analyses, making them valuable for evaluating crop-growing environments. By leveraging this authoritative data source, researchers and practitioners can more accurately characterize ﬁeld-level climate conditions, supplying reliable inputs for commodity market analysis and agricultural risk modelling.  

This study leverages daily meteorological and solar radiation data from the continental United States (excluding Hawaii and Alaska), sourced from NASA POWER between January 1, 2018, and December 31, 2024, to establish a multi-dimensional environmental database and develop time-series forecasting models. To dynamically predict trends in critical environmental parameters, we implemented a multi-task Long Short-Term Memory (LSTM) architecture for joint modelling. Building on these predictions, we innovatively developed an "Environmental Risk Index" to quantitatively characterize integrated environmental impacts. The index formulation adopts a dual-strategy approach, independently calculating results using both the classical Entropy Weight Method and an AI-assisted weighting scheme powered by OpenAI’s interface. Notably, the system incorporates a spatiotemporal query module that enables users to retrieve real-time index values for speciﬁed geographic coordinates and timeframes through an interactive interface.

## STEP1: US_GETALL_DAILY.py
The `US_GETALL_DAILY.py` script is designed to connect to the NASA POWER database API for automated data retrieval. This script allows users to specify the target variables, which will then be sequentially fetched from the NASA POWER API. After successful data extraction, the retrieved data for each variable will be saved locally as CSV files for further analysis or processing.

![US_GETALL_DAILY Example](Picture/GET_DATA_PICTURE1.png)

## STEP2: DATA_QUALITY_CHECK.py
The `DATA_QUALITY_CHECK.py` script is used to check the missing values and abnormal values within each variable of the dataset. It helps ensure data completeness and quality before further analysis.

After completing the data retrieval process for all variables, running this script will generate:

- A summary CSV file containing missing value statistics for all variables: `Data_Quality_Summary.csv`
- A visualization showing the distribution of abnormal data ratio across all files: `Bad_Ratio_Distribution.png`

### Output Example

#### 1. Data Quality Summary Table
This file records the number and proportion of missing and abnormal values for each variable.

![Example of Data Quality Summary](Picture/Data_Quality_Summary.png)

#### 2. Bad Ratio Distribution Plot
This figure shows the distribution of abnormal data ratio across all generated CSV files.

![Bad Ratio Distribution](Picture/Bad_Ratio_Distribution.png)

## STEP3: DATA_CLEAN.py

The `DATA_CLEAN.py` script is designed to perform spatial interpolation to fill missing values in the selected variable datasets. 

By running this script, each filtered variable's CSV file will be processed individually. Missing values will be filled using spatial interpolation techniques, and the completed datasets will be saved into a new output directory for further usage.

In addition, the script will generate a visualization showing the repair ratio (i.e., the proportion of missing values that have been filled) for each variable.

### Output Results

- Completed CSV files for each variable (after missing value interpolation)
- A visualization showing the repair ratio of each variable: `Repair_Ratio_Per_Variable.png`

#### Example of Repair Ratio Distribution

![Repair Ratio Per Variable](Picture/Repair_Ratio_Per_Variable.png)


## Conclusion
This study demonstrates the advanced integration and application of database systems in the field of environmental and natural resource analysis. By systematically collecting and processing datasets related to weather and solar radiation, and incorporating state-of-the-art data processing techniques, the research significantly enhances the efficiency of environmental resource forecasting and visualization. Importantly, it also provides decision-support functions tailored to financial practitioners who may not possess a background in environmental or geosciences. These efforts contribute to a more nuanced understanding of the relationship between meteorological indicators and environmental resources, thereby improving insight into sectors with high environmental dependency, such as agricultural insurance and commodity trading.
