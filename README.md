# Machine-Learning-Techniques-to-Forecast-Energy-Consumption-Indian-Context

**India is the third-largest global energy consumer, with demand increasing due to industrialization, urbanization, and population growth. The country still depends heavily on fossil fuels, even though renewable energy adoption is rising. Variations in energy use across residential, commercial, and industrial sectors along with peak demand periods often cause supply shortages and blackouts.**

**Accurate forecasting of energy consumption is crucial for:**
Ensuring energy security
Optimizing grid management
Supporting sustainable development
Research Objective
The project applies Machine Learning (ML) and Time Series Forecasting techniques to predict India’s future energy consumption. By analyzing historical state-wise data, the research identifies patterns, peak demand variations, and key drivers of energy use.

**Dataset**
Source: POSOCO weekly energy reports (via Kaggle)
Period: Jan 2019 – May 2020
Entries: 503 rows (daily records)
Columns: 33 states/union territories + date
Target Variable: Energy consumption (Mega Units, MU)

**Methodology**
Data Preprocessing – Cleaning, handling missing values, formatting time series.
Exploratory Data Analysis (EDA) – Correlation heatmaps, histograms, and consumption patterns.
Model Building – Implemented three algorithms:
ARIMA – Captures time-based trends and seasonality.
Support Vector Machine (SVM) – Handles complex, high-dimensional patterns.
Random Forest Regressor – Ensemble method capturing non-linear relationships.
Model Evaluation – Compared performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

**Results**
Random Forest Regressor outperformed other models:
MSE = 846.65
MAE = 15.57
R² = 0.80

ARIMA showed moderate accuracy (R² = 0.60).
SVM performed worst (R² = 0.43).

**Key Insights**
Random Forest best captures regional and sectoral variations in energy use.
Predictive models can help policy makers and energy providers reduce disruptions, plan for peak demand, and integrate renewable sources effectively.

**Conclusion**
Machine Learning techniques, especially Random Forest, provide reliable forecasts for energy consumption in India. This research shows the potential of AI-driven predictive models in ensuring sustainable, secure, and efficient energy management.
