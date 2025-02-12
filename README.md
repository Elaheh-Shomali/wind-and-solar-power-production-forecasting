# Renewable Energy Production Forecast

## Project Overview

This project focuses on forecasting renewable energy production—specifically wind and solar energy—using advanced time series forecasting and machine learning techniques. As the global energy landscape shifts toward cleaner sources, accurate energy predictions become crucial for grid management, energy trading, and policy-making. The project not only builds robust predictive models but also delivers an interactive Streamlit application that allows users to forecast energy production for a given date and hour.

## Methodology

The project’s workflow involves several key steps:

- **Data Preparation:**  
  - Splitting the available dataset into training and validation sets.
  - Applying windowing and slicing techniques to create sliding windows of past observations.
  - Batching the windows for efficient model training.

- **Model Building:**  
  - Defining neural network architectures (e.g., LSTM, CNN, or Transformer models) tailored for time series forecasting.
  - Compiling the model with a suitable loss function and optimizer.

- **Model Training and Evaluation:**  
  - Feeding batched time series data into the model for training.
  - Generating predictions on unseen test data.
  - Evaluating model performance using metrics such as Mean Absolute Percentage Error (MAPE).

- **Future Data Forecasting:**  
  - Applying the trained model to forecast energy production for the next 10 days.
  
- **Streamlit Application:**  
  - Developing an interactive interface that allows users to input a specific date and hour to retrieve energy production forecasts.

## Tools and Techniques

- **Programming Language:** Python
- **Frameworks and Libraries:**  
  - TensorFlow for time series forecasting.
  - Streamlit for building the interactive prediction app.
  - Jupyter Notebooks for data exploration, learning, and experimentation.
- **Evaluation Metrics:**  
  - Mean Absolute Percentage Error (MAPE), among others.

## Key Learnings

- The importance of proper data preparation, especially windowing and batching for time series data.
- Insights into selecting and tuning deep learning architectures (LSTM, CNN, Transformer) for forecasting tasks.
- Practical challenges in handling renewable energy data and translating technical results into actionable insights.
- Effective integration of predictive modeling with a user-friendly Streamlit app, enabling non-technical users to benefit from advanced forecasting.

## Challenges Overcome

- **Model Selection and Tuning:** Experimenting with various architectures to find the best fit for forecasting accuracy.
- 

## Repository Structure

- **Streamlit-prediction-app.py:** Python file for running the Streamlit application.
- **requirements.txt:** Contains the dependencies and prerequisites for installation.
- **models:** Contains saved model files and checkpoints.
- **data:** Contains processed data files.
- **Notebooks:** Contains Jupyter Notebooks for learning and experimentation. This folder includes:
   - **univariate_method.ipynb:** Notebooks for univariate forecasting experiments.
   - **multivariate_method.ipynb:** Notebooks for multivariate forecasting experiments.
- **Output:** include a presentation of the key findings from the analysis.


## Conclusion

This project demonstrates a comprehensive approach to forecasting renewable energy production, combining advanced machine learning techniques with practical deployment strategies. By accurately predicting wind and solar energy outputs, the solution supports energy planning and decision-making processes critical for the global energy transition. The integration of an interactive Streamlit app further enhances the accessibility and practical application of the forecasts, making this project a valuable resource for stakeholders looking to embrace renewable energy solutions.
