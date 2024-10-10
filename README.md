# Telepresent Nature's Regressive Neural Network for Tracking Deer

## About the Project

This project analyzes two regressive neural networks (a Future-Prediction Regressive model and a Past-Prediction Regressive model) that predict a deer’s longitudinal and latitudinal positions using values for a month, day, and temperature. The Future-Prediction Regressive model is trained to make predictions into the future. The Past-Prediction Regressive model is trained to make predictions into the past. The repository also possess training data for Moose movement in case future iterations wish to explore other species.

These models will be used in Telepresent Nature, a tangible user interface that allows users to pinpoint the position of animals on a map using physical robots.

## Contents
- CSVFiles
    - CleanCSV: Contains cleaned csv files for each deer/moose in the RawCSV folder's files
        - `[animal id]_interpolated.csv` uses interpolation to define datapoints at specific time intervals
        - `[animal id].csv` files use interpolation only to fill gaps
    - ExtendedCSV: Contains files from the CleanCSV folder appended with model predictions
    - RawCSV: Contains raw csv files with deer/moose movement data in addition to a folder with temperature data and a file 
        - `ABoVE_Boutin Alberta Moose.csv`
        - `EuroDeer_ Roe deer in Italy 2005-2008.csv`
        - `trento_temperature_data.csv`: Contains the temperature of Trento, Italy around the time of the EuroDeer study; the EuroDeer study does contain temperature data, but future and past predictions beyond the initial study require this data
    - TestPerformanceCSV: Contains folders with csv data analyzing test results
        - TestMetrics: Has mean absolute errors and mean squared errors for testing
        - TestPredictions: Has the original test outputs and the predicted test outputs; good for visualizations
- Images: Images for README
- ModelFiles: .keras files with the models' settings
- `auxiliaries.py`: Contains classes, functions, and constants associated with the models' design
- `cleaning_functions.py`: Contains functions for the creation of the csv files in the CSVFile/CleanCSV and CSVFile/RawCSV/trento_temperature_data.csv
- `data_analysis.ipynb`: Contains analysis of the ata from the moose and deer datasets
- `model_running.ipynb`: Contains code training and testing the models
- `README.md`
- `requirements.txt`

## Training and Testing the Models

## What Can Be Further Developed