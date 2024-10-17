# Telepresent Nature's Regressive Neural Network for Tracking Deer

## About the Project

This project analyzes two regressive neural networks (a Future-Prediction Regressive model and a Past-Prediction Regressive model) that predict a deer’s longitudinal and latitudinal positions using values for a month, day, and temperature. The Future-Prediction Regressive model is trained to make predictions into the future. The Past-Prediction Regressive model is trained to make predictions into the past. The repository also possess training data for Moose movement in case future iterations wish to explore other species.

These models will be used in Telepresent Nature, a tangible user interface that allows users to pinpoint the position of animals on a map using physical robots.

## Video of Future-Prediction Models Being Used
<video src="Images/TelepresentNatureDemo.mov"></video>

## Contents
- CSVFiles
    - CleanCSV: Contains cleaned csv files for each deer/moose in the RawCSV folder's files
        - `[animal id]_interpolated.csv` uses interpolation to define datapoints at specific time intervals
        - `[animal id].csv` files use interpolation only to fill gaps
    - ExtendedCSV: Contains files from the CleanCSV folder appended with model predictions
    - RawCSV: Contains raw csv files with deer/moose movement data in addition to a folder with temperature data and a file 
        - `ABoVE_Boutin Alberta Moose.csv`: Dataset acquired from [1]
        - `EuroDeer_ Roe deer in Italy 2005-2008.csv`: Dataset acquired from [2]
        - `trento_temperature_data.csv`: Contains the temperature of Trento, Italy around the time of the EuroDeer study from a concatenation of datasets in [3]; the EuroDeer study does contain temperature data, but future and past predictions beyond the initial study require this data
    - TestPerformanceCSV: Contains folders with csv data analyzing test results
        - TestMetrics: Has mean absolute errors and mean squared errors for testing
        - TestPredictions: Has the original test outputs and the predicted test outputs; good for visualizations
- Images_and_video: Contains media for README
- ModelFiles: .keras files with the models' settings
- `auxiliaries.py`: Contains classes, functions, and constants associated with the models' design
- `cleaning_functions.py`: Contains functions for the creation of the csv files in the CSVFile/CleanCSV and CSVFile/RawCSV/trento_temperature_data.csv
- `data_analysis.ipynb`: Contains analysis of the ata from the moose and deer datasets
- `model_running.ipynb`: Contains code training and testing the models
- `README.md`
- `requirements.txt`

## Model Descriptions
- The project analyzes two models:
    1) Future-Prediction Model: Regressive dense neural network that is trained to make predictions into the future
        - Input: (month, day, temperature)
        - Output: (longitude, latitude)
    2) Past-Prediction Model: Regressive dense neural network that is trained to make predictions into the past
        - Input: (month, day, temperature)
        - Output: (longitude, latitude)
- Dense neural networks were used due to them being easy to train and consistent outputs (i.e., the outputs do not change between inputs like in recurrent neural networks)
- The inputs (month, day, and temperature) were chosen due to their relation to animal tracking
    - Month and day accounts for the seasonal nature of animal movements
    - Temperature was picked due to its higher correlation (measured in pearson correlation coefficient) to longitude and latitude than other dataset variables

## Training and Testing the Models
- Currently, the models are trained to predict the positions of Deer GSM02927 from [2].
- The training set and testing set are a combination of data from [2] and [3] and consist of the following columns
    - Month: Number of month (1 for January, 2 for February, etc.)
    - Day: Day of the month
    - External-Temperature: Temperature of location (in celsius)
    - Longitude: The longitudinal position of the animal
    - Latititude: The latitudinal position of the animal
- The inputs (month, day, and external-temperature) are standardized before being inputted into the model. The model then outputs standardized versions of the output (longitude and latitude), requiring the user to reconvert the outputs to there original forms.
- The Future-Prediction model has a 70-30 split: First 70% of rows go to the training set and the last 30% of rows go to the testing set
- The Past-Prediction model is flipped: The last 70% of rows go to the training set and the first 30% of rows go to the testing set
- These training and testing sets are defined in the window classes found in `auxiliaires.py`

## What Can Be Further Developed
- Train the models against the other deers
- Train the models against the moose data
- Try RNN models that keep track of position as a state

## Sources
[1] H. Bohm, E. Neilson, C. de la Mare, and S. A. Boutin, 2014, “Wildlife habitat effectiveness and connectivity: moose ecology project summary report 2010–2012: Final report,” Movebank. [Online]. Available: https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study302664172.

[2] F. Cagnacci et al., “Partial migration in roe deer: migratory and resident tactics are end points of a behavioural gradient determined by ecological factors,” Oikos, vol. 120, no. 12, pp. 1790–1802, Nov. 2011, doi: https://doi.org/10.1111/j.1600-0706.2011.19441.x.

[3] Visual Crossing. "Weather Query Builder trento, italy 2006-10-02 to 2008-06-05." Visual Crossing. Aug. 22, 2024. [Online]. Available: https://www.visualcrossing.com/weather/weather-data-services.

[4] D. R. Williams, Jan. 11, 2024, "Earth Fact Sheet," NASA. [Online]. Available: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html.