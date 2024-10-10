import datetime

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf

MAX_EPOCS = 20
OUT_STEPS = 100

# Handles train and test split for Future-Prediction Model
class FuturePredictionWindow():
    def __init__(self, species, file_name):
        # Load and add month and day values
        self.orig_df = pd.read_csv(f'CSVFiles/CleanCSV/{species}/{file_name}.csv')
        
        self.orig_df['timestamp'] = pd.DatetimeIndex(self.orig_df['timestamp'])
        self.orig_df['month'] = self.orig_df['timestamp'].map(lambda x: x.month)
        self.orig_df['day'] = self.orig_df['timestamp'].map(lambda x: x.day)
        self.timeline = self.orig_df.pop('timestamp')
        self.orig_df.pop('altitude')

        # Split data up
        n = len(self.orig_df)
        self.train_df = self.orig_df[0:int(n*0.7)]
        self.test_df = self.orig_df[int(n*0.7):]

        # Apply Virtual Crossing temperature data to testing set
        # Currently, we only have deer temperature data; you would need to find temperature data for Fort McMurray, Alberta, Canada
        # and delete the assert statement
        assert species != 'Moose'

        self.temp_df = pd.read_csv(f'CSVFiles/CleanCSV/{species}/temperature_data.csv')
        self.temp_df = self.temp_df[['datetime', 'temp']]
        self.temp_df.rename(columns={'datetime': 'timestamp', 'temp':'temperature'}, inplace=True)
        self.temp_df['timestamp'] = pd.DatetimeIndex(self.temp_df['timestamp'])
        self.temp_df.set_index('timestamp', inplace=True)

        for index in self.test_df.index:
            curr_date = self.timeline.loc[index]
            curr_date = curr_date.replace(minute=0, second=0, microsecond=0)
            self.test_df.loc[index, 'external-temperature'] = self.temp_df.loc[curr_date, 'temperature']

        # Normalize data
        self.norm_train_df = (self.train_df - self.train_df.mean())/self.train_df.std()
        self.norm_test_df = (self.test_df - self.train_df.mean())/self.train_df.std()
        
        # Split input and labels
        self.train_input = self.norm_train_df[['external-temperature', 'month', 'day']].values
        self.test_input = self.norm_test_df[['external-temperature', 'month', 'day']].values

        self.train_label = self.norm_train_df[['longitude', 'latitude']].values
        self.test_label = self.norm_test_df[['longitude', 'latitude']].values

        # Reshape
        self.train_input = self.train_input.reshape((self.train_input.shape[0], 1, self.train_input.shape[1]))
        self.test_input = self.test_input.reshape((self.test_input.shape[0], 1, self.test_input.shape[1]))

        self.train_label = self.train_label.reshape((self.train_label.shape[0], 1, self.train_label.shape[1]))
        self.test_label = self.test_label.reshape((self.test_label.shape[0], 1, self.test_label.shape[1]))

    def model_compilation_and_fitting(self, model, patience=2):
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        
        history = model.fit(self.train_input, self.train_label, epochs=MAX_EPOCS,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')])
        
        return history

    def csv_extension(self, url_dest, species, model):
        # Define data for the add on dataframe
        add_on_data = {
            'timestamp': [],
            'external-temperature': [],
            'month': [],
            'day': [],
            'longitude': [],
            'latitude': [],
        }

        # Determine Time Delta
        assert species == 'Moose' or species == 'Deer'
        if (species == 'Moose'):
            timedelta = 3
        elif (species == 'Deer'):
            timedelta = 4

        # Determine Start Time
        curr_date = max(self.timeline).replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=timedelta)

        while (curr_date <= max(self.temp_df.index)):
            # Derive input variables
            i = 0
            while (True):
                # Gets the previous temperature value if it has not been found in the given time
                if ((curr_date - datetime.timedelta(hours=i*timedelta)) in self.temp_df.index):
                    external_temp = self.temp_df.loc[curr_date, 'temperature']
                    break
                if (i > 1000):
                    raise Exception(f"Missing temperature value for {curr_date}")
                i += 1

            month = curr_date.month
            day = curr_date.day

            # Retrieve position output
            if isinstance(model, tf.keras.Sequential):
                output_fields = model(np.array([(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                                (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                                (day - self.train_df.mean()['day'])/self.train_df.std()['day']]).reshape((1, 1, 3)))*self.train_df[['longitude', 'latitude']].std() + self.train_df[['longitude', 'latitude']].mean()
                output_fields = output_fields.numpy()[0][0]
            elif isinstance(model, KNeighborsRegressor):
                output_fields = model.predict([[(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                                (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                                (day - self.train_df.mean()['day'])/self.train_df.std()['day']]])[0]*self.train_df[['longitude', 'latitude']].std() + self.train_df[['longitude', 'latitude']].mean()
                output_fields = output_fields.values
            else:
                raise ValueError("Enter a Sequential Model or KNeighbors Regressor")

            longitude = output_fields[0]
            latitude = output_fields[1] 

            # Load values
            add_on_data['timestamp'].append(curr_date)
            add_on_data['external-temperature'].append(external_temp)
            add_on_data['month'].append(curr_date.month)
            add_on_data['day'].append(curr_date.day)

            add_on_data['longitude'].append(longitude)
            add_on_data['latitude'].append(latitude)

            # Update curr_date
            curr_date += datetime.timedelta(hours=timedelta)
        
        # Generate base_df and add_on_df before combining them into one dataframe/csv
        base_df = self.orig_df.copy(deep=True)
        base_df['timestamp'] = self.timeline
        base_df['modeled'] = False
        
        add_on_df = pd.DataFrame(add_on_data)
        add_on_df['modeled'] = True
        
        combined_df = pd.concat([base_df, add_on_df], ignore_index=True)
        combined_df = combined_df[['timestamp', 'external-temperature', 'month', 'day', 'longitude', 'latitude', 'modeled']]

        combined_df.to_csv(f'CSVFiles/ExtendedCSV/{url_dest}_extended.csv', index=False)

    def model_compilation_and_fitting(self, model, patience=2):
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        
        history = model.fit(self.train_input, self.train_label, epochs=MAX_EPOCS,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')])
        
        return history

# Handles train and test split for Past-Prediction Model
class PastPredictionWindow():
    def __init__(self, species, file_name):
        # Load and add month and day values
        self.orig_df = pd.read_csv(f'CSVFiles/CleanCSV/{species}/{file_name}.csv').iloc[::-1].reset_index(drop=True)
        
        self.orig_df['timestamp'] = pd.DatetimeIndex(self.orig_df['timestamp'])
        self.orig_df['month'] = self.orig_df['timestamp'].map(lambda x: x.month)
        self.orig_df['day'] = self.orig_df['timestamp'].map(lambda x: x.day)
        self.timeline = self.orig_df.pop('timestamp')
        self.orig_df.pop('altitude')

        # Split data up
        n = len(self.orig_df)
        self.train_df = self.orig_df[0:int(n*0.7)]
        self.test_df = self.orig_df[int(n*0.7):]

        # Apply Virtual Crossing temperature data to testing set
        # Currently, we only have deer temperature data; you would need to find temperature data for Fort McMurray, Alberta, Canada
        # and delete the assert statement
        assert species != 'Moose'

        self.temp_df = pd.read_csv(f'CSVFiles/CleanCSV/{species}/temperature_data.csv')
        self.temp_df = self.temp_df[['datetime', 'temp']]
        self.temp_df.rename(columns={'datetime': 'timestamp', 'temp':'temperature'}, inplace=True)
        self.temp_df['timestamp'] = pd.DatetimeIndex(self.temp_df['timestamp'])
        self.temp_df.set_index('timestamp', inplace=True)

        for index in self.test_df.index:
            curr_date = self.timeline.loc[index]
            curr_date = curr_date.replace(minute=0, second=0, microsecond=0)
            self.test_df.loc[index, 'external-temperature'] = self.temp_df.loc[curr_date, 'temperature']

        # Normalize data
        self.norm_train_df = (self.train_df - self.train_df.mean())/self.train_df.std()
        self.norm_test_df = (self.test_df - self.train_df.mean())/self.train_df.std()
        
        # Split input and labels
        self.train_input = self.norm_train_df[['external-temperature', 'month', 'day']].values
        self.test_input = self.norm_test_df[['external-temperature', 'month', 'day']].values

        self.train_label = self.norm_train_df[['longitude', 'latitude']].values
        self.test_label = self.norm_test_df[['longitude', 'latitude']].values

        # Reshape
        self.train_input = self.train_input.reshape((self.train_input.shape[0], 1, self.train_input.shape[1]))
        self.test_input = self.test_input.reshape((self.test_input.shape[0], 1, self.test_input.shape[1]))

        self.train_label = self.train_label.reshape((self.train_label.shape[0], 1, self.train_label.shape[1]))
        self.test_label = self.test_label.reshape((self.test_label.shape[0], 1, self.test_label.shape[1]))

    def model_compilation_and_fitting(self, model, patience=2):
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        
        history = model.fit(self.train_input, self.train_label, epochs=MAX_EPOCS,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')])
        
        return history

    def csv_extension(self, url_dest, species, model):
        # Define data for the add on dataframe
        add_on_data = {
            'timestamp': [],
            'external-temperature': [],
            'month': [],
            'day': [],
            'longitude': [],
            'latitude': []
        }

        # Determine Time Delta
        assert species == 'Moose' or species == 'Deer'
        if (species == 'Moose'):
            timedelta = 3
        elif (species == 'Deer'):
            timedelta = 4

        # Determine Start Time
        curr_date = min(self.timeline).replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=timedelta)

        while (curr_date >= min(self.temp_df.index)):
            # Derive input variables
            i = 0
            while (True):
                if ((curr_date + datetime.timedelta(hours=i*timedelta)) in self.temp_df.index):
                    external_temp = self.temp_df.loc[curr_date, 'temperature']
                    break
                if (i > 1000):
                    raise Exception(f"Missing temperature value for {curr_date}")
                i += 1

            month = curr_date.month
            day = curr_date.day

            # Retrieve position output
            if isinstance(model, tf.keras.Sequential):
                output_fields = model(np.array([(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                                (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                                (day - self.train_df.mean()['day'])/self.train_df.std()['day']]).reshape((1, 1, 3)))*self.train_df[['longitude', 'latitude']].std() + self.train_df[['longitude', 'latitude']].mean()
                output_fields = output_fields.numpy()[0][0]
            elif isinstance(model, KNeighborsRegressor):
                output_fields = model.predict([[(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                                (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                                (day - self.train_df.mean()['day'])/self.train_df.std()['day']]])[0]*self.train_df[['longitude', 'latitude']].std() + self.train_df[['longitude', 'latitude']].mean()
                output_fields = output_fields.values
            else:
                raise ValueError("Enter a Sequential Model or KNeighbors Regressor")

            longitude = output_fields[0]
            latitude = output_fields[1] 

            # Load values
            add_on_data['timestamp'].append(curr_date)
            add_on_data['external-temperature'].append(external_temp)
            add_on_data['month'].append(curr_date.month)
            add_on_data['day'].append(curr_date.day)

            add_on_data['longitude'].append(longitude)
            add_on_data['latitude'].append(latitude)

            # Update curr_date
            curr_date -= datetime.timedelta(hours=timedelta)
        
        # Generate base_df and add_on_df before combining them into one dataframe/csv
        base_df = self.orig_df.copy(deep=True)
        base_df['timestamp'] = self.timeline
        base_df['modeled'] = False
        
        add_on_df = pd.DataFrame(add_on_data)
        add_on_df['modeled'] = True
        
        combined_df = pd.concat([base_df, add_on_df], ignore_index=True).iloc[::-1]
        combined_df = combined_df[['timestamp', 'external-temperature', 'month', 'day', 'longitude', 'latitude', 'modeled']]

        combined_df.to_csv(f'CSVFiles/ExtendedCSV/{url_dest}_extended.csv', index=False)

    def model_compilation_and_fitting(self, model, patience=2):
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        
        history = model.fit(self.train_input, self.train_label, epochs=MAX_EPOCS,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')])
        
        return history