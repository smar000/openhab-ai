import logger
import traceback
import datetime
import os, sys
import requests
import csv
import pandas as pd
import numpy as np
import io
import os
import json
from joblib import dump, load
from sse import SSEReceiver
from sklearn.model_selection import train_test_split

log = logger.log

# Internal: DataFrame column names when reducing event datetime to day of week and mins from midnight
DF_TIMESTAMP_COL_DOW     = "dayOfWeek"        # TODO! Look at one hot encoding instead...
DF_TIMESTAMP_COL_MINS    = "minsFromMidnight" # TODO! Look at one hot encoding instead...



class Colour:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


class OpenhabItem():
    def __repr__(self):
        return "openHAB item '{}'. Current state: {}. Last predicted state: {}".format(self.name, self.state, self.predicted_state)

    def __str__(self):
        return "openHAB item '{}'. Current state: {}. Last predicted state: {}".format(self.name, self.state, self.predicted_state)

    def __init__(self, name, state=None):
        self.name = name
        self._state = _state
        self.state_ts = None
        self.predicted_state = None       
        self.predicted_state_ts = None

    @property
    def state(self):
        return self._state
    

    @state.setter
    def state(self, new_state):
        self._state = new_state
        self.state_ts = datetime.datetime.now()



class Models(dict):
    """ Dict of model objects, with some additional functionality to make it easier to find underlying model object etc    """

    def __init__(self,*arg,**kw):
      super(Models, self).__init__(*arg, **kw)

    def load_collection_from_dict(self, models_dict):
        for k, m in models_dict.items():
            self[k] = Model(k, m)


    def get_model_for_input_item(self, item_name):
        for k,m in self.items():
            if item_name in m.inputs:
                return m
        return None

    def get_model_for_output_item(self, item_name):
        for k,m in self.items():
            if item_name in m.outputs:
                return m
        return None

    def stop_all_sse(self):
        for k,m in self.items():
            if m.openhab_sse:
                m.openhab_sse.stop()

    def do_predict_all_models(self):
        for k,m in self.items():
            if m.classifer:
                m.do_predict()


class Model:
    """ An object representing a single openHAB 'rule'. inputs are the openHAB items triggerring the corresponding output(s).    """    

    def __repr__(self):
        return "Model '{}'. Inputs: {}. Outputs: {}".format(self.name, self.inputs, self.outputs)

    def __str__(self):
        return "Model '{}'. Inputs: {}. Outputs: {}".format(self.name, self.inputs, self.outputs)

    def __init__(self, name, model_dict=None, inputs=[], outputs=[], classifier_type=None):
        self.name = name
        if model_dict:                        
            self.inputs = model_dict["inputs"] if "inputs" in model_dict else None
            self.outputs = model_dict["outputs"] if "outputs" in model_dict else None
            self._classifier_type = model_dict["classifier"] if "classifer" in model_dict else classifier_type
        else:            
            self.inputs = inputs
            self.outputs = outputs
            self._classifier_type = classifier_type
        
        self.classifier = None
        self.ai_model_retrain_ts = None
        self.last_df = None

        self.model_save_path = None
        self._filename = None
        self.isloaded = False
               
        self.db_query_url = None
        self.db_query_headers = None
        self.db_query_base = None
        
        self.save_training_data = False
        self.series_timeslot_mins = None

        self.predictions_save_filename = None

        self.openhab_url = None
        self.openhab_sse = None


    # @property
    # def filename(self):
    #     if not self._filename:
    #         self.get_model_filename_for_path()
    #     return self._filename

    @property
    def classifier_type(self):
        return self._classifier_type
    
    @classifier_type.setter
    def classifier_type(self, classifier_type):
        self._classifier_type = classifier_type
        if self.inputs:
            self.get_model_filename_for_path()


    def get_model_filename_for_path(self, path=None):    
        if path:
            self.model_save_path = path

        if not self.model_save_path:
            self.model_save_path = "./"

        self._filename = os.path.join(self.model_save_path, "{}_{}.joblib".format(self.classifier, "-".join(self.outputs)))
        return self._filename


    def get_openhab_sse_topics(self):
        """ return openhab SSE topic list for the input items"""
        topics = []
        for item in self.inputs:
            topics.append("smarthome/items/{}/state".format(item)) #statechanged
        return topics


    def subscribe_to_openhab(self):
        """ Subscribe to openHAB SSE for the model's input items"""
        topics = self.get_openhab_sse_topics()
        url = "{}/rest/events?topics={}".format(self.openhab_url, ",".join(topics))
        log.debug("[{:12.12}] openHAB SSE url: {}".format(self.name, url))

        self.openhab_sse = SSEReceiver(url, self.sse_event_callback)
        self.openhab_sse.start()

        log.info("[{:12.12}] Subscribed to openHAB items: {}".format(self.name, ", ".join(self.inputs)))


    def sse_event_callback(self, event):
        """ SSE Callback function """        
        log.debug("[{:12.12}] Callback function called with event: {}".format(self.name, event))
        state_event = json.loads(event["data"])
        # print(state_event)
        state_str = json.loads(state_event["payload"])["value"] if "payload" in state_event  else "-"    
        item_name = state_event["topic"].replace("smarthome/items/","").replace("/statechanged","").replace("/state","") if "topic" in state_event else "-"

        self.do_predict(item_name, state_str)


    def get_openhab_states_for(self, items_list):
        states = {}
        for item_name in items_list:
            states[item_name] = self.get_openhab_state_for_item(item_name)
        return states


    def get_openhab_state_for_item(self, item_name):
        rest_url = "{}/rest/items/{}/state".format(self.openhab_url, item_name)
        response = requests.get(rest_url)            
        curr_state = response.text if response.status_code == 200 else None
        curr_state = self.convert_state_to_num(curr_state)
        return curr_state


    def convert_state_to_num(self, state):
        if state == "OFF":
            state = 0
        elif state == "ON":
            state = 1
        # Otherwise just return the state as is
        return state


    def do_predict(self, item_name=None, state=None, log_prefix=""):
        '''
            Do prediction. If item_name/state supplied, no need to look up values for this. 
            All other input items looked up from openHAB, before doing prediction.
        '''         

        # state = 1 if state == "ON" else 0
        if log_prefix: log_prefix = "[{}] ".format(log_prefix) 
        # print(state_event)

        now = datetime.datetime.now()
        df = self.get_df_for_current_input_states(item_name, state)
        log.debug("[{:12.12}] DF for current state:\n{}".format(self.name, df))

        if df is not None and not df.empty:
            try:
                log.debug("[{:12.12}] Last DF:\n{}".format(self.name, self.last_df))
                # printdf(self.last_df)
                y_pred=self.classifer.predict(df)
                log.debug("[{:12.12}] y_pred (type={}, shape: {}): {}".format(self.name, type(y_pred), y_pred.shape, y_pred))
                # Get current input and output items states for comparison
                curr_input_states = self.get_openhab_states_for(self.inputs)
                curr_output_states = self.get_openhab_states_for(self.outputs)
                # print("type(y_pred): {}, y_pred: {}".format(type(y_pred), y_pred))
                # y_pred_string = ", ".join(y_pred.tolist()) # if type(y_pred) == np.ndarray else y_pred
                predicted_states = {}
                count = 0
                for item_name in self.outputs:                
                    predicted_state = y_pred.round() if y_pred.size==1 else y_pred[0][count].round()               
                    predicted_states[item_name] = "{} (actual) -> {} (predicted)".format(curr_output_states[item_name], predicted_state)
                    count += 1

                log.debug("[{:12.12}] {}Input items states : {}'".format(self.name, log_prefix, curr_input_states))
                log.debug("[{:12.12}] {}Output items states:'{}'".format(self.name, log_prefix, predicted_states))


                
                if curr_output_states: log.debug("[{:12.12}] {}Current states (from openHAB):'{}'".format(self.name, log_prefix, curr_output_states))
                
                # Create DataFrame for the whole row of inputs/outputs,
                # - for pd concat, we need indexes to be aligned. y_pred does not have an index col. Add the timestamp
                #   used in the df
                full_df_row = pd.concat([df, pd.DataFrame(y_pred, columns=self.outputs, index=[df.index[0]])], axis=1)
                

                # Get any changes to previous dataframe
                if self.last_df is not None:     
                    if not np.array_equal(self.last_df.values, full_df_row.values):
                        for (item_name, value) in full_df_row.iteritems():      # value is pandas Series
                            if item_name in self.last_df.columns:   
                                last_value = self.last_df[item_name][0]
                                current_value = value.iloc[0]                    # the numpy.int64 value in location 0 of the Series
                                # print("item_name: '{}', value: '{}', value type: '{}', current_value: '{}', last_value type: {}".format(
                                #     item_name, value, type(value), current_value, type(last_value)))                            
                                in_out = "{}OUT{}".format(Colour.GREEN, Colour.END) if item_name in self.outputs else "IN "
                                suffix = " [Predicted]" if item_name in [self.outputs] else ""
                                if current_value != last_value:
                                    log.info("[{:12.12}] {:<3}: {:<30} {} -> {}{}".format(self.name,in_out, item_name, last_value, current_value, suffix))
                            # else:
                            #     log.error("[{:12.12}] Column '{}' not found in the self.last_df:\n{}".format(self.name, item_name, self.last_df))                
                    # else:
                    #     log.debug("No changes detected")
                else:
                    log.info("[{:12.12}] Inputs/predicted output(s) states: ".format(self.name))
                    for i in range(full_df_row.shape[1]): # iterate over all columns
                        col_name = full_df_row.columns[i]
                        suffix = " [Predicted]" if col_name in [self.outputs] else ""
                        log.info("[{:12.12}] --- {:<30} -> {}{}".format(self.name, col_name, full_df_row[col_name][0], suffix))
                    log.info("")

                    # log.info("[{:12.12}] Delta_DF: {}".format(self.name, full_df_row))
                self.last_df = full_df_row
                self.write_df_to_file(full_df_row)          

            except Exception as ex:
                log.error(ex)
                log.error("[{:12.12}] DataFrame: {}".format(self.name, df))
                log.error(traceback.format_exc())            

        else:
            log.error("[{:12.12}] Prediction failed as DataFrame of current input item states not obtained".format(self.name))


    def get_df_for_current_input_states(self, override_item_name=None, override_item_state=None):
        """
            Generate a DataFrame of the current states of the self.inputs obtained from the openHAB server, 
            along with day/time period column values
        """       
        input_item_states = {}

        # Get updated input item states from openHAB
        for item_name in self.inputs:                    
            if not override_item_name or item_name != override_item_name: # ignore if item is override_item_name as we should have the state for this already            
                rest_url = "{}/rest/items/{}/state".format(self.openhab_url, item_name)
                response = requests.get(rest_url)
                # log.info("response: {}, {}".format(response, response.text))
                # return
                if response.status_code == 200:
                    state = self.convert_state_to_num(response.text)

                    input_item_states[item_name] = state
                else:
                    log.error("[{:12.12}] Invalid response code from openHAB REST api for item '{}': {} {}".format(self.name, item_name, response.status_code, response.text))

        if override_item_name:
            if type(override_item_state) == str: override_item_state = self.convert_state_to_num(override_item_state)
            input_item_states[override_item_name] = override_item_state
        
        # Get the time of day 'slot', and then convert both ToD and DoW to cyclical equivalents
        if input_item_states:            
            now = datetime.datetime.now()
            
            # round to the next period as used in the training model
            period_mins = now.minute + (self.series_timeslot_mins - now.minute % self.series_timeslot_mins)
            # log.info("[{:12.12}] mins: {}".format(self.name, period_mins))
            mins_midnight = [now.hour * 60 + period_mins]
            now = datetime.datetime(now.year, now.month, now.day, now.hour, 0) + datetime.timedelta(minutes=period_mins)

            new_data = {
                DF_TIMESTAMP_COL_DOW    : datetime.datetime.today().weekday(), 
                DF_TIMESTAMP_COL_MINS   : mins_midnight, 
            }
            df_ts = pd.DataFrame(new_data, index=[pd.Timestamp(now)])
            dow_sin, dow_cos = self.convert_cyclic_to_sin_cos(df_ts[DF_TIMESTAMP_COL_DOW], 7)
            tod_sin, tod_cos = self.convert_cyclic_to_sin_cos(df_ts[DF_TIMESTAMP_COL_MINS], 1440)

            df = pd.DataFrame(input_item_states, index=[pd.Timestamp(now)])
            df.insert(0, 'dow_sin', dow_sin)               
            df.insert(1, 'dow_cos', dow_cos)

            
            df.insert(2, 'tod_sin', tod_sin)               
            df.insert(3, 'tod_cos', tod_cos)

            # df.set_index("_time")

            log.debug("[{:12.12}] DF for current state:\n{}".format(self.name, df))
            # printdf(df)

            return df
        else:
            log.error("[{:12.12}] Failed to get current states for input items from openHAB".format(self.name))
            return None


    def post_to_openhab(self, item_name, new_state):
        if OPENHAB_SEND_PREDICTIONS:
            log.info("[{:12.12}] Posting '{}' to openHAB item '{}'".format(self.name, new_state, item_name))
            
        # curl -X POST --header "Content-Type: text/plain" --header "Accept: application/json" -d "Test" "http://openhab:7070/rest/items/Voice_Command"
        headers = {"accept" : "application/json", "content-type" : "text/plain"}
        url = "{}/rest/items/{}".format(self.openhab_url,item_name)
        response = requests.post(url, data=new_state, headers=headers)
        log.debug("[{:12.12}] Posted '{}' to openHAB for item '{}'. Reponse: '{}'".format(self.name, new_state, item_name, response))


    def load_ai_model_from_file(self, filename = None):
        if filename:
            self._filename = filename
        
        if not self._filename:
            self.get_model_filename_for_path()

        log.info("[{:12.12}] Loading existing model '{}'".format(self.name, self._filename))

        self.classifer = load(self._filename)

        # Assume last modified time for the model is the model generation time
        last_modified_ts = os.path.getmtime(self._filename)        

        self.ai_model_retrain_ts = datetime.datetime.fromtimestamp(last_modified_ts) 

        log.info("[{:12.12}] Model loaded from file (last trained {:%H:%M %Y-%m-%d})".format(self.name, last_model_rebuild))        



    def write_df_to_file(self, df):
        if not self.predictions_save_filename:
            return False

        with open(self.predictions_save_filename, "a") as f:
            df.to_csv(f, header=f.tell()==0, mode="a")
        return True

    def retrain_ai_model(self, classifier_type=None):
        if classifier_type: 
            self.classifier_type = classifier_type

        if not self.classifier_type:
            raise Exception("No classifer type specified")

        if self.classifier_type == "RF":
            self.classifier = self.generate_model_randomforest()
        elif self.classifier_type == "MLP":
            self.classifier = self.generate_model_mlp()
        else:
            self.classifier = None
            log.error("[{:12.12}] Invalid classifier '{}'".format(self.name, self.classifier_type))
            return None

        if not self.classifier:
            log.error("[{:12.12}] Model training failed. Exiting...".format(self.name))
            sys.exit(0)

        # dump(self.classifier, self.filmename)
        log.info("[{:12.12}] Model trained and saved to file '{}'".format(self.name, self._filename))

        self.ai_model_retrain_ts = datetime.datetime.now()



    def convert_cyclic_to_sin_cos(self, cyc_series, total_periods=1440, shift=0):
        cyc_sin = np.sin((cyc_series + shift) * (2.*np.pi/total_periods))               # Normalised with number of mins in day
        cyc_cos = np.cos((cyc_series + shift) * (2.*np.pi/total_periods))

        return cyc_sin, cyc_cos


    def get_historical_data_for_item(self, item_name):
        if not item_name:
            log.error("No item name given. Aborting...")
            return

        log.debug("[{:12.12}] ---> Getting data for item '{}' from database...".format(self.name, item_name))    
        query = self.db_query_base.replace("<<>>",item_name)
        try:
            response = requests.post(self.db_query_url, data=query, headers=self.db_query_headers)        
            if response.status_code == 200:
                time_series = pd.read_csv(io.StringIO(response.content.decode('utf-8')), 
                    usecols=[5,6], names=["_time", item_name], 
                    header=3, parse_dates=[0], index_col=0, squeeze=True)            
                assert not time_series.empty, "Empty time series returned for '{}'. DB Query: {}".format(item_name, query)
                return time_series
            else:
                log.error("[{:12.12}] Failed to get data from server. Response: {}".format(self.name, response))
                log.error("[{:12.12}] --- URL: '{}'\n --- Query: '{}'\n --- Headers: '{}'".format(self.name, self.db_query_url, query, self.db_query_headers))
        
        except Exception as ex:
            log.error(ex)
            log.error(traceback.format_exc())
            return None


    def get_dataframe_historical_data(self):    
        time_series = []
        log.info("[{:12.12}] Creating new model for input items '{}".format(self.name, self.inputs))

        try:
            for item in self.inputs:
                time_series.append(self.get_historical_data_for_item(item))
            
            for item in self.outputs:
                time_series.append(self.get_historical_data_for_item(item))
            log.info("[{:12.12}] Loaded {} time series data from database".format(self.name, len(time_series)))
            # print(time_series)       

            start = max([i.index.min() for i in time_series])
            end = min([i.index.max() for i in time_series])

            # Move start/end to beginning/end of respective interval periods
            start = start + pd.Timedelta(minutes=self.series_timeslot_mins - (start.minute % self.series_timeslot_mins))
            end = end + pd.Timedelta(minutes=end.minute % self.series_timeslot_mins)     

            log.info("[{:12.12}] Earliest common start: \t{}".format(self.name, start))
            log.info("[{:12.12}] Latest common end    : \t{}".format(self.name, end))

            # Create a new column, _time, divided into the required time slots (note we also have date here)
            # and then merge in the individual series previously loaded into corresponding time slots
            df = pd.DataFrame({'_time': pd.date_range(start,end,freq='{}T'.format(self.series_timeslot_mins))})
            for s in time_series:
                df = pd.merge_asof(df, s, on='_time')
            df = df.set_index("_time")

            # Add columns for day of week and time period in minutes
            df.insert(0, DF_TIMESTAMP_COL_DOW, df.index.dayofweek)
            df.insert(1, DF_TIMESTAMP_COL_MINS, df.index.hour * 60 + df.index.minute)

            # Deal with cyclical nature of day/time etc (e.g. try mapping to circle). TODO! Test with one-hot...    
            dow_sin, dow_cos = self.convert_cyclic_to_sin_cos(df[DF_TIMESTAMP_COL_DOW], 7)  
            df.insert(2, 'dow_sin', dow_sin)               
            df.insert(3, 'dow_cos', dow_cos)

            tod_sin, tod_cos = self.convert_cyclic_to_sin_cos(df[DF_TIMESTAMP_COL_MINS], 1440) # Normalised with number of 1440 mins in day
            df.insert(4, 'tod_sin', tod_sin)               
            df.insert(5, 'tod_cos', tod_cos)


            # df.insert(2, 'tod_sin', np.sin(df[DF_TIMESTAMP_COL_MINS]*(2.*np.pi/1440)))               # Normalised with number of mins in day
            # df.insert(3, 'tod_cos', np.cos(df[DF_TIMESTAMP_COL_MINS]*(2.*np.pi/1440)))

            # Drop the original DoW and mins past midnight cols as no longer required
            df.drop(DF_TIMESTAMP_COL_DOW, axis=1, inplace=True)
            df.drop(DF_TIMESTAMP_COL_MINS, axis=1, inplace=True)

            # df['mnth_sin'] = np.sin((df.mnth-1)*(2.*np.pi/12))            # Normalised
            # df['mnth_cos'] = np.cos((df.mnth-1)*(2.*np.pi/12))

            log.info("[{:12.12}] DataFrame created and merged. Shape: {}".format(self.name, df.shape))
            log.info("[{:12.12}] DataFrame columns: {}".format(self.name, ", ".join(df.columns.values)))
            log.debug(df)

            # if self.save_training_data:
            #     df_file_name = os.path.join(MODELS_FOLDER, "{}_data.csv".format("-".join(self.outputs)))
            #     df.to_csv(self.filename)
            return df

        except Exception as ex:
            log.error(ex)        
            log.error(traceback.format_exc())
            if item: log.error("[{:12.12}] Error in processing series for '{}'".format(self.name, item))
            log.error("[{:12.12}] time_series[]:\n{}".format(self.name, time_series))
            return None


    def get_training_and_test_dataframes(self, test_size=0.2):
        """ Return separate dataframes for inputs and outputs, split into training and test data """

        try:
        
            df = self.get_dataframe_historical_data()   
            if df is None or df.empty:
                raise Exception("DataFrame of historical data returned None or empty")

            np.random.seed(7)           # fix random seed for reproducibility

            # Split data into training/test
            output_col = df.columns.get_loc(self.outputs[0])
            X = df.iloc[:, 0:output_col]

            # X = df[[DF_TIMESTAMP_COL_DOW, DF_TIMESTAMP_COL_MINS] + input_items]
            y = df[self.outputs]

            log.debug("[{:12.12}] INPUT ITEMS: X ({}): {}".format(self.name, X.shape, X))
            log.debug("[{:12.12}] OUTPUT ITEMS: y ({}): {}".format(self.name, y.shape, y))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) # i.e. training size = 1 - test_size
            log.debug("[{:12.12}] Total set shape: {}. Training set row count: {}. Test set row count: {}".format(self.name, df.shape,len(X_train), len(X_test)))

            return X, y, X_train, X_test, y_train, y_test

        except Exception as ex:
            log.error(ex)
            if not X.empty: log.error("[{:12.12}] X ({}): {}".format(self.name, len(X), X))
            if not y.empty: log.error("[{:12.12}] y ({}): {}".format(self.name, len(y), y))
            log.error(traceback.format_exc())
            return None, None, None, None


    def generate_model_randomforest(self, n_estimators=200):    
        from sklearn.ensemble import RandomForestClassifier

        try:

            X, y, X_train, X_test, y_train, y_test = self.get_training_and_test_dataframes()

            # Create/train model
            self.classifer = RandomForestClassifier(n_estimators=n_estimators)    #Create a Gaussian Classifier
            log.debug("[{:12.12}] X_train: {}".format(self.name, X_train))
            log.debug("[{:12.12}] y_train: {}".format(self.name, y_train))
            
            y_train_values = y_train.values.ravel() if len(self.outputs)==1 else y_train

            self.classifer.fit(X_train,y_train_values)                 #Train the model using the training sets
            y_pred=self.classifer.predict(X_test)                      # Predict against the test data

            from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
            scores_test = metrics.accuracy_score(y_test, y_pred) * 100
            log.info("[{:12.12}] Model generated:".format(self.name)) 
            log.info("[{:12.12}] --- {}Test data accuracy: {:.2f}%{}".format(self.name,Colour.BOLD, scores_test, Colour.END))
            # log.info("[{:12.12}] Model generated. Accuracy with training data: {}{:.1f}%{}".format(self.name, Colour.BOLD, metrics.accuracy_score(y_test, y_pred) * 100), Colour.END)

            return self.classifer 

        except Exception as ex:
            log.error(ex)
            log.error(traceback.format_exc())
            return None


    def generate_model_mlp(self, n_estimators=200):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.models import model_from_json

        try:
            X, y, X_train, X_test, y_train, y_test = self.get_training_and_test_dataframes()

            input_width = len(X_train.columns)
            log.debug("[{:12.12}] Training input column count: {}, Training set shape = {}".format(self.name, input_width, X_train.shape))
            
            # create 3 layer model. Assume number of neurons in 1st layer is double the input size, 2nd layer is 3/4 of 1st etc
            self.classifer = Sequential()
            self.classifer.add(Dense(input_width * 8, input_dim=input_width, activation='relu'))
            self.classifer.add(Dense(input_width * 4, activation='relu'))
            self.classifer.add(Dense(input_width * 2, activation='relu'))
            self.classifer.add(Dense(1, activation='sigmoid')) 
            
            self.classifer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       # Compile model
            
            self.classifer.fit(X, y, epochs=150, batch_size=16, verbose=0)                                   # Fit the model
            
            # evaluate the model
            scores_training = self.classifer.evaluate(X_train, y_train, verbose=0)
            log.info("Model generated:") 
            log.info(" --- Training data {}: {:.2f}%".format(self.classifer.metrics_names[1], scores_training[1]*100))
            scores_test = self.classifer.evaluate(X_test, y_test, verbose=0)
            log.info(" --- {}Test data {}: {:.2f}%{}".format(
                Colour.BOLD, self.classifer.metrics_names[1], scores_test[1]*100, Colour.END))

            return self.classifer


        except Exception as ex:
            log.error(ex)
            log.error(traceback.format_exc())
            return None

