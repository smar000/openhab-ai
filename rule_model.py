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
DF_TIMESTAMP_COL_MINS    = "minsFromMidnight" 
DF_TIMESTAMP_COL_DOW     = "_DOW"        
DF_TIMESTAMP_COL_TOD     = "_TOD"        
DF_TIMESTAMP_COL_MONTH   = "_MONTH"        

INTERNAL_DTM_ITEMS_LIST      = [DF_TIMESTAMP_COL_MONTH, DF_TIMESTAMP_COL_DOW, DF_TIMESTAMP_COL_TOD, DF_TIMESTAMP_COL_MINS]
DEFAULT_INPUT_ITEMS      = [DF_TIMESTAMP_COL_DOW, DF_TIMESTAMP_COL_TOD]



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
        return "openHAB item '{}'. Current state: {}. Last predicted state: {}".format(self.name_trunc, self.state, self.predicted_state)

    def __str__(self):
        return "openHAB item '{}'. Current state: {}. Last predicted state: {}".format(self.name_trunc, self.state, self.predicted_state)

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
            if m.classifier:
                m.do_predict()


class Model:
    """ An object representing a single openHAB 'rule'. inputs are the openHAB items triggerring the corresponding output(s).    
        If not specified, default inputs are day of week and time of day
    """    

    def __repr__(self):
        return "Model '{}'. Inputs: {}. Outputs: {}".format(self.name_trunc, self.inputs, self.outputs)


    def __str__(self):
        return "Model '{}'. Inputs: {}. Outputs: {}".format(self.name_trunc, self.inputs, self.outputs)


    def __init__(self, name, model_dict=None, inputs=[], outputs=[], classifier_type=None):
        self.name = name
        self.name_trunc = (self.name[:10] + '..') if len(self.name) >= 12 else self.name

        if model_dict:                        
            self.inputs = model_dict["inputs"] if "inputs" in model_dict else None
            self.outputs = model_dict["outputs"] if "outputs" in model_dict else None
            self._classifier_type = model_dict["classifier"] if "classifier" in model_dict else classifier_type
        else:            
            self.inputs = inputs
            self.outputs = outputs
            self._classifier_type = classifier_type
        
        # self.inputs_exc_internal = filter(lambda x: x not in INTERNAL_DTM_ITEMS_LIST, self.inputs)
        self.classifier = None
        self.ai_model_retrain_ts = None
        self.last_df = None

        self.model_save_path = "./"
        self.data_save_path = "./"
        self.ai_model_filename = None
        self.training_data_filename = None
        self.predictions_save_filename = None
        self.send_predictions_to_openhab = False

        self.show_all_predictions = False

        self.save_training_data = False
        self.save_trained_model = False
        self.save_predictions = False

        self.db_query_url = None
        self.db_query_headers = None
        self.db_query_base = None
        
        self.series_timeslot_mins = None



        self.openhab_url = None
        self.openhab_sse = None

        self.show_all_input_changes = False


    @property
    def classifier_type(self):
        return self._classifier_type
    
    @classifier_type.setter
    def classifier_type(self, classifier_type):
        self._classifier_type = classifier_type
        if self.inputs:
            self.get_ai_model_filename()


    def get_ai_model_filename(self, path=None):    
        if path:
            self.model_save_path = path

        if not self.model_save_path:
            self.model_save_path = "./"
        self.ai_model_filename = os.path.join(self.model_save_path, "{}_{}.joblib".format(self.classifier_type, "-".join(self.outputs)))
        return self.ai_model_filename


    def get_openhab_sse_topics(self):
        """ return openhab SSE topic list for the input items"""
        topics = []
        for item in self.inputs:
            if item not in INTERNAL_DTM_ITEMS_LIST:
                topics.append("smarthome/items/{}/state".format(item)) #statechanged
        log.debug("[{}] SSE Topics: {}".format(self.name_trunc, topics))
        return topics


    def subscribe_to_openhab(self):
        """ Subscribe to openHAB SSE for the model's input items"""
        topics = self.get_openhab_sse_topics()
        url = "{}/rest/events?topics={}".format(self.openhab_url, ",".join(topics))
        log.debug("[{:12.12}] openHAB SSE url: {}".format(self.name_trunc, url))

        self.openhab_sse = SSEReceiver(url, self.sse_event_callback)
        self.openhab_sse.start()

        log.info("[{:12.12}] Subscribed to openHAB items: {}".format(self.name_trunc, ", ".join(filter(lambda x: x not in INTERNAL_DTM_ITEMS_LIST, self.inputs))))


    def sse_event_callback(self, event):
        """ SSE Callback function """        
        log.debug("[{:12.12}] Callback function called with event: {}".format(self.name_trunc, event))
        state_event = json.loads(event["data"])
        # print(state_event)
        state_str = json.loads(state_event["payload"])["value"] if "payload" in state_event  else "-"    
        item_name = state_event["topic"].replace("smarthome/items/","").replace("/statechanged","").replace("/state","") if "topic" in state_event else "-"

        self.do_predict(item_name, state_str)


    def get_openhab_states_for(self, items_list):
        states = {}        
        for item_name in filter(lambda x: x not in INTERNAL_DTM_ITEMS_LIST, items_list):
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
        log.debug("[{:12.12}] DF for current state:\n{}".format(self.name_trunc, df))

        if df is not None and not df.empty:
            try:
                log.debug("[{:12.12}] Last DF:\n{}".format(self.name_trunc, self.last_df))
                y_pred=self.classifier.predict(df)
                log.debug("[{:12.12}] y_pred (type={}, shape: {}): {}".format(self.name_trunc, type(y_pred), y_pred.shape, y_pred))
                
                # Get current input and output items states for comparison
                curr_input_states = self.get_openhab_states_for(self.inputs)
                curr_output_states = self.get_openhab_states_for(self.outputs)
                predicted_states = {}
                count = 0
                for item_name in self.outputs:                
                    predicted_state = y_pred.round() if y_pred.size==1 else y_pred[0][count].round()               
                    predicted_states[item_name] = "{} (actual) -> {} (predicted)".format(curr_output_states[item_name], predicted_state)
                    count += 1

                log.debug("[{:12.12}] {}Input items states : {}'".format(self.name_trunc, log_prefix, curr_input_states))
                log.debug("[{:12.12}] {}Output items states:'{}'".format(self.name_trunc, log_prefix, predicted_states))
                
                if curr_output_states: log.debug("[{:12.12}] {}Current states (from openHAB):'{}'".format(self.name_trunc, log_prefix, curr_output_states))
                
                # Create DataFrame for the whole row of inputs/outputs,
                # - for pd concat, we need indexes to be aligned. y_pred does not have an index col. Add the timestamp
                #   used in the df
                full_df_row = pd.concat([df, pd.DataFrame(y_pred, columns=self.outputs, index=[df.index[0]])], axis=1)

                # Get any changes to previous dataframe
                predicted_changes = {}
                if self.last_df is not None:     
                    if not np.array_equal(self.last_df.values, full_df_row.values):
                        for (item_name, value) in full_df_row.iteritems():      # value is pandas Series
                            if item_name in self.last_df.columns:   
                                last_value = self.last_df[item_name][0]
                                current_value = value.iloc[0]                    # the numpy.int64 value in location 0 of the Series
                                in_out = "OUT" if item_name in self.outputs else "IN "
                                suffix = " [Predicted]" if item_name in [self.outputs] else ""
                                
                                if current_value != last_value:
                                    if in_out == "OUT":
                                        colour_start = Colour.GREEN 
                                        colour_end = Colour.END
                                        predicted_changes[item_name] = current_value
                                    else:
                                        colour_start = ""
                                        colour_end = ""

                                if self.show_all_input_changes or in_out == "OUT": 
                                    if self.show_all_predictions or current_value != last_value:
                                        log.info("[{:12.12}] {}{:<3}: {:<30} {} -> {}{}{}".format(
                                            self.name_trunc, colour_start, in_out, item_name, last_value, current_value, suffix, colour_end))
                else:
                    log.info("[{:12.12}] Starting inputs/predicted output(s) states: ".format(self.name_trunc))
                    for i in range(full_df_row.shape[1]): # iterate over all columns
                        col_name = full_df_row.columns[i]
                        suffix = " [Predicted]" if col_name in [self.outputs] else ""
                        log.info("[{:12.12}] --- {:<30} -> {}{}".format(self.name_trunc, col_name, full_df_row[col_name][0], suffix))
                    log.info("")

                    # log.info("[{:12.12}] Delta_DF: {}".format(self.name_trunc, full_df_row))
                self.last_df = full_df_row

                if self.save_predictions:
                    self.save_predictions_to_file(full_df_row)          

                if self.send_predictions_to_openhab and predicted_changes: # Only post if an actual *change* predicted - hence dict instead of full_df_row
                    self.post_predictions_to_openhab(predicted_changes)

            except Exception as ex:
                log.error(ex)
                log.error("[{:12.12}] DataFrame: {}".format(self.name_trunc, df))
                log.error(traceback.format_exc())            

        else:
            log.error("[{:12.12}] Prediction failed as DataFrame of current input item states not obtained".format(self.name_trunc))


    def get_df_for_current_input_states(self, override_item_name=None, override_item_state=None):
        """
            Generate a DataFrame of the current states of the self.inputs obtained from the openHAB server, 
            along with day/time period column values
        """       
        input_item_states = {}

        # Get updated input item states from openHAB
        for item_name in self.inputs:                    
            if item_name not in INTERNAL_DTM_ITEMS_LIST and (not override_item_name or item_name != override_item_name): # ignore if item is override_item_name as we should have the state for this already            
                rest_url = "{}/rest/items/{}/state".format(self.openhab_url, item_name)
                response = requests.get(rest_url)
                # log.info("response: {}, {}".format(response, response.text))
                # return
                if response.status_code == 200:
                    state = self.convert_state_to_num(response.text)

                    input_item_states[item_name] = state
                else:
                    log.error("[{:12.12}] Invalid response code from openHAB REST api for item '{}': {} {}".format(self.name_trunc, item_name, response.status_code, response.text))

        if override_item_name:
            if type(override_item_state) == str: override_item_state = self.convert_state_to_num(override_item_state)
            input_item_states[override_item_name] = override_item_state
        
        # Get the time of day 'slot', and insert any required calendar items
        if input_item_states or self.inputs == DEFAULT_INPUT_ITEMS:            
            now = datetime.datetime.now()
            
            # round to the next period as used in the training model
            period_mins = now.minute + (self.series_timeslot_mins - now.minute % self.series_timeslot_mins)
            # log.info("[{:12.12}] mins: {}".format(self.name_trunc, period_mins))
            mins_midnight = [now.hour * 60 + period_mins]
            slot_end = datetime.datetime(now.year, now.month, now.day, now.hour, 0) + datetime.timedelta(minutes=period_mins)

            new_data = {
                DF_TIMESTAMP_COL_DOW    : datetime.datetime.today().weekday(), 
                DF_TIMESTAMP_COL_MINS   : mins_midnight, 
            }

            df_ts = pd.DataFrame(new_data, index=[pd.Timestamp(slot_end)])
            if input_item_states:
                df = pd.DataFrame(input_item_states, index=[pd.Timestamp(slot_end)])
            else:
                df = pd.DataFrame(index=[pd.Timestamp(slot_end)])

            df = self.insert_calendar_item_inputs(df)

            return df
        else:
            log.error("[{:12.12}] Failed to get current states for input items from openHAB".format(self.name_trunc))
            return None


    def post_predictions_to_openhab(self, predictions):
        # predicions is a dict 
        if predictions:
            for item_name in predictions:
                    self.post_state_to_openhab(item_name, predictions[item_name])


    def post_state_to_openhab(self, item_name, new_state):        
        log.debug("[{:12.12}] Posting '{}' to openHAB item '{}'".format(self.name_trunc, new_state, item_name))
            
        # curl -X POST --header "Content-Type: text/plain" --header "Accept: application/json" -d "Test" "http://openhab:7070/rest/items/<item>"
        headers = {"accept" : "application/json", "content-type" : "text/plain"}
        url = "{}/rest/items/{}".format(self.openhab_url,item_name)
        response = requests.post(url, data=new_state, headers=headers)
        log.info("[{:12.12}] {}Posted '{}' to openHAB item '{}'. Reponse: '{}'{}".format(
            self.name_trunc, Colour.RED, new_state, item_name, response, Colour.END))


    def load_ai_model_from_file(self, filename = None):
        if filename:
            self.ai_model_filename = filename
        
        if not self.ai_model_filename:
            self.get_ai_model_filename()

        if os.path.exists(self.get_ai_model_filename()):
            log.info("[{:12.12}] Loading existing model '{}'".format(self.name_trunc, self.ai_model_filename))
            self.classifier = load(self.ai_model_filename)
            last_modified_ts = os.path.getmtime(self.ai_model_filename)        # Assume last modified time for the model is the model generation time
            self.ai_model_retrain_ts = datetime.datetime.fromtimestamp(last_modified_ts) 
            log.info("[{:12.12}] Model loaded from file (last trained {:%H:%M %Y-%m-%d})".format(self.name_trunc, self.ai_model_retrain_ts))        
        else:
            log.info("[{:12.12}] Failed to load model. File '{}' not found".format(self.name_trunc, self.ai_model_filename))        


    def save_predictions_to_file(self, df):        
        if not self.predictions_save_filename:
            self.predictions_save_filename = os.path.join(self.data_save_path, "{}_{}_predict.csv".format("-".join(self.outputs), self.classifier_type))

        with open(self.predictions_save_filename, "a") as f:
            df.to_csv(f, header=f.tell()==0, mode="a")
        return True


    def retrain_ai_model(self, classifier_type=None):
        if classifier_type: 
            self.classifier_type = classifier_type

        if not self.classifier_type:
            raise Exception("No classifier type specified")

        if self.classifier_type == "RF":
            self.classifier = self.generate_model_randomforest()
        elif self.classifier_type == "MLP":
            self.classifier = self.generate_model_mlp()
        else:
            self.classifier = None
            log.error("[{:12.12}] Invalid classifier '{}'".format(self.name_trunc, self.classifier_type))
            return None

        if not self.classifier:
            log.error("[{:12.12}] Model training failed".format(self.name_trunc))            
            return None

        if self.save_trained_model:
            self.get_ai_model_filename()
            dump(self.classifier, self.get_ai_model_filename())
            log.info("[{:12.12}] Model trained and saved to file '{}'".format(self.name_trunc, self.ai_model_filename))
        else:
            log.info("[{:12.12}] Model training completed".format(self.name_trunc))
        self.ai_model_retrain_ts = datetime.datetime.now()


    def convert_cyclic_to_sin_cos(self, cyc_series, total_periods=1440, shift=0):
        cyc_sin = np.sin((cyc_series + shift) * (2.*np.pi/total_periods))               # Normalised with number of mins in day
        cyc_cos = np.cos((cyc_series + shift) * (2.*np.pi/total_periods))

        return cyc_sin, cyc_cos


    def get_historical_data_for_item(self, item_name):
        if not item_name:
            log.error("No item name given. Aborting...")
            return

        log.debug("[{:12.12}] ---> Getting data for item '{}' from database...".format(self.name_trunc, item_name))    
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
                log.error("[{:12.12}] Failed to get data from server. Response: {}".format(self.name_trunc, response))
                log.error("[{:12.12}] --- URL: '{}'\n --- Query: '{}'\n --- Headers: '{}'".format(self.name_trunc, self.db_query_url, query, self.db_query_headers))
        
        except Exception as ex:
            log.error(ex)
            log.error(traceback.format_exc())
            return None


    def insert_calendar_item_inputs(self, df):
        col_offset = 0
        if DF_TIMESTAMP_COL_MONTH in self.inputs:
            df, new_col_count = self.insert_cyclic_cal_item_cols(df, df.index.month -1, 12, DF_TIMESTAMP_COL_MONTH, col_offset,)
            col_offset += new_col_count

        if (not self.inputs and DF_TIMESTAMP_COL_DOW in DEFAULT_INPUT_ITEMS) or DF_TIMESTAMP_COL_DOW in self.inputs:
            df, new_col_count = self.insert_cyclic_cal_item_cols(df, df.index.dayofweek, 7, DF_TIMESTAMP_COL_DOW, col_offset)
            col_offset += new_col_count

        if (not self.inputs and DF_TIMESTAMP_COL_TOD in DEFAULT_INPUT_ITEMS) or DF_TIMESTAMP_COL_TOD in self.inputs:
            df, new_col_count = self.insert_cyclic_cal_item_cols(df, df.index.hour * 60 + df.index.minute, 1440, DF_TIMESTAMP_COL_TOD, col_offset)

        return df


    def insert_cyclic_cal_item_cols(self, df, source_series, num_periods, col_name_prefix, insert_pos=0, keep_temp_col=False):
        """ Assume index column is a timestamp, which is then used for day of week, time of day etc """

         # Add columns for day of week. This is then converted into 'cyclic' format
        df.insert(insert_pos, col_name_prefix, source_series)

        # Map to cyclic format
        col_sin, col_cos = self.convert_cyclic_to_sin_cos(df[col_name_prefix], num_periods)  
        df.insert(insert_pos + 1, '{}_sin'.format(col_name_prefix), col_sin)               
        df.insert(insert_pos + 2, '{}_cos'.format(col_name_prefix), col_cos)
        
        if not keep_temp_col:
            df.drop(col_name_prefix, axis=1, inplace=True)
        new_cols = 3 if keep_temp_col else 2

        return df, new_cols # Returning number of new cols, in case we later use different 'cyclic' methods


    def get_historical_data_dataframe(self):    
        if not self.inputs:
            self.inputs = DEFAULT_INPUT_ITEMS

        log.info("[{:12.12}] Creating new model for input items '{}".format(self.name_trunc, self.inputs))
        log.info("[{:12.12}] Data grouped into time intervals of {}{} mins{}".format(self.name_trunc, Colour.BOLD, self.series_timeslot_mins, Colour.END))
        try:
            time_series = []
            for item in self.inputs:
                if item not in INTERNAL_DTM_ITEMS_LIST:
                    time_series.append(self.get_historical_data_for_item(item))
            
            for item in self.outputs:
                time_series.append(self.get_historical_data_for_item(item))
            log.info("[{:12.12}] Loaded {} time series data from database".format(self.name_trunc, len(time_series)))

            start = max([i.index.min() for i in time_series])
            end = min([i.index.max() for i in time_series])

            # Move start/end to beginning/end of respective interval periods
            start = start + pd.Timedelta(minutes=self.series_timeslot_mins - (start.minute % self.series_timeslot_mins))
            end = end + pd.Timedelta(minutes=end.minute % self.series_timeslot_mins)     

            log.info("[{:12.12}] Earliest common start: \t{}".format(self.name_trunc, start))
            log.info("[{:12.12}] Latest common end    : \t{}".format(self.name_trunc, end))

            # Create a new column, _time, divided into the required time slots (note we also have date here)
            # and then merge in the individual series previously loaded into corresponding time slots
            df = pd.DataFrame({'_time': pd.date_range(start,end,freq='{}T'.format(self.series_timeslot_mins))})
            for s in time_series:
                df = pd.merge_asof(df, s, on='_time')
            df = df.set_index("_time")
            df = self.insert_calendar_item_inputs(df)

            log.info("[{:12.12}] DataFrame created and merged. Shape: {}".format(self.name_trunc, df.shape))
            log.info("[{:12.12}] DataFrame columns: {}".format(self.name_trunc, ", ".join(df.columns.values)))
            log.debug(df)

            if self.save_training_data:
                if not self.training_data_filename:
                    self.training_data_filename = os.path.join(self.data_save_path, "{}_data.csv".format("-".join(self.outputs)))                    
                df.to_csv(self.training_data_filename)

            return df

        except Exception as ex:
            log.error(ex)        
            log.error(traceback.format_exc())
            if item: log.error("[{:12.12}] Error in processing series for '{}'".format(self.name_trunc, item))
            log.error("[{:12.12}] time_series[]:\n{}".format(self.name_trunc, time_series))
            return None


    def get_training_and_test_dataframes(self, test_size=0.2):
        """ Return separate dataframes for inputs and outputs, split into training and test data """

        try:
        
            df = self.get_historical_data_dataframe()   
            if df is None or df.empty:
                raise Exception("DataFrame of historical data returned None or empty")

            np.random.seed(7)           # fix random seed for reproducibility

            # Split data into training/test
            output_col = df.columns.get_loc(self.outputs[0])
            X = df.iloc[:, 0:output_col]
            y = df[self.outputs]

            log.debug("[{:12.12}] INPUT ITEMS: X ({}): {}".format(self.name_trunc, X.shape, X.columns.values))
            log.debug("[{:12.12}] OUTPUT ITEMS: y ({}): {}".format(self.name_trunc, y.shape, y.columns.values))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) # i.e. training size = 1 - test_size
            log.debug("[{:12.12}] Total set shape: {}. Training set row count: {}. Test set row count: {}".format(self.name_trunc, df.shape,len(X_train), len(X_test)))

            return X, y, X_train, X_test, y_train, y_test

        except Exception as ex:
            log.error(ex)
            if not X.empty: log.error("[{:12.12}] X ({}): {}".format(self.name_trunc, len(X), X))
            if not y.empty: log.error("[{:12.12}] y ({}): {}".format(self.name_trunc, len(y), y))
            log.error(traceback.format_exc())
            return None, None, None, None


    def generate_model_randomforest(self, n_estimators=200):    
        from sklearn.ensemble import RandomForestClassifier

        try:

            X, y, X_train, X_test, y_train, y_test = self.get_training_and_test_dataframes()

            # Create/train model
            self.classifier = RandomForestClassifier(n_estimators=n_estimators)    #Create a Gaussian Classifier
            log.debug("[{:12.12}] X_train: {}".format(self.name_trunc, X_train))
            log.debug("[{:12.12}] y_train: {}".format(self.name_trunc, y_train))
            
            y_train_values = y_train.values.ravel() if len(self.outputs)==1 else y_train

            self.classifier.fit(X_train,y_train_values)                 #Train the model using the training sets
            y_pred=self.classifier.predict(X_test)                      # Predict against the test data

            from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
            scores_test = metrics.accuracy_score(y_test, y_pred) * 100
            log.info("[{:12.12}] Model generated:".format(self.name_trunc)) 
            log.info("[{:12.12}] --- {}Test data accuracy: {:.2f}%{}".format(self.name_trunc,Colour.BOLD, scores_test, Colour.END))
            # log.info("[{:12.12}] Model generated. Accuracy with training data: {}{:.1f}%{}".format(self.name_trunc, Colour.BOLD, metrics.accuracy_score(y_test, y_pred) * 100), Colour.END)

            return self.classifier 

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
            log.debug("[{:12.12}] Training input column count: {}, Training set shape = {}".format(self.name_trunc, input_width, X_train.shape))
            
            # create 3 layer model. Assume number of neurons in 1st layer is double the input size, 2nd layer is 3/4 of 1st etc
            self.classifier = Sequential()
            self.classifier.add(Dense(input_width * 8, input_dim=input_width, activation='relu'))
            self.classifier.add(Dense(input_width * 4, activation='relu'))
            self.classifier.add(Dense(input_width * 2, activation='relu'))
            self.classifier.add(Dense(1, activation='sigmoid')) 
            
            self.classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       # Compile model
            
            self.classifier.fit(X, y, epochs=150, batch_size=16, verbose=0)                                   # Fit the model
            
            # evaluate the model
            scores_training = self.classifier.evaluate(X_train, y_train, verbose=0)
            log.info("Model generated:") 
            log.info(" --- Training data {}: {:.2f}%".format(self.classifier.metrics_names[1], scores_training[1]*100))
            scores_test = self.classifier.evaluate(X_test, y_test, verbose=0)
            log.info(" --- {}Test data {}: {:.2f}%{}".format(
                Colour.BOLD, self.classifier.metrics_names[1], scores_test[1]*100, Colour.END))

            return self.classifier


        except Exception as ex:
            log.error(ex)
            log.error(traceback.format_exc())
            return None

