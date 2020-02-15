import logging
import traceback
import configparser
import argparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import csv
import requests
import time
import datetime
import pandas as pd
import numpy as np
import io
import os
import json
from joblib import dump, load
from sse import SSEReceiver
import traceback
import signal
import sys

#---------------------------------------------------------------------------------------------------
VERSION         = "0.2"
CONFIG_FILE     = "./openhab_ai.cfg"

# --- Configs/Default
def getConfig(config,section,name,default):
    if config.has_option(section,name):
        # log.info("{}/{}: Returning {}".format(section, name, config.get(section,name)))
        return config.get(section,name)
    else:
        # log.info("Section/value {}/{} not found. Returning default value {}".format(section, name, default))
        return default

def strip_right_slash(string):
    return string[:len(string)-1] if string[-1] =="/" else string #Strip right '/' if any

# Get any configuration overrides that may be defined in  CONFIG_FILE
# If override not specified, then use the defaults here

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

log = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)-10s |%(levelname)-5s| %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log.info("System started...")
# parser = argparse.ArgumentParser(description='openHAB machine learning rule engine.')
# parser.add_argument('refresh', action="store_true", type=bool, default=False,
#                    help='Regenerate model')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                    const=sum, default=max,
#                    help='sum the integers (default: find the max)')

# args = parser.parse_args()

# Configuration params
DB_SERVER                = getConfig(config,"Database", "influxdb_server_name", "influxdb")
DB_PORT                  = getConfig(config,"Database", "influxdb_server_port", 8086)
DB_NAME                  = getConfig(config,"Database", "influxdb_server_db", "openhab")
DB_RETENTION             = getConfig(config,"Database", "influxdb_server_retention", "autogen")

INPUT_ITEMS              = getConfig(config,"MachineLearning", "input_items", None).split(",")
OUTPUT_ITEMS             = getConfig(config,"MachineLearning", "output_items", None).split(",")
TIME_PERIOD_MINUTES      = int(getConfig(config,"MachineLearning", "time_period_minutes", 10))
CLASSIFIER_TYPE          = getConfig(config,"MachineLearning", "classifier", "RF")
OPENHAB_URL              = strip_right_slash(getConfig(config,"openHAB", "openhab_url", "http://openhab:7070"))

RETRAIN_MODEL_TIME       = getConfig(config,"openHAB", "RETRAIN_MODEL_TIME", "0000")
OPENHAB_COMMAND_ITEM     = getConfig(config,"openHAB", "openhab_command_item", None)
OPENHAB_SEND_PREDICTIONS = bool(getConfig(config,"openHAB", "openhab_send_predictions", "False"))

MODELS_FOLDER            = getConfig(config,"MachineLearning", "models_folder", "./").replace('"','')
PREDICTIONS_FILE_FOLDER  = getConfig(config,"MachineLearning", "predictions_file_folder", "").replace('"','')
PREDICTIONS_FILE_NAME    = os.path.join(PREDICTIONS_FILE_FOLDER, "{}_predictions.csv".format("-".join(OUTPUT_ITEMS))) if PREDICTIONS_FILE_FOLDER else None

DB_QUERY_URL             = "http://{}{}/api/v2/query".format(DB_SERVER, ":{}".format(DB_PORT) if DB_PORT else "")
DB_QUERY_HEADERS         = {"accept" : "application/csv", "content-type" : "application/vnd.flux"}
DB_QUERY_BASE            = 'from(bucket: "{}/{}") |> range(start: -30d) |> filter(fn: (r) => r._measurement == "<<>>")'.format(DB_NAME, DB_RETENTION)


# Internal: DataFrame column names when reducing event datetime to day of week and mins from midnight
DF_TIMESTAMP_COL_DOW     = "dayOfWeek"        # TODO! Look at one hot encoding instead...
DF_TIMESTAMP_COL_MINS    = "minsFromMidnight" # TODO! Look at one hot encoding instead...


# DB_QUERY_URL = getConfig(config,"Database", "db_query_url", None)
# DB_QUERY_HEADERS = json.loads(getConfig(config,"Database", "db_query_headers", {'accept':'application/csv', 'content-type':'application/vnd.flux'}))
# DB_QUERY_BASE = getConfig(config,"Database", "db_query_base", None) 

# Globals
last_model_rebuild       = None     # datetime of last model rebuild
clf                      = None     # Classifier
last_df                  = None     # Last dataframe
model_file_name          = None     # File name for current model

#-------------------------
def write_df_to_file(df):
    if not PREDICTIONS_FILE_NAME:
        return

    with open(PREDICTIONS_FILE_NAME, "a") as f:
        df.to_csv(f, header=f.tell()==0, mode="a")


def get_historical_data_for_item(item_name):
    if not item_name:
        log.error(" No item name given. Aborting...")
        return

    log.info("\t-> Calling DB: {}".format(item_name))    
    query = DB_QUERY_BASE.replace("<<>>",item_name)
    try:
        response = requests.post(DB_QUERY_URL, data=query, headers=DB_QUERY_HEADERS)
        # log.info("Response: {}, Text: {}".format(response, response.text))
        if response.status_code == 200:
            time_series = pd.read_csv(io.StringIO(response.content.decode('utf-8')), 
                usecols=[5,6], names=["_time", item_name], 
                header=3, parse_dates=[0], index_col=0, squeeze=True)
            return time_series
        else:
            log.error("Failed to get data from server. Response: {}".format(response))
            log.error(" --- URL: '{}'\n --- Query: '{}'\n --- Headers: '{}'".format(DB_QUERY_URL, query, DB_QUERY_HEADERS))
    
    except Exception as ex:
        log.error(ex)
        log.error(traceback.format_exc())
        return None


def get_dataframe_historical_data():    
    time_series = []
    log.info("Creating new model for INPUT_ITEMS items '{}".format(INPUT_ITEMS))
    for item in INPUT_ITEMS:
        time_series.append(get_historical_data_for_item(item))
    log.info("Getting data for OUTPUT_ITEMS items:")
    for item in OUTPUT_ITEMS:
        time_series.append(get_historical_data_for_item(item))

    log.info("Loaded {} time_series".format(len(time_series)))
    # print(time_series)

    start = max([i.index.min() for i in time_series])
    end = min([i.index.max() for i in time_series])

    # Move start/end to beginning/end of respective interval periods
    start = start + pd.Timedelta(minutes=TIME_PERIOD_MINUTES - (start.minute % TIME_PERIOD_MINUTES))
    end = end + pd.Timedelta(minutes=end.minute % TIME_PERIOD_MINUTES)     

    log.info("Earliest common start: \t{}".format(start))
    log.info("Latest common end    : \t{}".format(end))

    df = pd.DataFrame({'_time': pd.date_range(start,end,freq='{}T'.format(TIME_PERIOD_MINUTES))})
    for s in time_series:
        df = pd.merge_asof(df, s, on='_time')
    df = df.set_index("_time")

    # Add columns for day of week and time period in minutes
    df[DF_TIMESTAMP_COL_DOW] = df.index.dayofweek
    mins = []
    for d in df.index:
      mins.append(d.hour * 60 + d.minute)
    df[DF_TIMESTAMP_COL_MINS] = mins

    # Reorder columns so day/time period on left and output cols on right
    org_cols = list(df.columns)
    [org_cols.remove(o) for o in OUTPUT_ITEMS]
    org_cols.remove(DF_TIMESTAMP_COL_DOW)
    org_cols.remove(DF_TIMESTAMP_COL_MINS)

    new_cols = [DF_TIMESTAMP_COL_DOW, DF_TIMESTAMP_COL_MINS]
    new_cols += org_cols
    [new_cols.append(o) for o in OUTPUT_ITEMS]
    # log.info("new cols: {}".format(new_cols))
    df = df[new_cols]

    log.info("DataFrame created and merged. Shape: {}".format(df.shape))
    log.debug(df)
    model_file_name = os.path.join(MODELS_FOLDER, "{}_data.csv".format("-".join(OUTPUT_ITEMS)))
    df.to_csv(model_file_name)
    return df


def generate_model_randomforest(n_estimators=200):
    df = get_dataframe_historical_data()

    # Split data into training/test
    np.random.seed(7)           # fix random seed for reproducibility

    try:
        X = df[[DF_TIMESTAMP_COL_DOW, DF_TIMESTAMP_COL_MINS] + INPUT_ITEMS]
        y = df[OUTPUT_ITEMS]

        log.debug("INPUT ITEMS: X ({}): {}".format(X.shape, X))
        log.debug("OUTPUT ITEMS: y ({}): {}".format(y.shape, y))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% training and 30% te
        log.debug("Total set shape: {}. Training set row count: {}. Test set row count: {}".format(df.shape,len(X_train), len(X_test)))

        # Create/train model
        clf=RandomForestClassifier(n_estimators=n_estimators)    #Create a Gaussian Classifier
        log.debug("X_train: {}".format(X_train))
        log.debug("y_train: {}".format(y_train))
        
        y_train_values = y_train.values.ravel() if len(OUTPUT_ITEMS)==1 else y_train

        clf.fit(X_train,y_train_values)                 #Train the model using the training sets
        y_pred=clf.predict(X_test)                      # Predict against the test data

        # from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
        # log.info("Model generated. Accuracy with training data: {:.1f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))

        return clf

    except Exception as ex:
        log.error(ex)
        if not X.empty: log.error("X ({}): {}".format(len(X), X))
        if not y.empty: log.error("y ({}): {}".format(len(y), y))
        log.error(traceback.format_exc())
        return None


def generate_model_mlp(n_estimators=200):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import model_from_json

    # fix random seed for reproducibility
    np.random.seed(7)

    df = get_dataframe_historical_data()

    # Split data into training/test
    X = df[[DF_TIMESTAMP_COL_DOW, DF_TIMESTAMP_COL_MINS] + INPUT_ITEMS]
    y = df[OUTPUT_ITEMS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% training and 30% te
    log.info("[Total set: {}] Training:Test =  {}:{}".format(df.shape,len(X_train), len(X_test)))

    # create model
    model = Sequential()
    model.add(Dense(5, input_dim=len(INPUT_ITEMS), activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       # Compile model
    model.fit(X, y, epochs=150, batch_size=16, verbose=0)                                   # Fit the model
    
    # evaluate the model
    scores_training = model.evaluate(X_train, y_train, verbose=0)
    log.info("Model generated. Accuracy with training data: %s: %.2f%%" % (model.metrics_names[1], scores_training[1]*100))

    return model


def sse_event_callback(event):
    # log.info("Callback function called with event: {}".format(event))
    state_event = json.loads(event["data"])
    # print(state_event)
    state_str = json.loads(state_event["payload"])["value"] if "payload" in state_event  else "-"    
    item_name = state_event["topic"].replace("smarthome/items/","").replace("/statechanged","").replace("/state","") if "topic" in state_event else "-"

    do_predict(item_name, state_str)


def get_openhab_states_for(items_list):
    states = {}
    for item_name in items_list:
        states[item_name] = get_openhab_state_for_item(item_name)
    return states


def get_openhab_state_for_item(item_name):
    rest_url = "{}/rest/items/{}/state".format(OPENHAB_URL, item_name)
    response = requests.get(rest_url)            
    curr_state = response.text if response.status_code == 200 else None
    curr_state = convert_state_to_num(curr_state)
    return curr_state


def convert_state_to_num(state):
    if state == "OFF":
        state = 0
    elif state == "ON":
        state = 1
    # Otherwise just return the state as is
    return state


def do_predict(item_name=None, state=None, log_prefix=""):
    '''
        Do prediction. If item_name/state supplied, no need to look up values for this. 
        All other input items looked up from openHAB, before doing prediction.
    '''

    # state = 1 if state == "ON" else 0
    log_prefix = "[{}] ".format(log_prefix) if log_prefix else ""
    # print(state_event)

    now = datetime.datetime.now()
    df = get_df_for_inputs_current_states(item_name, state)
    # log.debug("DF for current state:\n{}".format(df))

    if df is not None and not df.empty:
        try:
            y_pred=clf.predict(df)

            # Get current input and output items states for comparison
            curr_input_states = get_openhab_states_for(INPUT_ITEMS)
            curr_output_states = get_openhab_states_for(OUTPUT_ITEMS)
            # print("type(y_pred): {}, y_pred: {}".format(type(y_pred), y_pred))
            # y_pred_string = ", ".join(y_pred.tolist()) # if type(y_pred) == np.ndarray else y_pred
            predicted_states = {}
            count = 0
            for item_name in OUTPUT_ITEMS:                
                predicted_state = y_pred if y_pred.size==1 else y_pred[0][count]                
                predicted_states[item_name] = "'{}' -> [{}]".format(curr_output_states[item_name], predicted_state)
                count += 1

            log.debug("{}Input items states : {}'".format(log_prefix, curr_input_states))
            log.debug("{}Output items states:'{}'".format(log_prefix, predicted_states))
            
            # Create DataFrame for the whole row of inputs/outputs,
            # - for pd concat, we need indexes to be aligned. y_pred does not have an index col. Add the timestamp
            #   used in the df
            full_df_row = pd.concat([df, pd.DataFrame(y_pred, columns=OUTPUT_ITEMS, index=[df.index[0]])], axis=1)
            
            global last_df        

            # Get any changes to previous dataframe
            if last_df is not None:     
                if not np.array_equal(last_df.values, full_df_row.values):
                    # log.info("openHAB item state change triggered:")
                    for (item_name, value) in full_df_row.iteritems():      # value is pandas Series
                        if item_name in last_df.columns:
                            last_value = last_df[item_name][0]              # last_df[item_name][0] returns a numpy.int64
                            current_value = value.iloc[0]                    # the numpy.int64 value in location 0 of the Series
                            # print("item_name: '{}', value: '{}', value type: '{}', current_value: '{}', last_value type: {}".format(
                            #     item_name, value, type(value), current_value, type(last_value)))                            
                            in_out = "INP" if (item_name in INPUT_ITEMS or item_name in DF_TIMESTAMP_COL_DOW or item_name in DF_TIMESTAMP_COL_MINS) else "OUT"
                            if current_value != last_value:
                                log.info(" --- [{}] {:<25} [{}] -> [{}]".format(in_out, item_name, last_value, current_value))
                        # else:
                        #     log.error("Column '{}' not found in the last_df:\n{}".format(item_name, last_df))                
                # else:
                #     log.debug("No changes detected")
            else:
                log.info("Input/output item states: ")
                for i in range(full_df_row.shape[1]): # iterate over all columns
                    col_name = full_df_row.columns[i]
                    log.info(" --- {:<25} -> [{}]".format(col_name, full_df_row[col_name][0]))
                log.info("")

                # log.info("Delta_DF: {}".format(full_df_row))
            last_df = full_df_row
            write_df_to_file(full_df_row)          

        except Exception as ex:
            log.error(ex)
            log.error("DataFrame: {}".format(df))
            log.error(traceback.format_exc())            

    else:
        log.error("Prediction failed as DataFrame of current input item states not obtained")


def get_df_for_inputs_current_states(override_item_name=None, override_item_state=None):
    """
        Generate a DataFrame of the current states of the INPUT_ITEMS obtained from the openHAB server, 
        along with day/time period column values
    """       
    input_item_states = {}
    
    for item_name in INPUT_ITEMS:                    # Get updated events (for items other than override_item_name)
        if not override_item_name or item_name != override_item_name:
            rest_url = "{}/rest/items/{}/state".format(OPENHAB_URL, item_name)
            response = requests.get(rest_url)
            if response.status_code == 200:
                state = convert_state_to_num(response.text)

                input_item_states[item_name] = state
            else:
                log.error("Invalid response code from openHAB REST api for item '{}': {} {}".format(item_name, response.status_code, response.text))

    if override_item_name:
        if type(override_item_state) == str: override_item_state = convert_state_to_num(override_item_state)
        input_item_states[override_item_name] = override_item_state
    
    if input_item_states:    
        now = datetime.datetime.now()
        # round to the next period as used in the training model
        period_mins = now.minute + (TIME_PERIOD_MINUTES - now.minute % TIME_PERIOD_MINUTES)
        mins_midnight = now.hour * 60 + period_mins           

        new_data = {
            # "_time"                 : pd.Timestamp.now(), 
            DF_TIMESTAMP_COL_DOW    : datetime.datetime.today().weekday(), 
            DF_TIMESTAMP_COL_MINS   : mins_midnight, 
        }
        new_data.update(input_item_states)  # merge the dicts               
        df = pd.DataFrame(new_data, index=[pd.Timestamp.now()]) #index=[0]) # index=["_time"]) #
        # df.set_index("_time")

        return df
    else:
        log.error("No input item states obtainable from openHAB")
        return None


def post_to_openhab(item_name, new_state):
    if OPENHAB_SEND_PREDICTIONS:
        log.info("Posting '{}' to openHAB item '{}'".format(new_state, item_name))
        
    # curl -X POST --header "Content-Type: text/plain" --header "Accept: application/json" -d "Test" "http://openhab:7070/rest/items/Voice_Command"
    headers = {"accept" : "application/json", "content-type" : "text/plain"}
    url = "{}/rest/items/{}".format(OPENHAB_URL,item_name)
    response = requests.post(url, data=new_state, headers=headers)
    log.debug("Posted '{}' to openHAB for item '{}'. Reponse: '{}'".format(new_state, item_name, response))


def load_model_from_file(model_file_name):
    log.info("Loading existing model '{}'".format(model_file_name))

    global clf
    clf = load(model_file_name)

    # Assume last modified time for the model is the model generation time
    last_modified_ts = os.path.getmtime(model_file_name)        

    global last_model_rebuild
    last_model_rebuild = datetime.datetime.fromtimestamp(last_modified_ts) 

    log.info("Model loaded from file (last trained {:%H:%M %Y-%m-%d})".format(last_model_rebuild))        


def retrain_model():
    global clf
    if CLASSIFIER_TYPE == "RF":
        clf = generate_model_randomforest()
    elif CLASSIFIER_TYPE == "MLP":
        clf = generate_model_mlp()
    else:
        clf = None
        log.error("Invalid classifier '{}'".format(CLASSIFIER_TYPE))
        return None

    if not clf:
        log.error("Model training failed. Exiting...")
        sys.exit(0)

    dump(clf, model_file_name)
    log.info("Model trained and saved to file '{}'".format(model_file_name))

    global last_model_rebuild
    last_model_rebuild = datetime.datetime.now()


def check_retrain_model():
    ''' Rebuild model if required based on rebuild time settings '''

    if RETRAIN_MODEL_TIME and RETRAIN_MODEL_TIME != "0000" and RETRAIN_MODEL_TIME.isdigit():
        n = datetime.datetime.today()
        retrain_at_dtm = datetime.datetime(n.year, n.month, n.day, RETRAIN_MODEL_TIME[:2], RETRAIN_MODEL_TIME[2:4])

        if retrain_at_dtm < last_model_rebuild:
            log.info("Model rebuild triggerred by 'RETRAIN_MODEL_TIME' setting of {}".format(RETRAIN_MODEL_TIME))
            retrain_model()


def get_model_filename():    
    return os.path.join(MODELS_FOLDER, "{}_{}.joblib".format(CLASSIFIER_TYPE, "-".join(OUTPUT_ITEMS)))


def signal_handler(sig, frame):
    if sse:
        sse.stop()

    print('Done.')
    sys.exit(0)

#----------------------------------------------------------------------
# Trap CTL-C
signal.signal(signal.SIGINT, signal_handler)


log.debug("DB_QUERY_URL: {}".format(DB_QUERY_URL))
log.debug("DB_QUERY_BASE: {}".format(DB_QUERY_BASE))
log.info("INPUT_ITEMS: {}".format(INPUT_ITEMS))
log.info("OUTPUT_ITEMS: {}".format(OUTPUT_ITEMS))

if not (DB_QUERY_URL and DB_QUERY_BASE and INPUT_ITEMS and OUTPUT_ITEMS):
    log.info("Database and/or input/output items configuration missing in file '{}'".format(CONFIG_FILE))
    sys.exit(0)

try:
    model_file_name = get_model_filename()
    if os.path.exists(model_file_name):
        load_model_from_file(model_file_name)
    else:
        retrain_model()
    
    if not clf:
        sys.exit(0)

except Exception as ex:
    log.error(ex)
    log.error(traceback.format_exc())
    sys.exit(0)


# Listen to openHAB events via SSE. Also monitor openhab_command_item if it has been defined
if OPENHAB_COMMAND_ITEM:
    topics = [OPENHAB_COMMAND_ITEM]
else:
    topics = []

for item in INPUT_ITEMS:
    topics.append("smarthome/items/{}/state".format(item)) #statechanged
url = "{}/rest/events?topics={}".format(OPENHAB_URL, ",".join(topics))
log.debug("openHAB SSE url: {}".format(url))

sse = SSEReceiver(url, sse_event_callback)
sse.start()

log.info("Subscribed to openHAB events for the input items")

try:
    while True:
        check_retrain_model()                       # Check if we have to retrain the model
        do_predict()                                # Do prediction 
        time.sleep(60)                              # Repeat every 60s, as some triggers may be time dependent and not just item state driven 

except Exception:
    log.error(traceback.format_exc())

sse.stop()

    
    
    

