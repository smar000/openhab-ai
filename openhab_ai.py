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

# Get any configuration overrides that may be defined in  CONFIG_FILE
# If override not specified, then use the defaults here

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

log = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)-10s |%(levelname)-5s| %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
# ch = logging.StreamHandler()
# ch.setLevel(level=logging.INFO)
# ch.setFormatter(formatter)
# log.addHandler(ch)

log.info("System started...")
# parser = argparse.ArgumentParser(description='openHAB machine learning rule engine.')
# parser.add_argument('refresh', action="store_true", type=bool, default=False,
#                    help='Regenerate model')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                    const=sum, default=max,
#                    help='sum the integers (default: find the max)')

# args = parser.parse_args()

DB_QUERY_URL = getConfig(config,"Database", "DB_QUERY_URL", None)
DB_QUERY_HEADERS = json.loads(getConfig(config,"Database", "DB_QUERY_HEADERS", {'accept':'application/csv', 'content-type':'application/vnd.flux'}))
DB_QUERY_BASE = getConfig(config,"Database", "DB_QUERY_BASE", None) 
INPUT_ITEMS = getConfig(config,"MachineLearning", "INPUT_ITEMS", None).split(",")
OUTPUT_ITEMS = getConfig(config,"MachineLearning", "OUTPUT_ITEMS", None).split(",")
TIME_PERIOD_MINUTES = int(getConfig(config,"MachineLearning", "TIME_PERIOD_MINUTES", 10))
OPENHAB_URL = getConfig(config,"openHAB", "OPENHAB_URL", "http://openhab:7070")
MODELS_FOLDER = getConfig(config,"MachineLearning", "MODELS_FOLDER", "./").replace('"','')

PREDICTIONS_FILE_FOLDER = getConfig(config,"MachineLearning", "PREDICTIONS_FILE_FOLDER", "").replace('"','')
PREDICTIONS_FILE_NAME = os.path.join(PREDICTIONS_FILE_FOLDER, "{}_predictions.csv".format("-".join(OUTPUT_ITEMS))) if PREDICTIONS_FILE_FOLDER else None

DF_TIMESTAMP_COL_DOW = "dayOfWeek"
DF_TIMESTAMP_COL_MINS = "minsFromMidnight"

OPENHAB_URL = OPENHAB_URL[:len(OPENHAB_URL)-1] if OPENHAB_URL[-1] =="/" else OPENHAB_URL #Strip right '/' if any


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
    response = requests.post(DB_QUERY_URL, data=query, headers=DB_QUERY_HEADERS)
    # log.info("Response: {}, Text: {}".format(response, response.text))
    if response.status_code == 200:
        time_series = pd.read_csv(io.StringIO(response.content.decode('utf-8')), 
            usecols=[5,6], names=["_time", item_name], 
            header=3, parse_dates=[0], index_col=0, squeeze=True)
        return time_series
    else:
        log.error(" Response: {}".format(response))


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

    log.info("Earliest common start: \t{}\nLatest common end: \t{}".format(start, end))

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

        from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
        # log.info("Model generated. Accuracy with training data: {:.1f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))

        return clf

    except Exception as ex:
        log.error(ex)
        if not X.empty: log.error("X ({}): {}".format(len(X), X))
        if not y.empty: log.error("y ({}): {}".format(len(y), y))
        log.error(traceback.format_exc())
        return None


def generate_model_nn(n_estimators=200):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import model_from_json

    # fix random seed for reproducibility
    numpy.random.seed(7)

    df = get_dataframe_historical_data()

    # Split data into training/test
    X = df[[DF_TIMESTAMP_COL_DOW, DF_TIMESTAMP_COL_MINS] + INPUT_ITEMS]
    y = df[OUTPUT_ITEMS]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% training and 30% te
    log.info("[Total set: {}] Training:Test =  {}:{}".format(df.shape,len(X_train), len(X_test)))

    # create model
    model = Sequential()
    model.add(Dense(5, input_dim=3, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])       # Compile model
    model.fit(X, Y, epochs=150, batch_size=16, verbose=0)                                   # Fit the model
    
    # evaluate the model
    scores_training = model.evaluate(X_train, y_train, verbose=0)
    log.info("Model generated. Accuracy with training data: %s: %.2f%%" % (model.metrics_names[1], scores_training[1]*100))

    return model


def sse_event_callback(event):
    # log.info("Callback function called with event: {}".format(event))
    state_event = json.loads(event["data"])
    # pprint.pprint(state_event)
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
                predicted_states[item_name] = "{} -> {}".format(curr_output_states[item_name], predicted_state)
                count += 1

            log.info("{}Input items states : {}'".format(log_prefix, curr_input_states))
            log.info("{}Output items states:'{}'".format(log_prefix, predicted_states))

            # Create DataFrame for the whole row of inputs/outputs,
            # - for pd concat, we need indexes to be aligned. y_pred does not have an index col. Add the timestamp
            #   used in the df
            df_full_row = pd.concat([df, pd.DataFrame(y_pred, columns=OUTPUT_ITEMS, index=[df.index[0]])], axis=1)
            # df_full_row.dropna(inplace=True)
            write_df_to_file(df_full_row)


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
        # else:
        #     log.debug("ignoring item: {} (override_item_name: {}), override_item state {}. item_name != override_item_name: {}".format(item_name, 
        #         override_item_name, 
        #         override_item_state, 
        #         item_name != override_item_name))

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
    exit

try:
    model_file_name = os.path.join(MODELS_FOLDER, "{}.joblib".format("-".join(OUTPUT_ITEMS)))
    if os.path.exists(model_file_name):
        log.info("Loading existing model '{}'".format(model_file_name))
        clf = load(model_file_name)
        log.info("Model loaded")
    else:
        clf = generate_model_randomforest()
        if not clf:
            log.error("Model generation failed. Exiting...")
            sys.exit(0)

        dump(clf, model_file_name)
        log.info("Model saved to file '{}'".format(model_file_name))
except Exception as ex:
    log.error(ex)
    log.error(traceback.format_exc())
    sys.exit(0)


# Listen to openHAB events via SSE 
topics = []
for item in INPUT_ITEMS:
    topics.append("smarthome/items/{}/state".format(item)) #statechanged
url = "{}/rest/events?topics={}".format(OPENHAB_URL, ",".join(topics))
log.debug("openHAB SSE url: {}".format(url))

sse = SSEReceiver(url, sse_event_callback)
log.info("Subscribed to openHAB SSE")
try:
    sse.start()
    rest_url = "{}/rest/items/{}/state".format(OPENHAB_URL, INPUT_ITEMS[0])
    while True:
        time.sleep(60)                              # Do prediction every 60s, as some may be time dependent and not just item state driven 
        for item_name in INPUT_ITEMS:                    # Get updated events
            response = requests.get(rest_url)
            if response.status_code == 200:
                do_predict(item_name, response.text)
            else:
                log.error("State update of '{}' from openHAB returned error '{}'".format(item, response.status_code))


except Exception:
    log.error(traceback.format_exc())

sse.stop()

    
    
    

