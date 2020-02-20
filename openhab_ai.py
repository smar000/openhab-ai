import logger
import traceback
import configparser
import argparse

import datetime
import time
import os
import json
import traceback
import signal
import sys
from rule_model import Model, Models

#---------------------------------------------------------------------------------------------------
VERSION         = "0.3"
CONFIG_FILE     = "./openhab_ai.cfg"


log = logger.log
log.info("System started...")

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


parser = argparse.ArgumentParser(description='openHAB machine learning based item state predictor.')
parser.add_argument('-r', '--retrain', dest='retrain', default=False, action='store_true', help='Force re-train of the model on system startup')
COMMANDLINE_ARGS = parser.parse_args()

# Get any configuration overrides that may be defined in  CONFIG_FILE
# If override not specified, then use the defaults here

config = configparser.ConfigParser()
config.read(CONFIG_FILE)


# Configuration params
DB_SERVER                = getConfig(config,"Database", "influxdb_server_name", "influxdb")
DB_PORT                  = getConfig(config,"Database", "influxdb_server_port", 8086)
DB_NAME                  = getConfig(config,"Database", "influxdb_server_db", "openhab")
DB_RETENTION             = getConfig(config,"Database", "influxdb_server_retention", "autogen")

MODELS                   = json.loads(getConfig(config,"MachineLearning", "models", None))

INPUT_ITEMS              = getConfig(config,"MachineLearning", "input_items", None).replace(" ","").split(",")
OUTPUT_ITEMS             = getConfig(config,"MachineLearning", "output_items", None).replace(" ","").split(",")
TIME_PERIOD_MINUTES      = int(getConfig(config,"MachineLearning", "time_period_minutes", 10))
DAYS_BACK                = abs(int(getConfig(config,"MachineLearning", "days_back", 0)))
SAVE_TRAINING_DATA       = bool(getConfig(config,"MachineLearning", "save_training_data", "False"))


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

range_filter             = "" if DAYS_BACK==0  else "|> range(start: -{}d) ".format(DAYS_BACK)
DB_QUERY_BASE            = 'from(bucket: "{}/{}") {}|> filter(fn: (r) => r._measurement == "<<>>")'.format(DB_NAME, DB_RETENTION, range_filter)



def check_retrain_model():
    ''' Rebuild model if required based on rebuild time settings '''
    if RETRAIN_MODEL_TIME and RETRAIN_MODEL_TIME != "0000" and RETRAIN_MODEL_TIME.isdigit():
        n = datetime.datetime.today()
        retrain_at_dtm = datetime.datetime(n.year, n.month, n.day, RETRAIN_MODEL_TIME[:2], RETRAIN_MODEL_TIME[2:4])

        for k, m in models:
            if retrain_at_dtm < m.ai_model_retrain_ts:
                log.info("'{}' rule model retrain triggerred by 'RETRAIN_MODEL_TIME' setting of {}".format(m.name, RETRAIN_MODEL_TIME))
                m.retrain_model()


def signal_handler(sig, frame):
    if models:
        models.stop_all_sse()

    print('Done.')
    sys.exit(0)

#----------------------------------------------------------------------

signal.signal(signal.SIGINT, signal_handler)        # Trap CTL-C
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'            # Ignore the various TensorFlow CPU support warnings


log.debug("DB_QUERY_URL: {}".format(DB_QUERY_URL))
log.debug("DB_QUERY_BASE: {}".format(DB_QUERY_BASE))

log.info("{} openHAB AI rule models defined:".format(len(MODELS)))
for k in MODELS:
    log.info(" --- {}: {} -> {}".format(k, MODELS[k]["inputs"], MODELS[k]["outputs"]))

if not (DB_QUERY_URL and DB_QUERY_BASE and INPUT_ITEMS and OUTPUT_ITEMS):
    log.info("Database and/or input/output items configuration missing in file '{}'".format(CONFIG_FILE))
    sys.exit(0)

try:
    models = Models()
    models.load_collection_from_dict(MODELS)

    for k, m in models.items():
        m.model_save_path = MODELS_FOLDER
        m.db_query_base = DB_QUERY_BASE
        m.db_query_url = DB_QUERY_URL
        m.db_query_headers = DB_QUERY_HEADERS
        m.series_timeslot_mins = TIME_PERIOD_MINUTES
        m.openhab_url = OPENHAB_URL
        m.retrain_ai_model(CLASSIFIER_TYPE)        

        if not m.classifier:
            raise Exception("Model {} has no classifier".format(m.name))

        m.subscribe_to_openhab()

except Exception as ex:
    log.error(ex)
    log.error(traceback.format_exc())
    sys.exit(0)


try:
    while True:
        check_retrain_model()                       # Check if we have to retrain the model
        models.do_predict_all_models()              # Do prediction 
        time.sleep(60)                              # Repeat every 60s, as some triggers may be time dependent and not just item state driven 

except Exception:
    log.error(traceback.format_exc())

if models:
    models.stop_all_sse()

    
    
    

