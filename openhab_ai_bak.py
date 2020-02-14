import csv
import requests
import time
import datetime
import pandas as pd
import io

url = 'http://ifserver:8086/api/v2/query'
headers = {'accept':'application/csv', 'content-type':'application/vnd.flux'}
query_base = 'from(bucket: "openhab/two_years") |> range(start: -7d) |> filter(fn: (r) => r._measurement == "<<>>")'

INPUT_ITEMS = ["CH_LivingRoom_HeatDemand1"]
OUTPUT_ITEMS = ["CH_Boiler_Relay"]

TIME_PERIOD_MINUTES = 10

def get_data_for_item(item_name):
    query = query_base.replace("<<>>",item_name)
    response = requests.post(url, data=query, headers=headers)
    # print("Response: {}".format(response))
    if response.status_code == 200:
        # print("Text: \n{}".format(response.text))
        
        # df = pd.read_csv(io.StringIO(s.decode('utf-8')), usecols=[5,6])
        # #df.drop([0,1,2], axis=0)

        time_series = pd.read_csv(io.StringIO(response.content.decode('utf-8')), 
            usecols=[5,6], names=["_time", item_name], 
            header=3, parse_dates=[0], index_col=0, squeeze=True)
        # [item_name if x=='_value' else x for x in time_series.columns]
        return time_series

    #     csv_data = response.text.split("\r\n")
    #     first_row = csv_data.index(',result,table,_start,_stop,_time,_value,_field,_measurement') + 1
    #     csv = []
    #     for i in range(first_row, len(csv_data) - 1):
    #         row = csv_data[i].split(",")
    #         # print(row)
    #         if len(row) > 6:
    #             dtm = datetime.datetime.strptime(row[5][:19], "%Y-%m-%dT%H:%M:%S")
    #             csv.append([dtm, row[6]])
    #             # print("Row {}: {}".format(i, [row[5], row[6]]))
    # return csv                # time.sleep(0.5)

time_series = []
for item in INPUT_ITEMS:
    time_series.append(get_data_for_item(item))
for item in OUTPUT_ITEMS:
    time_series.append(get_data_for_item(item))


print("{} time_series loaded".format(len(time_series)))
# arr = [i.index.min() for i in time_series]

start = max([i.index.min() for i in time_series])
end = min([i.index.max() for i in time_series])

# Move start/end to beginning/end of respective interval periods
start = start + pd.Timedelta(minutes=TIME_PERIOD_MINUTES - (start.minute % TIME_PERIOD_MINUTES))
end = end + pd.Timedelta(minutes=end.minute % TIME_PERIOD_MINUTES)     

print("Start: {}, End: {}".format(start, end))
df_time = pd.DataFrame({'_time': pd.date_range(start,end,freq='{}T'.format(TIME_PERIOD_MINUTES))})

df = df_time
for s in time_series:
    df = pd.merge_asof(df, s, on='_time')
# print("df: {}".format(df))

# df = pd.merge_asof(pd.merge_asof(df_time, s1, on='_time'), s2, on='_time', suffixes=('_BoilerRelay', '_OutsideT'))
df = df.set_index("_time")

print(df)

# start = max(s1.index.min(), s2.index.min())

# s1 = get_data_for_item("CH_LivingRoom_HeatDemand1")
# s2 = get_data_for_item("CH_Boiler_Relay")

# start = max(s1.index.min(), s2.index.min())
# end = min(s1.index.max(), s2.index.max())
# df_time = pd.DataFrame({'_time': pd.date_range(start,end,freq='10min')})
# df = pd.merge_asof(pd.merge_asof(df_time, s1, on='_time'), s2, on='_time', suffixes=('_BoilerRelay', '_OutsideT'))
# df = df.set_index("_time")

# print(df)
# print("Has NaN values: {}".format(df.isnull().values.any()))
# print(type(df.index))


# data = {}
# for row in csv1:
#     dtm = row[0]
#     period_end_dtm = dtm + datetime.timedelta(minutes=10-(dtm.minute%10))
#     period_key = period_end_dtm.strftime("%Y%m%d %H:%M")
#     if period_key in data:
#         pass
#     else:
#         data[period_key] = row[1]

# df1 = pd.DataFrame(csv1)
# df1.columns = ["DateTime", "CH_LivingRoom_HeatDemand1"]
# df1.set_index("DateTime", inplace=True)

# df2 = pd.DataFrame(csv2)
# df2.columns = ["DateTime", "CH_Boiler_Relay"]
# df2.set_index("DateTime", inplace=True)

# merged = df1.join(df2, how="outer")
# merged.fillna(method="ffill")       # Fill with last NaN value
# merged.fillna(0,inplace=True)       # Fill any remaining NaNs - e.g. first rows - with 0

# print(merged)

# for period_end in data:
#     print("{} {}".format(period_end, data[period_end]))




    # print("{}: {}\t{}".format(item_name, row[0], row[1]))

    

        

