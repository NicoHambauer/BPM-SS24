from datetime import datetime

import pandas as pd
import numpy as np


datasets = {
    #
    1: {"dataset": "helpdesk_raw"}, # no additional params
    2: {"dataset": "bpi2020r_raw"},
    3: {"dataset": "bpi2020PrepaidTravelCost"},
    4: {"dataset": "bpi2013i_raw"},
    5: {"dataset": "sepsis_raw"},
    6: {"dataset": "bpi2012w_complete_raw"},

    7: {"dataset": "nasa"},
    8: {"dataset": "bpi2019"},
}

df_case_name = 'case_id'
df_activity_name = 'activity'
dataset_dir = "../../../../datasets/"
output_dir = "../"
for dataset_i in datasets:
    dataset_name = datasets[dataset_i]["dataset"]
    file = dataset_dir + dataset_name + ".csv"
    output = output_dir + dataset_name + "_characteristics" + ".txt"
    with open(output, 'w') as f:
        df = pd.read_csv(file, delimiter=';')
        #df_groups = df.groupby([df_case_name])
        n_caseid = df[df_case_name].nunique()
        n_activity = df[df_activity_name].nunique()
        print("Number of CaseID", n_caseid, file=f)
        print("Number of Unique Activities", n_activity, file=f)
        print("Number of Activities", df[df_activity_name].count(), file=f)
        cont_trace = df[df_case_name].value_counts(dropna=False)
        max_trace = max(cont_trace)
        print("Max length trace", max_trace, file=f)
        print("Avg length trace", np.mean(cont_trace), file=f)
        print("Min length trace", min(cont_trace), file=f)

        durations = []
        grouped = df.groupby(df_case_name)
        curr_index = 0
        for case_id, group in grouped:

            duration = datetime.strptime(group['time'][curr_index + len(group['time']) - 1], "%d.%m.%Y-%H:%M:%S") - datetime.strptime(group['time'][curr_index + 0], "%d.%m.%Y-%H:%M:%S")
            durations.append(duration)
            curr_index += len(group['time'])

        durations = np.array(durations)
        mean_duration = np.mean(durations)
        max_duration = np.max(durations)
        print("Max duration", max_duration, file=f)
        print("Avg duration", mean_duration, file=f)

    f.close()
