import pandas as pd

ws_val = 10
wd_val = 45
no2_val = 10
n = 1000

ws = [ws_val for _ in range(n)]
wd = [wd_val for _ in range(n)]
no2 =[no2_val for _ in range(n)]

df_test = pd.DataFrame({
                        'ws':ws,
                        'wd':wd,
                        'NO2':no2}
                    )