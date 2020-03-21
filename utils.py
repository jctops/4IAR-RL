import time
import datetime

def printl(*args, **kwargs):
    time_str = f'[{datetime.datetime.today()}]'
    print(time_str, *args, **kwargs)