import datetime
from datetime import timedelta
import time

def Get_Date(dy):
    dy = str(dy)
    y = dy[0:4]
    m = dy[4:6]
    d = dy[6:8]
    return datetime.date(int(y), int(m), int(d))

def Dates_Del(d1, d2):
    d1 = Get_Date(d1)
    d2 = Get_Date(d2)
    delta = d1 - d2
    return abs(delta.days)

def Date_from_Epoch(epoch):
	return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))