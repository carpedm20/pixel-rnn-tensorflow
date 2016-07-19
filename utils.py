import datetime
import dateutil.tz

def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y_%m_%d_%H_%M_%S')
