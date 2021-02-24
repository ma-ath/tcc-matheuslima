import requests
from logging import Handler, Formatter
import logging
import datetime
import json
import os

class RequestsHandler(Handler):
    def emit(self, record):
        log_entry = self.format(record)
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': log_entry,
            'parse_mode': 'HTML'
        }
        return requests.post("https://api.telegram.org/bot{token}/sendMessage".format(token=TELEGRAM_TOKEN),
                             data=payload).content

class LogstashFormatter(Formatter):
    def __init__(self):
        super(LogstashFormatter, self).__init__()

    def format(self, record):
        t = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        return "<i>{datetime}</i><pre>\n{message}</pre>".format(message=record.msg, datetime=t)

#I use a simple telegram bot logger to get notifications about the
# training process.
#

try:
    with open(os.path.join('include','telegram_credentials.json')) as f:
        my_credentials = json.load(f)

        TELEGRAM_TOKEN =  my_credentials['token']  #Telegram bot token
        TELEGRAM_CHAT_ID = my_credentials['myid']  #My telegram user id
        TELEGRAM_LOG_ACTIVATE = True               #A variable to turn on/off notifications
except:
    TELEGRAM_LOG_ACTIVATE = False

#Setup the telegram logger. This way, everytime we want to send something to the smartphone,
# we only need to call for logger.error("<message>")
logger = logging.getLogger('telegram-logger')
logger.setLevel(logging.WARNING)
handler = RequestsHandler()
formatter = LogstashFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)

def telegramSendMessage(message):
    try:
        """
            Envia uma mensagem de log para o telegram
        """
        if TELEGRAM_LOG_ACTIVATE:
            logger.error(message)
    except:
        print("[ERROR]: No internet connection")

if __name__ == '__main__':
    telegramSendMessage("TESTE")