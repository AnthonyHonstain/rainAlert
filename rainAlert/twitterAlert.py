import twitter
import time

class TwitterAlert(object):
    """Lets send some tweets"""
    def __init__(self):
          consumer_key = "ENTER_YOUR_OWN"
          consumer_secret = "ENTER_YOUR_OWN"
          access_key = "ENTER_YOUR_OWN"
          access_secret = "ENTER_YOUR_OWN"
          self.username = "target"
          self.api = twitter.Api(consumer_key=consumer_key, consumer_secret=consumer_secret,
                    access_token_key=access_key, access_token_secret=access_secret,
                    input_encoding=None)
          self.message = "run for cover looks like rain!"


    def sendAlert(self):
        timeStamp = time.mktime(time.gmtime())
        message = "@%s %s %s"%(self.username,self.message,timeStamp)
        self.api.PostUpdate(message)
        print "tweet sent"

