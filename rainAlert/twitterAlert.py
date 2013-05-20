import twitter
import time

class TwitterAlert(object):
    """Lets send some tweets"""
    def __init__(self):
          consumer_key = "aHfvtF0vNXpJHUT7412Cw" 
          consumer_secret = "9FIdL0bnrUFAgWj5nk1pwZAAp7aK3eJQLFGXs9CiUw"
          access_key = "1427879054-JBAi3D2j4QZbOpHxC10BbROxjf856B2ZB49191Q"
          access_secret = "OSQw1Ydmbm45EtEDrHsqXGgqk7RK3vaSNn0xvCqjw0"
          self.username = "atilev"
          self.api = twitter.Api(consumer_key=consumer_key, consumer_secret=consumer_secret,
                    access_token_key=access_key, access_token_secret=access_secret,
                    input_encoding=None)
          self.message = "run for cover looks like rain!"


    def sendAlert(self):
        timeStamp = time.mktime(time.gmtime())
        message = "@%s %s %s"%(self.username,self.message,timeStamp)
        self.api.PostUpdate(message)
        print "tweet sent"
