import tweepy


api_key = "9fP6hbSt9x93DP7C36BYOPEu8"
api_key_secret = "rTevezd43Si0IFhWs3a3Usa6sMP57TDUtGalT3LQA8l0u1l1KA"
access_token = "1487433066838896640-KOvJomwDKznX9A3hjFyyNMrFubtGPz"
access_token_secret = "sapML6nXwOMpUsp8WGxSTSu1CQgr1ujezW5ZPP2JWVONH"

auth=tweepy.OAuthHandler(consumer_key=api_key, consumer_secret=api_key_secret)
auth.set_access_token(access_token,access_token_secret)

api=tweepy.API(auth)
print(api)

india_woeid=23424848

trend_result=api.get_place_trends(india_woeid)

try:
 
  for trend in trend_result[0]["trends"][:10]:  
     print(trend["name"])
     print(trend["tweet_volume"])   
     
except:
     print("wrong")