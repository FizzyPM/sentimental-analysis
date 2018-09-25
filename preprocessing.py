import pandas as pd
import re
import emoji


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' :smiley: ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' :laugh: ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' :love: ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' :wink: ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' :sad: ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' :cry: ', tweet)
    # Handling other emojis
    tweet = emoji.demojize(tweet)
    return tweet


def preprocess_tweet(tweet):
    # Convert to lower case
    tweet = tweet.lower()
    # Removing non-ASCII characters
    # tweet = ''.join(i for i in tweet if ord(i) < 128)
    # Replace the URLs to empty strings
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)
    # Replace @handle with the empty string
    tweet = re.sub(r'@[\S]+', '', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove not alphanumeric symbols white spaces
    tweet = re.sub(r'[^\w]', ' ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet


df = pd.read_table('./datasets/test-A-input.txt', sep='\t', names=('A', 'B', 'C', 'D', 'E', 'F'))
# f = open("preprocessed.txt", "w")
# print(df.head())
# print(df.iloc[:, 5])
for row in (df.iloc[:, 5].head()):
    if (row != 'Not Available'):
        row = preprocess_tweet(row)
        print(row)
