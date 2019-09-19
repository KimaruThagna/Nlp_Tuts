from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence) # obtain polarity index of given sentence.
    # produces the index of the -ve, +ve, neutral and compound sentiment
    # the compound score is a sum of all lexicon ratings normalized between -1(very negative) to 1 (very positive)
    print(f'{sentence} polarity scores: {score}')

sentiment_analyzer_scores("This phone is super cool")

# punctuation. ! shows a degree of intensity. This is good !!!! is more intense compared to this is good
print(">>>>>>>>>>>>>>>>>Punctuation>>>>>>>>>>>>>>>>>")
sentiment_analyzer_scores("This food is good")
sentiment_analyzer_scores("This food is good!!")
sentiment_analyzer_scores("This food is good!!!")

# capitalization. Denotes emphasis
print(">>>>>>>>>>>>>>>>>Capitalization>>>>>>>>>>>>>>>>>")
sentiment_analyzer_scores("This food is good!")
sentiment_analyzer_scores("This food is GREAT")
sentiment_analyzer_scores("This food is great")


# Degree  modifiers/intensifier. Impact intensity positively or negatively
print(">>>>>>>>>>>>>>>>>Intensifiers>>>>>>>>>>>>>>>>>")
sentiment_analyzer_scores("The lunch was extremely good")
sentiment_analyzer_scores("The lunch was EXTREMELY bad")
sentiment_analyzer_scores("The lunch was marginally good")


#Conjunctions. They denote a shift in sentiment. eg but
print(">>>>>>>>>>>>>>>>>Intensifiers>>>>>>>>>>>>>>>>>")
sentiment_analyzer_scores("The lunch was extremely good")
sentiment_analyzer_scores("The lunch was EXTREMELY bad")
sentiment_analyzer_scores("The lunch was marginally good")


# preceding Tri-gram
print(">>>>>>>>>>>>>>>>>Intensifiers>>>>>>>>>>>>>>>>>")
sentiment_analyzer_scores("The lunch was extremely good")
sentiment_analyzer_scores("The lunch was EXTREMELY bad")
sentiment_analyzer_scores("The lunch was marginally good")


# Emojis
print(">>>>>>>>>>>>>>>>>Emoji>>>>>>>>>>>>>>>>>")
sentiment_analyzer_scores("I am 😄 today")
sentiment_analyzer_scores("😊")
sentiment_analyzer_scores("😥")
print(sentiment_analyzer_scores('☹️'))


# Common slang
print(">>>>>>>>>>>>>>>>>Slang>>>>>>>>>>>>>>>>>")
print(sentiment_analyzer_scores("Today SUX!"))
print(sentiment_analyzer_scores("Today only kinda sux! But I'll get by, lol"))


# Emoticon
print(">>>>>>>>>>>>>>>>>Emoticons>>>>>>>>>>>>>>>>>")
sentiment_analyzer_scores("You either feel :) or :D")
sentiment_analyzer_scores("wow. That was :(")
