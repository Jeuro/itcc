import urllib2
import nltk
from nltk.tokenize import RegexpTokenizer


text_source = "http://www.gutenberg.org/cache/epub/158/pg158.txt"
data = urllib2.urlopen(text_source)
content_text = data.read()
tokenizer = RegexpTokenizer('\[[^\]]*\( |[+/\-@&*]|\w+|\$[\d\.]+|\S+')

text = tokenizer.tokenize(content_text)

