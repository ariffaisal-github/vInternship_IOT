import numpy as np
import nltk
from nltk.corpus import stopwords
import spacy

tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords = stopwords.words('english')
stopwords.append('mm')
stopwords.append('c')
stopwords.append('md')
nlp = spacy.load('en_core_web_sm')
def clean_naration(text):
    parser = nlp(text)
    cleaned = []
    remove_types = ['PERSON', 'GPE', 'LOC']
    for item in parser:
        if item.ent_type in remove_types:
            pass
        else:
            cleaned.append(item.text)
    return (" ".join(cleaned))

def remove_special_chars(sentence):
    result = sentence.replace('[^A-Za-z]', ' ', regex = True)
    result = result.replace('withdrawal', 'withdraw', regex = True)
    return result

def remove_stop_words(sentence):
    return ' '.join(word.lower() for word in str(sentence).split() if word not in stopwords)

def lemmatize(text):
    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)]

def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature, score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def get_keywords(vectorizer, feature_names, doc, k=10):
    """Return top k keywords from a doc using TF-IDF method"""

    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])
    
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only TOP_K_KEYWORDS
    keywords=extract_topn_from_vector(feature_names,sorted_items,k)
    
    return list(keywords.keys())

def mean_word2vec(text, model, vector_size=150):
    sum = np.zeros(vector_size)
    cnt = 1
    for word in text:
        sum+=model.wv[word]
        cnt+=1
    return sum/cnt
