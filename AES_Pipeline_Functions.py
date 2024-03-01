import pandas as pd
import re
import spacy
import numpy as np
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from math import sqrt
from nltk.tokenize import sent_tokenize
import math
import textstat
from spellchecker import SpellChecker
import language_check
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from nltk.tag import pos_tag
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer
nltk.download('averaged_perceptron_tagger')
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error




def replace_nan_with_mean_or_median(group_data):
    "function to replace NaN values with calculated mean or median"
    for column in group_data.columns:
        if group_data[column].isnull().any():
            # Randomly choose whether to use mean or median
            replace_value = np.random.choice(['mean', 'median'])
            if replace_value == 'mean':
                value = group_data['domain1_score'].mean()
            else:
                value = group_data['domain1_score'].median()
            group_data[column].fillna(value, inplace=True)
    return group_data


## TAGS MANIPULATION
def extract_special_words_from_column(essay_column):

    """
    This function extracts special words prefixed with "@" from each essay (what we called tagged words).
    Input: essays.
    Output: A list of special words prefixed with "@" found in the essays.
    """

    special_words = []
    
    for essay in essay_column:
        words = essay.split()
        special_words.extend([word for word in words if word.startswith('@')])
    
    return special_words



def group_words_by_category(special_words):

    """"
    This function takes a list of special words prefixed with "@" as input.
    It extracts unique categories from the special words based on a predefined pattern (format: "@" followed by Capitalized (uppercase) category and an optional numeric part).
    The output is a list of unique categories found in the special words.
    """

    # Define the regular expression pattern
    pattern = re.compile(r'@([A-Z]+)([0-9]*)')

    # Initialize a dictionary to store words grouped by categories
    categories = {}

    # Loop through each special word and extract the categories using the pattern
    for word in special_words:
        match = pattern.match(word)
        if match:
            category = match.group(1)
            if category not in categories:
                categories[category] = []
            categories[category].append(word)
    
    return list(categories.keys())




def count_words_by_category(essay, special_words):
    
    """
    This function takes an essay (text) and a list of special words prefixed with "@" as input.
    It counts the occurrences of words for each category found in the essay based on the provided special words.
    The output is a dictionary containing the count of words for each category extracted from the essay.
    """

    # Define the regular expression pattern
    pattern = re.compile(r'@([A-Z]+)([0-9]*)')

    # Initialize a dictionary to store word counts for each category
    word_counts = {category: 0 for category in special_words}

    # Loop through each special word in the essay and update the counts
    for word in re.findall(pattern, essay):
        category = word[0]
        if category in word_counts:
            word_counts[category] += 1

    return word_counts

def process_essays(df, special_words):

    """
    This function processes essays stored in a DataFrame by extracting and categorizing special words.
    It adds a new column ('tag_counts') to the DataFrame, containing dictionaries of word counts for each category extracted from the essays.
    The output is the modified DataFrame with additional information about word counts for each category in the essays.
    """

    # Apply the function to each row in the dataframe
    df['tag_counts'] = df['essay'].apply(lambda x: count_words_by_category(x, special_words))

    # # Expand the dictionary into separate columns
    # df_categories = pd.json_normalize(df['category_word_counts']).add_prefix('@')

    # # Concatenate the original dataframe with the new columns
    # df = pd.concat([df, df_categories], axis=1)

    # # Drop the intermediate column
    # df = df.drop('category_word_counts', axis=1)

    return df



### ORGANIZTION AND STRUCTURES

def calculate_sentences_to_paragraphs_ratio(essay):
    
    """
    This function calculates the ratio of sentences to paragraphs in an essay.
    Input: essay (string): The text of the essay.
    Output: Ratio of sentences to paragraphs.
    Process: It splits the essay into paragraphs based on double line breaks ('\n\n'), then splits each paragraph into sentences based on periods ('.'). Finally, it calculates the ratio of the total number of sentences to the total number of paragraphs.
    """

    # Split the essay into paragraphs and sentences
    paragraphs = essay.split('\n')  # Assuming paragraphs are separated by double line breaks
    sentences = [sentence.strip() for paragraph in paragraphs for sentence in paragraph.split('.')]

    # Avoid division by zero
    if len(sentences) == 0:
        return 0

    # Calculate the ratio
    ratio = len(sentences) /len(paragraphs)

    return ratio

# Load the NLP spaCy English model
nlp = spacy.load("en_core_web_sm")
    

def calculate_nlp(essay):

    """
    This function processes an essay using the spaCy English model.
    Input: essay .
    Output: spaCy doc object representing the processed essay.
    Process: It loads the spaCy English model and applies it to the input essay, returning the resulting doc object.
    """

    doc = nlp(essay)
    return doc

def calculate_sentence_length_variation(essay):
    
    """
    This function calculates the variance of sentence lengths in an essay.
    Input: essay (spaCy doc object): The processed essay.
    Output: Variance of sentence lengths.
    Process: It extracts the lengths of individual sentences from the spaCy doc object, calculates the mean sentence length, and then computes the variance of sentence lengths based on the mean. If there's only one sentence, it sets the variance to 0.
    """

    # Process the essay with spaCy
    # doc = nlp(essay)
    doc = essay
    
    # Extract the lengths of individual sentences
    sentence_lengths = [len(sent) for sent in doc.sents]
    
    # Calculate the variance
    if len(sentence_lengths) > 1:
        mean_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((length - mean_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
    else:
        # If there's only one sentence, set the variance to 0
        variance = 0.0
    
    return variance

def count_transition_words(essay):

    """
    This function counts the occurrences of transition words in an essay.
    Input: The processed essay.
    Output: Count of transition words.
    Process: It defines a list of common transition words and then counts the occurrences of these words in the essay.
    """

    # Process the essay with spaCy
    # doc = nlp(essay)
    doc= essay
    
    # Define a list of common transition words
    transition_words = [
    'additionally', 'furthermore', 'moreover', 'in addition', 'also', 'and', 'as well as',
    'however', 'nevertheless', 'nonetheless', 'on the other hand', 'but', 'yet', 'still', 'conversely', 'in contrast',
    'similarly', 'likewise', 'in the same way', 'compared to', 'just as', 'akin to', 'similarly to',
    'therefore', 'thus', 'consequently', 'as a result', 'because', 'since', 'due to', 'owing to', 'hence',
    'meanwhile', 'afterward', 'subsequently', 'next', 'then', 'finally', 'in the meantime', 'during', 'before', 'after', 'now',
    'in conclusion', 'to sum up', 'overall', 'in summary', 'to conclude', 'consequently',
    'for example', 'for instance', 'to illustrate', 'specifically', 'such as',
    'indeed', 'certainly', 'in fact', 'of course', 'undoubtedly', 'unquestionably',
    'in addition to', 'moreover', 'furthermore', 'on the other hand', 'as a result', 'in conclusion', 'to sum up', 'for instance', 'in fact', 'for example',
    'first', 'second', 'third', 'finally', 'next', 'then', 'before', 'after', 'meanwhile'
        ]   

    # Count the occurrences of transition words
    count = sum(1 for token in doc if token.text.lower() in transition_words)
    
    return count


""""
The NLTK (Natural Language Toolkit) punkt and stopwords datasets provide resources 
for tokenization and stopword removal, respectively, in natural language processing tasks. 
The punkt dataset contains tokenizers for splitting text into sentences and words, while 
the stopwords dataset contains a list of common words that are typically filtered out during
 text preprocessing as they do not carry significant meaning. By downloading these datasets,
you ensure that you have access to the necessary tools for text processing with NLTK.
"""
nltk.download('punkt')
nltk.download('stopwords')

def calculate_cttr(essay):
    # Tokenize the text and remove stopwords
    tokens = [word.lower() for word in word_tokenize(essay) if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Calculate total number of words and unique words
    total_words = len(tokens)
    unique_words = len(set(tokens))

    # Calculate CTTR
    cttr = unique_words / sqrt(2 * (total_words / unique_words))

    return cttr



def count_sentences(essay):

    """
    Counts the number of sentences in the provided essay.
    Input: Essay 
    Output: Total number of sentences
    """

    # Define endpoint punctuation marks
    endpoint_punctuation = ['.', '!', '?']

    # Initialize count
    num_sentences = 1

    # Iterate through each character in the essay
    for i in range(len(essay)):
        # If the character is an endpoint punctuation mark
        if essay[i] in endpoint_punctuation:
            # Check if there's a character immediately after it
            if i + 1 < len(essay) and (essay[i + 1].isalpha() or essay[i + 1] == ' '):
                # Increment the sentence count
                num_sentences += 1

    # Return the total number of sentences
    return num_sentences



def average_sentence_length(essay):
    
    """
    Calculates the average length of sentences in the provided essay.
    Input: Essay 
    Output: Average sentence length 
    """

    # Process the essay using spaCy
    # doc = nlp(essay)
    doc = essay
    
    # Calculate the length of each sentence
    sentence_lengths = [len(sent) for sent in doc.sents]
    
    # Calculate the average sentence length
    if sentence_lengths:
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        return math.ceil(avg_length)
    else:
        return 0  # Handle the case of an empty essay or no sentences


"""
The following function calculates the frequency of words with selected parts of speech (POS) in the provided essay.
In linguistics, POS refers to the grammatical category of words based on their syntactic functions within a sentence. 
Input: Essay , number (number of selected words to take into consideration)
Output: List containing the most common words and their frequencies, based on selected POS
"""
# Define the POS tags to exclude (e.g., prepositions, conjunctions)
exclude_pos = ['ADP', 'CCONJ', 'SCONJ', 'PUNCT', 'DET', 'PRON', 'SYM', 'NUM', 'X']

def words_frequency_selected_pos(essay, number):
    # Tokenize the essay and convert to lowercase
    words = [word.lower() for word in word_tokenize(essay) if word.isalpha()]

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Get POS tags using spaCy
    pos_tags = [token.pos_ for token in nlp(' '.join(words))]

    # Filter out unwanted POS
    selected_words = [word for word, pos in zip(words, pos_tags) if pos not in exclude_pos]

    # Calculate word frequency for the selected POS
    freq_dist = FreqDist(selected_words)
    return freq_dist.most_common(number)


spell_checker = SpellChecker()

def check_spelling_pyspellchecker(text):

    """
    The check_spelling_pyspellchecker function takes a piece of text as input, splits it into words, 
    and then checks each word for spelling errors the spell checker object. It returns a list of words 
    that are identified as misspelled.
    """
    
    words = text.split()
    misspelled_words = spell_checker.unknown(words)
    return misspelled_words



def count_pos(text):

    """
    This function analyzes the text to determine the percentages of words classified as adjectives, adverbs, nouns, and verbs.
    Input: Essay
    Output: Percentages of adjectives, adverbs, nouns, and verbs in the text.
    """

    # Tokenize the text
    sentences = nltk.sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    
    # Tag the words with their parts of speech
    tagged_words = [nltk.pos_tag(sentence) for sentence in words]
    
    # Initialize lists to store words tagged as adjectives, adverbs, nouns, and verbs
    adjectives = []
    adverbs = []
    nouns = []
    verbs = []
    
    # Iterate over tagged words and categorize them
    for sentence in tagged_words:
        for word, tag in sentence:
            if len(word) >= 2:  # Check if the word has at least two characters
                if tag.startswith('JJ'):  # Adjectives
                    adjectives.append(word)
                elif tag.startswith('RB'):  # Adverbs
                    adverbs.append(word)
                elif tag.startswith('NN'):  # Nouns
                    nouns.append(word)
                elif tag.startswith('VB'):  # Verbs
                    verbs.append(word)
    
    # Calculate the percentages
    total_words = sum(len(sentence) for sentence in words)
    percent_adjectives = (len(adjectives) / total_words) * 100
    percent_adverbs = (len(adverbs) / total_words) * 100
    percent_nouns = (len(nouns) / total_words) * 100
    percent_verbs = (len(verbs) / total_words) * 100
    
    # return percent_adjectives, adjectives, percent_adverbs, adverbs, percent_nouns, nouns, percent_verbs, verbs
    return percent_adjectives, percent_adverbs, percent_nouns, percent_verbs


def long_word_token_ratio(nlp_spacy):
    """
    This function calculates the ratio of word tokens with more than 6 characters to total word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of long word tokens to total word tokens
    """
    long_word_tokens = sum(1 for token in nlp_spacy if len(token.text) > 6 and token.is_alpha)
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return long_word_tokens / total_word_tokens if total_word_tokens != 0 else 0


def short_word_token_ratio(nlp_spacy):
    """
    This function calculates the ratio of word tokens with less than 4 characters to total word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of short word tokens to total word tokens
    """
    short_word_tokens = sum(1 for token in nlp_spacy if len(token.text) < 4 and token.is_alpha)
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return short_word_tokens / total_word_tokens if total_word_tokens != 0 else 0


def lemma_token_ratio(nlp_spacy):
    """
    This function calculates the ratio of lemmas to total word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of lemmas to total word tokens
    """
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    total_lemmas = len(set(token.lemma_ for token in nlp_spacy if token.is_alpha))
    return total_lemmas / total_word_tokens if total_word_tokens != 0 else 0


def token_sentence_ratio(nlp_spacy, num_sentences):
    """
    This function calculates the ratio of word tokens to number of sentences.
    Input:
    - nlp_spacy: Spacy NLP processed document
    - num_sentences: Number of sentences in the document
    Output:
    - Ratio of word tokens to number of sentences
    """
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return total_word_tokens / num_sentences if num_sentences != 0 else 0


def non_initial_caps_word_ratio(nlp_spacy, num_sentences):
    """
    This function calculates the ratio of non-initial capital words to number of sentences.
    Input:
    - nlp_spacy: Spacy NLP processed document
    - num_sentences: Number of sentences in the document
    Output:
    - Ratio of non-initial capital words to number of sentences
    """
    non_initial_caps_words = sum(1 for token in nlp_spacy if token.is_alpha and not token.is_title)
    return non_initial_caps_words / num_sentences if num_sentences != 0 else 0


def char_sentence_ratio(nlp_spacy, num_sentences):
    """
    This function calculates the ratio of characters to number of sentences.
    Input:
    - nlp_spacy: Spacy NLP processed document
    - num_sentences: Number of sentences in the document
    Output:
    - Ratio of characters to number of sentences
    """
    total_chars = sum(len(token.text) for token in nlp_spacy if token.is_alpha)
    return total_chars / num_sentences if num_sentences != 0 else 0

import math

def fourth_root_word_tokens(nlp_spacy):
    """
    This function calculates the fourth root of the number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Fourth root of the number of word tokens
    """
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return math.pow(total_word_tokens, 1/4)


def word_variation_index(nlp_spacy, nouns_percentage):
    """
    This function calculates the Word Variation Index (OVIX).
    Input:
    - nlp_spacy: Spacy NLP processed document
    - nouns_percentage: Percentage of nouns in the document
    Output:
    - Word Variation Index (OVIX)
    """
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return total_word_tokens / (1 - nouns_percentage / 100)


def nominal_ratio(nlp_spacy, nouns_percentage):
    """
    This function calculates the Nominal Ratio (NR).
    Input:
    - nlp_spacy: Spacy NLP processed document
    - nouns_percentage: Percentage of nouns in the document
    Output:
    - Nominal Ratio (NR)
    """
    total_nouns = sum(1 for token in nlp_spacy if token.pos_ == 'NOUN')
    return total_nouns / (1 - nouns_percentage / 100)

def conjunction_ratio(nlp_spacy):
    """
    This function calculates the ratio of conjunctions divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of conjunctions to total word tokens
    """
    conjunction_count = sum(1 for token in nlp_spacy if token.pos_ == 'CONJ')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return conjunction_count / total_word_tokens if total_word_tokens != 0 else 0


def subjunction_ratio(nlp_spacy):
    """
    This function calculates the ratio of sub-junctions divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of sub-junctions to total word tokens
    """
    subjunction_count = sum(1 for token in nlp_spacy if token.pos_ == 'SCONJ')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return subjunction_count / total_word_tokens if total_word_tokens != 0 else 0


def genitive_form_ratio(nlp_spacy):
    """
    This function calculates the ratio of genitive forms divided by the total number of nouns.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of genitive forms to total nouns
    """
    genitive_forms = {'poss', 'gen'}  # Example: consider 'poss' and 'gen' tags as genitive forms
    noun_count = sum(1 for token in nlp_spacy if token.pos_ == 'NOUN')
    genitive_form_count = sum(1 for token in nlp_spacy if token.tag_ in genitive_forms)
    return genitive_form_count / noun_count if noun_count != 0 else 0


# You can implement similar functions for other features mentioned.


def minor_delimiter_ratio(nlp_spacy):
    """
    Calculate the ratio of minor delimiters (MID) divided by the total number of word tokens.

    Input:
    - nlp_spacy: Spacy NLP processed document

    Output:
    - Ratio of minor delimiters to total word tokens
    """
    minor_delimiter_count = sum(1 for token in nlp_spacy if token.pos_ == 'PUNCT' and token.dep_ != 'punct')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return minor_delimiter_count / total_word_tokens if total_word_tokens != 0 else 0

def major_delimiter_ratio(nlp_spacy):
    """
    The function calculates the ratio of major delimiters (MAD) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of major delimiters to total word tokens
    """
    major_delimiter_count = sum(1 for token in nlp_spacy if token.pos_ == 'PUNCT' and token.dep_ == 'punct')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return major_delimiter_count / total_word_tokens if total_word_tokens != 0 else 0

def particle_ratio(nlp_spacy):
    """
    The function calculates the ratio of particles (PL) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of particles to total word tokens
    """
    particle_count = sum(1 for token in nlp_spacy if token.pos_ == 'PART')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return particle_count / total_word_tokens if total_word_tokens != 0 else 0

def relative_adverb_ratio(nlp_spacy):
    """
    The function calculates the ratio of relative adverbs (HA) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of relative adverbs to total word tokens
    """
    relative_adverb_count = sum(1 for token in nlp_spacy if token.pos_ == 'ADV' and token.dep_ == 'advmod')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return relative_adverb_count / total_word_tokens if total_word_tokens != 0 else 0

def determiner_ratio(nlp_spacy):
    """
    This function calculates the ratio of determiners (DT) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of determiners to total word tokens
    """
    determiner_count = sum(1 for token in nlp_spacy if token.pos_ == 'DET')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return determiner_count / total_word_tokens if total_word_tokens != 0 else 0

def interrogative_relative_determiner_ratio(nlp_spacy):
    """
    This function calculates the ratio of interrogative relative determiners (HD) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of interrogative relative determiners to total word tokens
    """
    interrogative_relative_determiner_count = sum(1 for token in nlp_spacy if token.tag_ == 'WDT')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return interrogative_relative_determiner_count / total_word_tokens if total_word_tokens != 0 else 0

def participle_ratio(nlp_spacy):
    """
    This function calculates the ratio of participles (PC) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of participles to total word tokens
    """
    participle_count = sum(1 for token in nlp_spacy if token.pos_ == 'VERB' and token.tag_ == 'VBG')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return participle_count / total_word_tokens if total_word_tokens != 0 else 0


def paired_delimiter_ratio(nlp_spacy):
    """
    It calculates the ratio of paired delimiters (PAD) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of paired delimiters to total word tokens
    """
    paired_delimiter_count = sum(1 for token in nlp_spacy if token.pos_ == 'PUNCT' and token.dep_ == 'punct')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return paired_delimiter_count / total_word_tokens if total_word_tokens != 0 else 0

def passive_voice_ratio(nlp_spacy):
    """
    It calculates the ratio of passive voice sentences divided by the total number of sentences.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of passive voice sentences to total sentences
    """
    total_sentences = len(list(nlp_spacy.sents))
    passive_voice_count = sum(1 for token in nlp_spacy if token.dep_ == 'auxpass')
    return passive_voice_count / total_sentences if total_sentences != 0 else 0

def active_voice_ratio(nlp_spacy):
    """
    It calculates the ratio of active voice sentences divided by the total number of sentences.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of active voice sentences to total sentences
    """
    total_sentences = len(list(nlp_spacy.sents))
    active_voice_count = sum(1 for token in nlp_spacy if token.dep_ == 'nsubj' or token.dep_ == 'nsubjpass')
    return active_voice_count / total_sentences if total_sentences != 0 else 0

def possessive_form_ratio(nlp_spacy):
    """
    It calculates the ratio of possessive forms (PS) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of possessive forms to total word tokens
    """
    possessive_form_count = sum(1 for token in nlp_spacy if token.tag_ == 'POS')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return possessive_form_count / total_word_tokens if total_word_tokens != 0 else 0

def proposition_ratio(nlp_spacy):
    """
    It calculates the ratio of propositions (PP) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of propositions to total word tokens
    """
    proposition_count = sum(1 for token in nlp_spacy if token.pos_ == 'ADP')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return proposition_count / total_word_tokens if total_word_tokens != 0 else 0

def adjective_ratio(nlp_spacy):
    """
    It calculates the ratio of adjectives (JJ) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of adjectives to total word tokens
    """
    adjective_count = sum(1 for token in nlp_spacy if token.pos_ == 'ADJ')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return adjective_count / total_word_tokens if total_word_tokens != 0 else 0

def preposition_ratio(nlp_spacy):
    """
    It calculates the ratio of prepositions (PP) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of prepositions to total word tokens
    """
    preposition_count = sum(1 for token in nlp_spacy if token.pos_ == 'ADP')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return preposition_count / total_word_tokens if total_word_tokens != 0 else 0

def interrogative_relative_pronoun_ratio(nlp_spacy):
    """
    It calculates the ratio of interrogative relative pronouns (HP) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of interrogative relative pronouns to total word tokens
    """
    interrogative_relative_pronoun_count = sum(1 for token in nlp_spacy if token.tag_ == 'WP')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return interrogative_relative_pronoun_count / total_word_tokens if total_word_tokens != 0 else 0

def foreign_word_ratio(nlp_spacy):
    """
    It calculates the ratio of foreign words (UO) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of foreign words to total word tokens
    """
    foreign_word_count = sum(1 for token in nlp_spacy if token.lang_ != 'en')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return foreign_word_count / total_word_tokens if total_word_tokens != 0 else 0

def counting_word_ratio(nlp_spacy):
    """
    It calculates the ratio of counting words (RG) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of counting words to total word tokens
    """
    counting_word_count = sum(1 for token in nlp_spacy if token.tag_ == 'CD')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return counting_word_count / total_word_tokens if total_word_tokens != 0 else 0

def ordinal_counting_word_ratio(nlp_spacy):
    """
    It calculates the ratio of ordinal counting words (RO) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of ordinal counting words to total word tokens
    """
    ordinal_counting_word_count = sum(1 for token in nlp_spacy if token.tag_ == 'OD')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return ordinal_counting_word_count / total_word_tokens if total_word_tokens != 0 else 0

def pronoun_ratio(nlp_spacy):
    """
    It calculates the ratio of pronouns (PN) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of pronouns to total word tokens
    """
    pronoun_count = sum(1 for token in nlp_spacy if token.pos_ == 'PRON')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return pronoun_count / total_word_tokens if total_word_tokens != 0 else 0

def pronoun_object_form_ratio(nlp_spacy):
    """
    It calculates the ratio of pronouns in object form (OBJ) divided by the total number of pronouns.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of pronouns in object form to total pronouns
    """
    pronoun_count = sum(1 for token in nlp_spacy if token.pos_ == 'PRON')
    pronoun_object_form_count = sum(1 for token in nlp_spacy if token.pos_ == 'PRON' and token.dep_ == 'dobj')
    return pronoun_object_form_count / pronoun_count if pronoun_count != 0 else 0

def verb_ratio(nlp_spacy):
    """
    It calculates the ratio of verbs (VB) divided by the total number of word tokens.
    Input:
    - nlp_spacy: Spacy NLP processed document
    Output:
    - Ratio of verbs to total word tokens
    """
    verb_count = sum(1 for token in nlp_spacy if token.pos_ == 'VERB')
    total_word_tokens = sum(1 for token in nlp_spacy if token.is_alpha)
    return verb_count / total_word_tokens if total_word_tokens != 0 else 0


def train_predict_models(x_train, y_train, x_test):
    """
    Input:
    x_train: Features used for training the regression models.
    y_train: Target variable used for training the regression models.
    x_test: Features used for predicting the target variable.
    output:
    predictions made by the Random Forest and XGBoost models
    This function trains two regression models: Random Forest Regressor and XGBoost Regressor, using the training data (x_train, y_train).
    It then predicts the target variable for the test data (x_test) using both trained models.
    """
    # Initialize Random Forest Regressor
    rf_regressor = RandomForestRegressor(random_state=42)
    
    # Train Random Forest Regressor
    rf_regressor.fit(x_train, y_train)
    
    # Predict on test set using Random Forest Regressor
    rf_predictions = rf_regressor.predict(x_test)
    
    # Initialize XGBoost Regressor
    xgb_regressor = XGBRegressor(random_state=42)
    
    # Train XGBoost Regressor
    xgb_regressor.fit(x_train, y_train)
    
    # Predict on test set using XGBoost Regressor
    xgb_predictions = xgb_regressor.predict(x_test)
    
    return rf_predictions, xgb_predictions

def evaluate_regression_model(y_true, y_pred, column_names=None):
    """
    Inputs:
    y_true, the true target values, and y_pred, the predicted target values
    This function evaluates the performance of regression models by calculating various metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) score.
    Additionally, it plots a bar chart showing the values of the evaluation metrics for visual comparison.
    """
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Check if y_true and y_pred have the same shape
    if y_true.shape != y_pred.shape:
        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)
        # print("Error: y_true and y_pred must have the same shape.")
        # return
    
    # Check if y_true and y_pred are 1D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate evaluation metrics for each column
    for i in range(y_true.shape[1]):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        
        # Print evaluation metrics
        if column_names and len(column_names) > i:
            print(f"Metrics for {column_names[i]}:")
        else:
            print(f"Metrics for Column {i + 1}:")
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("Mean Absolute Error (MAE):", mae)
        print("R-squared (R2):", r2)
        print()
         # Plot the results
        plt.figure(figsize=(10, 4))
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        values = [mse, rmse, mae, r2]
        plt.bar(metrics, values, color=['skyblue', 'salmon', 'lightgreen', 'gold'])
        plt.title(f"Metrics for {column_names[i]}" if column_names and len(column_names) > i else f"Metrics for Column {i + 1}")
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.show()
        

    
def round_to_nearest_integer(predicted_values):
    """
    Inputs:
    - Predicted values
    Output:

    round_to_nearest_integer:
    rounded values as integers
    This function rounds the predicted values to the nearest integer.
    """
    rounded_values = np.round(predicted_values)
    return rounded_values.astype(int)