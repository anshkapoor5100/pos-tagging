from flask import Flask, request, render_template
import nltk
import numpy as np
from collections import defaultdict, Counter
from nltk.corpus import brown, stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

app = Flask(__name__)

class HMM_POS_Tagger:
    def __init__(self):
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.initial_probs = defaultdict(float)
        self.states = set()
        self.words = set()
        self.lemmatizer = WordNetLemmatizer()

    def train(self, tagged_sentences):
        transitions = defaultdict(Counter)
        emissions = defaultdict(Counter)
        initial_states = Counter()

        for sentence in tagged_sentences:
            prev_tag = None
            for i, (word, tag) in enumerate(sentence):
                word = word.lower()
                word = self.lemmatizer.lemmatize(word)  # Lemmatize for general cases

                self.states.add(tag)
                self.words.add(word)
                emissions[tag][word] += 1
                if i == 0:
                    initial_states[tag] += 1
                if prev_tag is not None:
                    transitions[prev_tag][tag] += 1
                prev_tag = tag

        # Transition and emission probability calculations
        self.transition_probs = {tag: {next_tag: (count + 1) / (sum(next_tags.values()) + len(self.states)) 
                                        for next_tag, count in next_tags.items()} 
                                    for tag, next_tags in transitions.items()}
        
        self.emission_probs = {tag: {word: (count + 1) / (sum(words.values()) + len(self.words)) 
                                      for word, count in words.items()} 
                                for tag, words in emissions.items()}

        # Add uniform probability for unseen words
        for tag in self.states:
            self.emission_probs[tag] = defaultdict(lambda: 1 / (sum(emissions[tag].values()) + len(self.words)),
                                                   self.emission_probs[tag])

        total_sentences = sum(initial_states.values())
        self.initial_probs = {tag: count / total_sentences for tag, count in initial_states.items()}

    def predict(self, sentence):
        viterbi = [{}]
        path = {}

        for state in self.states:
            viterbi[0][state] = self.initial_probs.get(state, 0) * self.emission_probs[state].get(sentence[0], 1/len(self.words))
            path[state] = [state]

        for t in range(1, len(sentence)):
            viterbi.append({})
            new_path = {}

            for state in self.states:
                (prob, prev_state) = max((viterbi[t-1][y0] * self.transition_probs[y0].get(state, 0) * 
                                          self.emission_probs[state].get(sentence[t], 1/len(self.words)), y0) 
                                         for y0 in self.states)
                viterbi[t][state] = prob
                new_path[state] = path[prev_state] + [state]

            path = new_path

        (prob, state) = max((viterbi[-1][y], y) for y in self.states)
        return path[state]

    def tag_sentence(self, sentence):
        try:
            words = sentence.split()
            # Keep verbs in their original form, lemmatize other words
            processed_words = [self.lemmatizer.lemmatize(word.lower()) if word not in stopwords.words('english') else word for word in words]
            tags = self.predict(processed_words)
            return list(zip(words, tags))  # Return original words with tags
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

# Load and prepare data
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('wordnet')
nltk.download('stopwords')
data = brown.tagged_sents(tagset='universal')

# Initialize the POS tagger
hmm_tagger = HMM_POS_Tagger()
hmm_tagger.train(data)

@app.route('/', methods=['GET', 'POST'])
def index():
    tagged_output = None
    if request.method == 'POST':
        sentence = request.form['sentence']
        tagged_output = hmm_tagger.tag_sentence(sentence)
        if tagged_output is not None:
            tagged_output = ', '.join([f"{word}: {tag}" for word, tag in tagged_output])
        else:
            tagged_output = "An error occurred while tagging the sentence."
    return render_template('index.html', tagged_output=tagged_output)

if __name__ == '__main__':
    app.run(debug=True)
