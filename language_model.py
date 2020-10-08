import sys
from math import log10
import numpy as np
import matplotlib.pyplot as plt
from random import choice as rand_choice

"""
Yuval Timen
N-Gram statistical language model generalized for all n-gram orders >= 1
Main method will train a unigram and bigram model, will generate 50 sentences
from each of these models, will create histograms from the probabilities of
each test set, and will calculate the perplexity of the first 10 sentences of
the test set.
"""


class LanguageModel:
    
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"
    MIN_FREQUENCY = 2  # Threshold for <UNK>
    
    
    def __init__(self, n_gram, is_laplace_smoothing, verbose=False):
        """Initializes an untrained LanguageModel
        Parameters:
          n_gram (int): the n-gram order of the language model to create
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
          backoff (str): optional argument with default value None of whether or not to use a backoff stragety. Options: ["katz"]
        """
        # The order of the n-grams used for this model
        if (n_gram < 1) or (type(n_gram) != int):
            raise Exception("N-gram order must be an integer >= 1")
        self.ngram_order = n_gram
        # The set of words encountered by the model
        self.vocabulary = set()
        # The size of the vocabulary - used in some calculations
        self.vocab_size = 0
        # Counts for the n-grams
        self.n_gram_counts = dict()
        # List of n-grams for sentence generation
        self.n_gram_list = list()
        # Whether to use laplace smoothing
        self.use_laplace_smoothing = is_laplace_smoothing
        # Whether to print verbose output
        self.verbose = verbose
        # The log to print out at the end
        self.log = ""


    def load_tokens(self, training_file_path):
        """Reads the training data from the given path and produces a list of tokens.
        Parameters:
          training_file_path (str) the location of the training data to read
        Returns:
          list: the list of tokenized sentences
        """
        output_list = []
    
        with open(training_file_path) as f:
            for line in f.readlines():
                output_list.extend(line.split())

        if self.verbose:
            self.log += f"Loaded in {training_file_path}\n Found {len(output_list)} tokens\n"
    
        return output_list
        

    def create_ngrams(self, tokens, n):
        """Creates a list of n-grams from the provided list of tokens
        Parameters:
          tokens (list) the list of tokenized words
          n (int) the order of the n-grams to produce
        Returns:
          list: the list of tuples representing each n-gram
        """
        
        output_list = []
        
        for idx in range(len(tokens) - n + 1):
            output_list.append(tuple(tokens[idx:idx + n]))
        
        if self.verbose:
            self.log += f"Splicing: {tokens[:4]} \nFound {len(output_list)} n-grams\n"
        
        return output_list
    
    
    def replace_by_unk(self, tokens):
        """Replaces any token not encountered by the model with <UNK>
        Parameters:
            tokens (list) the list of tokenized words
        Returns:
            list: the list of tokens, with any unknown words replaced with <UNK>
        """
        
        for idx in range(len(tokens)):
            if tokens[idx] not in self.vocabulary:
                tokens[idx] = self.UNK
        
        return tokens
    
    
    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
          has tokens that are white-space separated, has one sentence per line, and
          that the sentences begin with <s> and end with </s>
        Parameters:
          training_file_path (str): the location of the training data to read
        Returns:
          None
        """
        
        # Refresh the counting dict and n-gram list in case we re-train an existing model
        self.n_gram_counts = dict()
        self.n_gram_list = list()
        
        tokens = self.load_tokens(training_file_path)
        
        # First pass - replace any word below the MIN_FREQUENCY with <UNK>
        initial_counts = dict()
        for token in tokens:
            if token in initial_counts.keys():
                initial_counts[token] += 1
            else:
                initial_counts[token] = 1
        
        for idx in range(len(tokens)):
            if initial_counts[tokens[idx]] < self.MIN_FREQUENCY:
                tokens[idx] = self.UNK
            
        # Save our vocab size for future computations
        self.vocabulary = set(tokens)
        self.vocab_size = len(self.vocabulary)
        
        
        # Second pass - create n-grams and update counts in stored dict
        n_grams = self.create_ngrams(tokens, self.ngram_order)
        for ngram in n_grams:
            # Put this n-gram in our list - we will sample from it later during sentence generation
            self.n_gram_list.append(ngram)
            # Update the count associated with each n-gram in our dictionary
            if ngram in self.n_gram_counts.keys():
                self.n_gram_counts[ngram] += 1
            else:
                self.n_gram_counts[ngram] = 1

        if self.verbose:
            self.log += f"Training {self.ngram_order}-gram model...\n"
            self.log += f"|V| = {self.vocab_size}\n"
            self.log += f"Range: [{min(self.n_gram_counts.values())},{max(self.n_gram_counts.values())}]\n"

        
    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of
        Returns:
          float: the probability value of the given string for this model
        """
        
        if self.verbose:
            self.log += "Scoring...\n"
            
        # If the string is empty then we have a problem
        if not sentence:
            raise Exception("Cannot score an empty string")
        
        # Given an example sentence of '<s> w1 w2 w3 w4 w5 </s>'
        # We want to compute bigram probability of this sentence
        # So for bigrams, we calculate:
        # MLE = P(w1 | <s>) * P(w2 | w1) * P(w3 | w2) * P(w4 | w3) * P(w5 | w4) * P(</s> | w5)
        sentence_tokens = sentence.split()
        sentence_tokens = self.replace_by_unk(sentence_tokens)
        sentence_ngrams = self.create_ngrams(sentence_tokens, self.ngram_order)
        
        # Now sentence_ngrams = [(<s>, w1), (w1, w2), (w2, w3), (w3, w4), (w4, w5), (w5, </s>)]
        # For a unigram model, P(w) = C(w) / sum_w(C(w))
        # For a bigram model, P(w2 | w1) = C((w1, w2)) / C(w1)
        # For an n-gram model, P(wN | w1, w2, ... w_N-1) = C(w1...wN) / C(w1 ... w_N-1)
        log_probability = 0
        for sentence_ngram in sentence_ngrams:
            # How many times does this n-gram occur?
            if sentence_ngram in self.n_gram_counts.keys():
                numerator = self.n_gram_counts[sentence_ngram]
            elif self.use_laplace_smoothing:
                numerator = 0
            else:
                if self.verbose:
                    self.log += f"P(\"{sentence}\") = 0\n"
                return 0
            # How many times do all n-grams with words w1...w_N-1 occur?
            # Sum up the counts for all keys that share the first N-1 tokens of our n-gram
            if self.ngram_order == 1:
                # Save on computation time - if we're using a unigram model, then
                # the denominator is just the total count of unigrams
                denominator = sum(self.n_gram_counts.values())
            else:
                # Find all matching tuples up to N-1
                denominator = 0
                for key in self.n_gram_counts.keys():
                    if key[:-1] == sentence_ngram[:-1]:
                        denominator += self.n_gram_counts[key]
            
            if self.verbose:
                if self.use_laplace_smoothing:
                    self.log += f"(({numerator} + 1)/({denominator} + {self.vocab_size})) * \n"
                else:
                    self.log += f"(({numerator})/({denominator})) * \n"
    
            # Add to the running log-probability total
            if self.use_laplace_smoothing:
                log_probability += log10((numerator + 1) / (denominator + self.vocab_size))
            else:
                log_probability += log10(numerator / denominator)
            
        # Return the result as a probability -> 2^log_prob
        ans = pow(10, log_probability)
        
        if self.verbose:
            self.log += f"P(\"{sentence}\") = {ans}\n"
        
        return ans


    def perplexity(self, test_sequence):
        """Measures the perplexity for the given test sequence with this trained model.
          As described in the text, you may assume that this sequence may consist of many
          sentences "glued together".
        Parameters:
          test_sequence (string): a sequence of space-separated tokens to measure the perplexity of
        Returns:
          float: the perplexity of the given sequence
        """
        # Measure the probability of the sequence using our language model
        sequence_probability = self.score(test_sequence)
        
        # PP(W) = [P(w1 w2.... wN)]^(-1/N)
        perplexity = pow(sequence_probability, -1/self.vocab_size)
        if self.verbose:
            self.log += f"PP(\"{test_sequence}\") = {perplexity}\n"
        
        return perplexity
    
    
    def print_log(self):
        """Prints out the log to the console
        Parameters:
            None
        Returns:
            None
        """
        print(self.log)
        
    
    def sew_together(self, tupls):
        """Takes in a list of tuples of order n and 'sews' them together
        to form a string.
        Parameters:
            tupls (list) The list of tuples to be combined
        Returns:
            str: The sewn together n-grams
        """
        output = []
        # Iterate through the list
        for idx in range(len(tupls)):
            # If it's the last tuple, take all the entries
            if idx == len(tupls) - 1:
                output.extend(tupls[idx])
            # Otherwise, take just the first entry
            else:
                output.append(tupls[idx][0])
        
        return " ".join(output)
    
    
    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.
        Returns:
          str: the generated sentence
        """
        output = []
        
        # Define the initial filter function - the first N-1 terms of the tuple must be <s>
        init_filt = lambda tup: tup[:-1] == tuple(self.SENT_BEGIN for _ in range(self.ngram_order - 1))
        # Filter out all n-grams that don't satisfy this
        init_candidates = list(filter(init_filt, self.n_gram_list))
        # Choose randomly from list - each n-gram is duplicated according to its frequency already
        output.append(rand_choice(init_candidates))
        
        # While we have not yet found the end token
        while output[-1][-1] != self.SENT_END:
            # Filter out all n-grams that don't succeed the last one
            current_candidates = list(filter(lambda tup: tup[:-1] == output[-1][1:], self.n_gram_list))
            current_candidates = list(filter(lambda tup: tup != (self.SENT_BEGIN,), current_candidates))
            output.append(rand_choice(current_candidates))
        
        # Tack on any remaining </s> tokens
        add_to_end = [self.SENT_END for _ in range(self.ngram_order - 2)]
        output[-1] = tuple(list(output[-1]) + add_to_end)
        if output[0] != (self.SENT_BEGIN,) and self.ngram_order == 1:
            output.insert(0, (self.SENT_BEGIN,))
        
        return self.sew_together(output)
        
    
    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate
        Returns:
          list: a list containing strings, one per generated sentence
        """
        output = []
        
        self.log += f"Generating {n} sentences for {self.ngram_order}-gram model:\n"
        
        for _ in range(n):
            sent = self.generate_sentence()
            output.append(sent)
            self.log += f"{sent}\n"
        
        return output



def load_sentences(testing_file_path):
    """Reads the testing data line by line and returns a list of lines.
    Parameters:
        testing_file_path (str) the location of the data to be read
    Returns:
        list: the list of sentences to test
    """
    output = []
    with open(testing_file_path, 'r') as f_read:
        for line in f_read.readlines():
            line = line.strip()
            if line:
                output.append(line)

    return output
    
    
def create_histogram(values, f_path_1, f_path_2, file_save_name=None):
    """Plot a histogram showing the relative frequency of the probabilities.
    Returns:
      None
    """

    f = plt.figure()
    
    all_vals = values[0] + values[1]
    overall_min = min(all_vals)
    min_exponent = np.floor(np.log10(np.abs(overall_min)))
    plt.hist(values, bins=np.logspace(np.log10(10 ** min_exponent), np.log10(1.0)), label=["label1", "label2"],
             stacked=True)
    plt.xlabel("Probability of test sentence")
    plt.ylabel("Frequency of score")
    plt.xscale('log')
    plt.legend([f_path_1, f_path_2])
    plt.title("Test Score vs. Frequency of Score")
    plt.show()
    
    if file_save_name is None:
        print("Figure not saved")
    else:
        f.savefig(file_save_name)
    
        
def main():
    
    # Create the models
    unigram_model = LanguageModel(1, True)
    bigram_model = LanguageModel(2, True)

    # Load in the testing data
    testing_data_1 = load_sentences(sys.argv[2])
    testing_data_2 = load_sentences(sys.argv[3])
    
    # Train, generate sentences
    unigram_model.train(sys.argv[1])
    print("-----SENTENCES FROM UNIGRAMS-----")
    for sent in unigram_model.generate(50):
        print(sent)
    unigram_scores_tests_1 = []
    unigram_scores_tests_2 = []
    for sent in testing_data_1:
        unigram_scores_tests_1.append(unigram_model.score(sent))
    for sent in testing_data_2:
        unigram_scores_tests_2.append(unigram_model.score(sent))

    # Train, generate sentences
    bigram_model.train(sys.argv[1])
    print("-----SENTENCES FROM BIGRAMS-----")
    for sent in bigram_model.generate(50):
        print(sent)
    bigram_scores_tests_1 = []
    bigram_scores_tests_2 = []
    for sent in testing_data_1:
        bigram_scores_tests_1.append(bigram_model.score(sent))
    for sent in testing_data_2:
        bigram_scores_tests_2.append(bigram_model.score(sent))

    # Create histograms and save as file
    savename_1 = "hw2-unigram-histogram.pdf"
    savename_2 = "hw2-bigram-histogram.pdf"
    create_histogram([unigram_scores_tests_1, unigram_scores_tests_2], sys.argv[2], sys.argv[3], savename_1)
    create_histogram([bigram_scores_tests_1, bigram_scores_tests_2], sys.argv[2], sys.argv[3], savename_2)
    
    # Calculate perplexity for each file and model
    print("\nPerplexity for 1-grams:")
    perplexity_1_1 = unigram_model.perplexity(" ".join(testing_data_1[:10]))
    perplexity_1_2 = unigram_model.perplexity(" ".join(testing_data_2[:10]))
    print(f"hw2-test.txt: {perplexity_1_1}")
    print(f"hw2-my-test.txt: {perplexity_1_2}")

    print("\nPerplexity for 2-grams:")
    perplexity_2_1 = bigram_model.perplexity(" ".join(testing_data_1[:10]))
    perplexity_2_2 = bigram_model.perplexity(" ".join(testing_data_2[:10]))
    print(f"hw2-test.txt: {perplexity_2_1}")
    print(f"hw2-my-test.txt: {perplexity_2_2}")


    
if __name__ == '__main__':
    

    if len(sys.argv) != 4:
        print("Usage:", "python hw2_lm.py training_file.txt textingfile1.txt textingfile2.txt")
        sys.exit(1)
    
    main()
