# Language Model

## About
My implementation of a N-Gram statistical language model.
This problem was assigned for homework for my Natural Language Processing course.
It uses counts of n-grams to estimate the conditional probability of some word given the previous n-gram.


## Prerequisites
NumPy and Matplotlib are required when running anything from language_model.py. You can run the following:

    pip3 install -r requirements.txt
    
    
## How to Use
You can run the main module (language_model.py) with the following command-line arguments:

    python3 language_model.py <training-file> <testing-file-1> <testing-file-2>

Running this program from the language_model.py module will train a unigram and bigram model on the training data and will evaluate the probability of the both test sets. It will produce the following output:
- 50 randomly generated sentences produced by the unigram model
- 50 randomly generated sentences produced by the bigram model
- A histogram comparing testing-file-1 data to testing-file-2 data on the unigram model
- A histogram comparing testing-file-1 data to testing-file-2 data on the bigram model
- The perplexity calculation for the first 10 sentences of the testing-file-1 data for the unigram model
- The perplexity calculation for the first 10 sentences of the testing-file-2 data for the unigram model
- The perplexity calculation for the first 10 sentences of the testing-file-1 data for the bigram model
- The perplexity calculation for the first 10 sentences of the testing-file-2 data for the bigram model

It is highly suggested that when running this module, you only use testing data formatted for unigrams and bigrams (ie. each sentence must start with the "\<s>" token and end with the  "\</s>" token.)

If you would like to use a higher-order n-gram model, you may create your own module and import the LanguageModel class.
The format_data.py module is created so that you can format any given text file into the correct format for the LanguageModel to train on. To create the formatted data, run:

    python3 format_data.py <raw-data-file> <name-of-new-data-file> <n-gram-order>

So for example, given the file raw_training_data.txt, in order to format the data for use on a 6-gram language model, the command would be:

    python3 format_data raw_training_data.txt formatted_training_data.txt 6









