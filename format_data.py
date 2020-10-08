import sys
from nltk import sent_tokenize
from random import shuffle

def main():
    """
    Tokenizes the text by sentence, removes undesirable punctuation, and saves
    the file with correct number of SENT_BEGIN and SENT_END tokens.
    Parameters:
        None
    Returns:
        None
    """
    with open(sys.argv[1], "r") as to_format:
        text = to_format.read()
        
    sentences = sent_tokenize(text)
    for idx in range(len(sentences)):
        sentences[idx] = clean_sentence(sentences[idx])
    
    shuffle(sentences)
    
    with open(sys.argv[2], "w+") as to_write:
        prepend = ["<s>" for _ in range(int(sys.argv[3]) - 1)]
        postpend = ["</s>" for _ in range(int(sys.argv[3]) - 1)]
        for sent in sentences:
            to_write.write(" ".join(prepend) + " " + sent + " " + " ".join(postpend) + "\n")
        
            
def clean_sentence(sent):
    """Removes undesirable punctuation from the given sentence.
    Parameters:
        sent (str) The string sentence to clean.
    Returns:
        str: The cleaned sentence.
    """
    sent = sent.replace(".", "")
    sent = sent.replace(",", "")
    sent = sent.replace(":", "")
    sent = sent.replace(";", "")
    sent = sent.replace("!", "")
    sent = sent.replace("?", "")
    sent = sent.replace("<", "")
    sent = sent.replace(">", "")
    sent = sent.replace("-", "")
    sent = sent.replace("/", "")
    sent = sent.replace("\\", "")
    sent = sent.replace("{", "")
    sent = sent.replace("}", "")
    sent = sent.replace("%", "")
    sent = sent.replace("|", "")
    sent = sent.replace("_", "")
    sent = sent.replace(".", "")
    sent = sent.replace("”", "")
    sent = sent.replace("“", "")
    sent = sent.replace("\n", "")
    return sent


if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        raise Exception("Usage: python3 format_data.py [unformatted_file_path] [file_save_path] [n-gram_order]")
    main()