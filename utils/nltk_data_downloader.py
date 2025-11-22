import nltk

def download_wordnet():
    """
    Downloads the WordNet corpus using NLTK's downloader.
    """
    nltk.download('wordnet')



if __name__ == "__main__":
    download_wordnet()
    print("WordNet corpus downloaded successfully.")