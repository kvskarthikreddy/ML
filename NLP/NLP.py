import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt_tab')  # Download the missing resource


# Sample data (can be replaced with a larger dataset)
data = [
    "I love programming in Python.",
    "Python is great for data analysis.",
    "Data science is an exciting field.",
    "I enjoy learning new libraries.",
    "Machine learning is fascinating.",
    "Natural language processing is a key area of AI.",
    "I often use NLTK for text analysis.",
    "Word clouds are fun to create.",
]

# Download NLTK resources (only needs to be run once)
nltk.download('punkt')  
nltk.download('stopwords')  

# Define stop words
stop_words = set(stopwords.words('english'))

# Preprocessing function to clean and tokenize text
def preprocess(text):
    # Tokenize text and convert to lowercase
    tokens = word_tokenize(text.lower())  
    # Keep only alphabetic tokens (removing punctuation and numbers)
    tokens = [word for word in tokens if word.isalpha()]  
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]  
    return tokens

# Process the data
processed_data = [preprocess(sentence) for sentence in data]

# Flatten the list of lists into a single list of tokens
flat_list = [item for sublist in processed_data for item in sublist]

# Calculate word frequency distribution
word_freq = nltk.FreqDist(flat_list)

# Function to generate and display the word cloud
def create_wordcloud(word_freq):
    # Create WordCloud from frequency distribution
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))  
    plt.imshow(wordcloud, interpolation='bilinear')  
    plt.axis('off')  # Turn off axis
    plt.show()

# Display the word cloud
create_wordcloud(word_freq)
