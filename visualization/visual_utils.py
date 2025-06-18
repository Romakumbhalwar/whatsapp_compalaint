import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def generate_wordcloud(text_data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_data))
    return wordcloud.to_image()

def plot_bar(df, column):
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f'{column} Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt
