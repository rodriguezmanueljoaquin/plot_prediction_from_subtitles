from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(df, column, title = None, max_words = 100):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=max_words,
        max_font_size=40, 
        scale=3,
        random_state=1,
        colormap='Dark2',
    ).generate(df.loc[:, column].str.cat(sep=" "))

    print("Showing wordcloud for column {}".format(column))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('./{}_{}.png'.format(column, max_words))
    plt.show()

def analysis(data_df):
    print("Starting analysis")
    max_words = 100
    show_wordcloud(data_df, 'subtitles', 'Most {} common words in subtitles'.format(max_words), max_words)
    show_wordcloud(data_df, 'overview', 'Most {} common words in overview'.format(max_words), max_words)

    max_words = 35
    show_wordcloud(data_df, 'subtitles', 'Most {} common words in subtitles'.format(max_words), max_words)
    show_wordcloud(data_df, 'overview', 'Most {} common words in overview'.format(max_words), max_words)

    print("Finished analysis")