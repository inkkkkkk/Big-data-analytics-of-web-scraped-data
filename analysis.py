import re
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import html
import unicodedata
import spacy
import gensim
import pyLDAvis
import pyLDAvis.sklearn
from bs4 import BeautifulSoup
from sklearn.decomposition import LatentDirichletAllocation
from spacy.matcher import PhraseMatcher  # Add special vocabulary when lemmatize
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import ward, dendrogram  # hierachical clustering
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import collections
pd.options.mode.chained_assignment = None


# Extract salary from dataset
def salary_extract(df_salary):  # df_salary is a series, df['salary']
    df = pd.DataFrame(np.full([df_salary.shape[0], 4], np.nan))
    df.columns = ['salaryMax', 'salaryMin', 'value', 'unit']
    df.index = df_salary.index

    for row in list(df_salary.index):
        if df_salary.loc[row] is not np.nan:
            # Put strings in lowercase and remove comma between numbers
            salary = df_salary.loc[row].lower().replace(',', '')

            # Replace k by 000
            if len(re.findall(r'[\d\s]+k', salary)) > 0:
                salary = salary.replace('k', '000')

            # Annum salary
            if re.search('annu', salary) is not None or re.search('per year', salary) is not None:
                if len(re.findall(r'\d+\.?\d+.{0,2}[-|to]+.{0,2}\d+\.?\d+',
                                  salary)) > 0:  # format From £??? to £??? / £35000 - £40000
                    salary = re.findall(r'\d+\.?\d+.{0,2}[-|to]+.{0,2}\d+\.?\d+', salary)
                    salary = [float(i) for i in re.findall(r'[\d.]+', salary[0])]
                    df['salaryMin'].loc[row] = min(salary)
                    df['salaryMax'].loc[row] = max(salary)
                    df['unit'].loc[row] = 'year'
                elif len(re.findall(r'up to.*£[\d.]+', salary)) > 0:  # format up to xxx
                    salary = re.findall(r'up to.*£[\d.]+', salary)
                    salary = re.findall(r'[\d.]+', salary[0])
                    df['salaryMax'].loc[row] = float(salary[0])
                    df['unit'].loc[row] = 'year'
                elif len(re.findall(r'(?<!\d)\d{4,}(?!\d)', salary)) > 0:
                    df['value'].loc[row] = float(re.findall(r'(?<!\d)\d{4,}(?!\d)', salary)[0])
                    df['unit'].loc[row] = 'year'

            # Hourly salary
            elif re.search('per hour', salary) is not None or re.search('hour', salary) is not None:
                if len(re.findall(r'\d+\.?\d+.{0,2}[-|to]+.{0,2}\d+\.?\d+', salary)) > 0:
                    salary = re.findall(r'\d+\.?\d+.{0,2}[-|to]+.{0,2}\d+\.?\d+', salary)
                    salary = [float(i) for i in re.findall(r'[\d.]+', salary[0])]
                    df['salaryMin'].loc[row] = min(salary)
                    df['salaryMax'].loc[row] = max(salary)
                    df['unit'].loc[row] = 'hour'
                elif len(re.findall(r'up to.*£[\d.]+', salary)) > 0:  # format up to xxx
                    salary = re.findall(r'up to.*£[\d.]+', salary)
                    salary = re.findall(r'[\d.]+', salary[0])
                    df['salaryMax'].loc[row] = float(salary[0])
                    df['unit'].loc[row] = 'hour'
                elif len(re.findall(r'£[\d.]+.*hour', salary)) > 0:  # format up to xxx
                    salary = re.findall(r'£[\d.]+.*hour', salary)
                    salary = re.findall(r'[\d.]+', salary[0])
                    df['value'].loc[row] = float(salary[0])
                    df['unit'].loc[row] = 'hour'

            # Change daily salary
            elif re.search('per day', salary) is not None or re.search('day', salary) is not None:
                if len(re.findall(r'\d+\.?\d+.{0,2}[-|to]+.{0,2}\d+\.?\d+', salary)) > 0:
                    salary = re.findall(r'\d+\.?\d+.{0,2}[-|to]+.{0,2}\d+\.?\d+', salary)
                    salary = [float(i) for i in re.findall(r'[\d.]+', salary[0])]
                    df['salaryMin'].loc[row] = min(salary)
                    df['salaryMax'].loc[row] = max(salary)
                    df['unit'].loc[row] = 'day'
                elif len(re.findall(r'up to.*£[\d.]+', salary)) > 0:  # format up to xxx
                    salary = re.findall(r'up to.*£[\d.]+', salary)
                    salary = re.findall(r'[\d.]+', salary[0])
                    df['salaryMax'].loc[row] = float(salary[0])
                    df['unit'].loc[row] = 'day'
                elif len(re.findall(r'£[\d.]+.*day', salary)) > 0:  # format up to xxx
                    salary = re.findall(r'£[\d.]+.*day', salary)
                    salary = re.findall(r'[\d.]+', salary[0])
                    df['value'].loc[row] = float(salary[0])
                    df['unit'].loc[row] = 'day'
        else:
            continue

    df_num = df[['salaryMax', 'salaryMin', 'value']]
    df_num.where(df_num < 1000000, np.nan, inplace=True)
    df_num.where(df_num > 10, np.nan, inplace=True)
    df_final = pd.concat([df_num, df['unit']], axis=1)

    # depend on digit length
    return df_final


# Normalize text
def norm_text(text, no_point_digit=True):
    # Decode html entities
    text = html.unescape(text)

    # Remove html tags
    patt = re.compile('>(.*?)<')
    text = list(filter(None, patt.findall(text)))
    text = list(filter(str.strip, text))
    text = '. '.join(text).replace('\xa0', ' ')

    # Case conversion
    text = text.lower()

    # Remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove special character
    text = re.sub(r'[^a-zA-z0-9\s/.+\\]+', ' ', text)
    text = re.sub(r'[^a-zA-Z]+[\d]+[^a-zA-Z]+', ' ', text)
    text = re.sub(r'\d{2,}', ' ', text)

    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[. ]+\. +', '. ', text)

    if no_point_digit:
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)

    return text


# Pre-processing steps on the entire dataset
def preprocess_text(df_desc):
    df = []
    nlp = spacy.load('en_core_web_sm')

    # Import skill vocabulary to match skill words in document
    with open('./vocab_pattern/skills.txt') as f:  # ./vocab_pattern/skills.txt
        skills_vocab = f.read().split('\n')
    matcher_skills = PhraseMatcher(nlp.vocab)
    pattern = list(nlp.pipe(skills_vocab))
    matcher_skills.add('skills_vocab', None, *pattern)

    for i in df_desc.index:
        text = norm_text(text=df_desc.loc[i], no_point_digit=True)

        # Text tokenization and lemmatization
        sp_stopwords = nlp.Defaults.stop_words
        doc = nlp(text)

        # Match phrases in the document
        matches = matcher_skills(doc)
        word_list = [doc[start:end] for match_id, start, end in matches]
        word_list = spacy.util.filter_spans(
            word_list)  # To eliminate the overlap cases, e.g. data and data science (data science will be kept)
        word_list = [span_i.lemma_ if ' ' not in span_i.text else span_i.text for span_i in word_list]

        # Remove stopwords
        result = []
        for token in word_list:
            if token not in sp_stopwords:  # len(token) > 3 is not added because there are words like R, C
                result.append(token)
        result = [j if j != 'datum' else 'data' for j in result]
        df.append(result)
    df = pd.Series(df)
    df.index = df_desc.index
    return df


# Variable selection
def var_select(collection):
    data = collection.aggregate([
        {'$addFields': {
            'date1': {'$dateFromString': {
                'dateString': '$datePosted'}},
            'date2': {'$dateFromString': {
                'dateString': '$validThrough'}},
            'company': '$hiringOrganization.name',
            'country': '$jobLocation.address.addressCountry',
            # 'locality': '$jobLocation.address.addressLocality',
            'region': '$jobLocation.address.addressRegion',
            # 'latitude': '$jobLocation.geo.latitude',
            # 'longitude': '$jobLocation.geo.longitude'
        }},
        {'$project': {
            'title': 1,
            'dateP': {'$dateToString': {
                'format': '%Y-%m-%d %H:%M:%S',
                'date': "$date1"}},
            'dateV': {'$dateToString': {
                'format': '%Y-%m-%d %H:%M:%S',
                'date': "$date2"}},
            'description': 1,
            'company': 1,
            'industry': 1,
            'employmentType': 1,
            'country': 1,
            # 'locality': 1,
            'region': 1,
            # 'latitude': 1,
            # 'longitude': 1,
            'salary': 1,
            'url': 1}}
    ])
    df = pd.DataFrame(list(data))
    df.replace('', np.nan, inplace=True)

    df["dateP"] = pd.to_datetime(df["dateP"])
    df["dateV"] = pd.to_datetime(df["dateV"])
    df["_id"] = df["_id"].astype('int32')
    df['employmentType'] = df['employmentType'].str.capitalize()
    df.sort_values(by='dateP', inplace=True)
    df.drop_duplicates(subset=['title', 'company'], keep='last', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # # Preprocess description data
    # df['description'] = preprocess_text(df_desc=df['description'])
    #
    # Extract salary
    df = pd.concat([df, salary_extract(df_salary=df['salary'])], axis=1)
    return df


# Preprocess page
# Dictionary, TF-IDF and vocabulary
def data_init(processed_docs, dict_filter=False, no_below=5, no_above=0.5, keep_n=100000):
    # Create a dictionary
    dictionary = gensim.corpora.Dictionary(processed_docs)

    if not dict_filter:
        wordlist = {dictionary[idx]: freq for idx, freq in sorted(dictionary.dfs.items(), key=lambda x: x[1],
                                                                  reverse=True)}
        wordcloud = WordCloud(font_path='./static/assets/fonts/Comfortaa-VariableFont_wght.ttf', max_words=100,
                              width=7000, height=4000, random_state=1, background_color='white',
                              colormap='Paired', collocations=False).generate_from_frequencies(wordlist)
        wordcloud.to_file('./static/assets/images/plot_capture/wordcloud_descrip.png')
        param = {}

    else:  # Apply filter
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        wordlist = {dictionary[idx]: freq for idx, freq in sorted(dictionary.dfs.items(), key=lambda x: x[1],
                                                                  reverse=True)}
        wordcloud_after = WordCloud(font_path='./static/assets/fonts/Comfortaa-VariableFont_wght.ttf', max_words=100,
                                    width=7000, height=4000, random_state=1, background_color='white',
                                    colormap='Paired', collocations=False).generate_from_frequencies(wordlist)
        wordcloud_after.to_file('./static/assets/images/plot_capture/wordcloud_descrip_after.png')
        param = {'no_below': no_below, 'no_above': no_above}

    # Tf-idf
    corpus = [' '.join(word_list) for word_list in processed_docs]
    vocabulary = [dictionary[i] for i in range(len(dictionary))]
    pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
                     ('tfid', TfidfTransformer())]).fit(corpus)
    tf_idf = pipe.transform(corpus)
    wordlist = {'words': list(wordlist.keys())[:20], 'count': list(wordlist.values())[:20]}
    return {'tf_idf': tf_idf, 'dictionary': dictionary, 'vocabulary': vocabulary,
            'corpus': corpus, 'wordlist': wordlist, 'param': param}


# Clustering page
# Calculate Linkage Matrix using Cosine Similarity
def ward_hierarchical_clustering(feature_matrix):
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix


# Plot Hierarchical Structure as a Dendrogram
def plot_hierarchical_clusters(linkage_matrix, df, p=100, figure_size=(8, 12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size)
    df_title = df['title'].values.tolist()
    # plot dendrogram
    r = dendrogram(linkage_matrix, orientation="left", labels=df_title,
                   truncate_mode='lastp',
                   p=p,
                   no_plot=True)
    temp = {r["leaves"][ii]: df_title[ii] for ii in range(len(r["leaves"]))}

    def llf(xx):
        return "{}".format(temp[xx])

    ax = dendrogram(
        linkage_matrix,
        truncate_mode='lastp',
        orientation="left",
        p=p,
        leaf_label_func=llf,
        leaf_font_size=10,
    )
    plt.tick_params(axis='x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('./static/assets/images/plot_capture/hierarchical_clusters.png', dpi=400)


# Hierarchical clustering
def analy_cluster_hier(tf_idf, df):
    linkage_matrix = ward_hierarchical_clustering(tf_idf)
    plot_hierarchical_clusters(linkage_matrix, p=100, df=df, figure_size=(12, 14))


# K-means
def analy_kmeans(df, tf_idf, feature_names, num_cluster=5, topn_features=15, topn_jobs=15):
    km = KMeans(n_clusters=num_cluster, max_iter=10000, n_init=50, random_state=666).fit(tf_idf)
    df['kmeans_cluster'] = km.labels_
    n_clusters = dict(collections.Counter(km.labels_))

    # Group observations
    df_clusters = (df[['title', 'url', 'kmeans_cluster']]
                   .sort_values(by=['kmeans_cluster'],
                                ascending=False)
                   .groupby('kmeans_cluster').head(20))
    df_clusters = df_clusters.copy(deep=True)

    ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]

    # get key features for each cluster
    # get movies belonging to each cluster
    df_cluster_comp = pd.DataFrame(columns=['cluster_num', 'key_features', 'job_ads', 'url'])
    for cluster_num in range(num_cluster):
        key_features = [feature_names[index] for index in ordered_centroids[cluster_num, :topn_features]]
        job_ads = df_clusters[df_clusters['kmeans_cluster'] == cluster_num]['title'].values.tolist()[:topn_jobs]
        url = df_clusters[df_clusters['kmeans_cluster'] == cluster_num]['url'].values.tolist()[:topn_jobs]
        df_cluster_comp = df_cluster_comp.append({'cluster_num': str(cluster_num + 1) + ' (%d)' % n_clusters[cluster_num],
                                                  'key_features': '; '.join(key_features),
                                                  'job_ads': job_ads,
                                                  'url': url},
                                                 ignore_index=True)
    return df_cluster_comp


# Topic modeling page
# LDA model and its visualization
def lda_fit(num_topic, vocabulary, corpus, top_terms=20):
    cv = CountVectorizer(vocabulary=vocabulary)
    cv_features = cv.fit_transform(corpus)

    # lda_model = LatentDirichletAllocation(n_components=num_topic, max_iter=500, max_doc_update_iter=50,
    #                                       learning_method='online', batch_size=1740, learning_offset=50.,
    #                                       random_state=42).fit(cv_features)

    lda_model = LatentDirichletAllocation(n_components=num_topic, random_state=666).fit(cv_features)
    topic_terms = lda_model.components_
    topic_key_term_idxs = np.argsort(-np.absolute(topic_terms), axis=1)[:, :top_terms]
    topic_keyterms = np.array(vocabulary)[topic_key_term_idxs]
    topics = [', '.join(topic_keyterms[i, :].tolist()) for i in range(topic_keyterms.shape[0])]
    topic_id = ['Topic' + str(t) for t in range(1, num_topic + 1)]
    topics_df = pd.DataFrame({'topic_id': topic_id, 'topicTerm': topics})
    topic_list = list(topics_df.T.to_dict().values())

    p = pyLDAvis.sklearn.prepare(lda_model, cv_features, cv)
    pyLDAvis.save_html(p, 'lda_bigdata.html')
    lda_html = pyLDAvis.prepared_data_to_html(p, template_type='general', visid='lda_vis')
    soup = BeautifulSoup(lda_html, 'html.parser')

    with open('./static/assets/js/lda.js', 'w') as f:
        f.write(soup.script.string.lstrip())

    return {'topic_list': topic_list, 'param': {'num_topic': num_topic}}
