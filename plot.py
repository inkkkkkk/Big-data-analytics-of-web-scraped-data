import re
import spacy
import collections
import numpy as np
from datetime import datetime
from wordcloud import WordCloud
from spacy.matcher import PhraseMatcher  # Add special vocabulary when lemmatize


def stat_descrip(df, db, collection_name):
    stat = dict()
    col_name = collection_name.split('0')[0].replace('-', ' ').capitalize()
    stat['total ads'] = df.shape[0]
    stat['date range'] = '%s - %s' % (str(min(df['dateP'].dt.date).strftime('%d/%m/%Y')),
                                      str(max(df['dateP'].dt.date).strftime('%d/%m/%Y')))
    stat['db'] = [db.capitalize(), col_name]
    stat['available ads'] = sum(df['dateV'] >= datetime.now())

    data = df[['salaryMax', 'salaryMin', 'value', 'unit']][df['country'] == 'GB']
    data = data[['salaryMax', 'salaryMin', 'value']][data['unit'] == 'year']
    data.where(data > 10000, np.nan, inplace=True)
    stat['salary'] = data.describe()

    try:
        stat['salary'] = stat['salary'].astype('int32').to_dict()
    except Exception as reason:
        print('Cannot convert to integer.', reason)
        stat['salary'] = stat['salary'].to_dict()

    stat['salary_index'] = list(stat['salary']['salaryMax'].keys())
    stat['salary']['salaryMax'] = list(stat['salary']['salaryMax'].values())
    stat['salary']['salaryMin'] = list(stat['salary']['salaryMin'].values())
    stat['salary']['value'] = list(stat['salary']['value'].values())

    return stat


# Count of job offers by date (data for the bar plot)
def freq_date(df):
    count_date = collections.Counter(df['dateP'].map(lambda x: str(x.strftime('%Y/%m'))))
    date = list(count_date.keys())
    count = list(count_date.values())

    return {'date': date, 'count': count}


# Count by region
def freq_region(df):
    total = df['region'].dropna().count()
    count_region = dict(collections.Counter(df['region'].dropna()).most_common(10))
    count_region = {key: round(value/total*100, 2) for key, value in count_region.items()}
    count_country = dict(collections.Counter(df['country'].where(df['country'] == 'GB', 'Others')))
    # count_country['UK'] = count_country.pop('GB')

    return {'count_region': count_region, 'count_country': count_country, 'all_count': total}


# Pie plot of employment type and industries
def pie_ind_emp_type(df):
    # Frequency table by industry
    count_ind = collections.Counter(df['industry'][df['industry'].notna()])
    ind = [{'name': key, 'value': value} for key, value in count_ind.items()]

    # Frequency table by employment type
    count_emp_type = collections.Counter(df['employmentType'][df['employmentType'].notna()])
    emp_type = [{'name': key, 'value': value} for key, value in count_emp_type.items()]

    leg_pie = list(count_ind.keys())
    return {'emp_type': emp_type, 'ind': ind, 'leg_pie': leg_pie}


# Salary analysis
def salary_analy(df):
    df_col = df['salary']
    # Word cloud
    word_list = []
    # Import salary vocab
    with open('./vocab_pattern/salary_vocab.txt') as f:
        salary_vocab = f.read().split('\n')
    nlp = spacy.load("en_core_web_sm")
    matcher_salary = PhraseMatcher(nlp.vocab)
    pattern = list(nlp.pipe(salary_vocab))
    matcher_salary.add('salary_vocab', None, *pattern)

    for data in df_col:
        if data is not np.nan:
            data = data.lower()
            doc = nlp(re.sub(r'[^a-z0-9]+', ' ', data))
            matches = matcher_salary(doc)
            word_list.extend([doc[start:end].lemma_ for match_id, start, end in matches])
        else:
            continue

    cont_word = collections.Counter(word_list)
    wordcloud = WordCloud(font_path='./static/assets/fonts/Comfortaa-VariableFont_wght.ttf',
                          width=1600, height=1400, random_state=1, background_color='white',
                          colormap='Paired', collocations=False).generate_from_frequencies(dict(cont_word))
    wordcloud.to_file('./static/assets/images/plot_capture/wordcloud_salary.png')
    words = []
    count = []
    for i, j in cont_word.most_common(15):
        words.append(i)
        count.append(j)
    return {'words': words, 'count': count}


# Dataset for plot
def plot_set(df, db, collection_name):
    plot_db = dict()
    plot_db['stat'] = stat_descrip(df, db, collection_name)
    plot_db['bar_freq_date'] = freq_date(df=df)
    plot_db['pie_ind_emptype'] = pie_ind_emp_type(df=df)
    plot_db['salary'] = salary_analy(df=df)
    plot_db['region'] = freq_region(df=df)
    return plot_db
