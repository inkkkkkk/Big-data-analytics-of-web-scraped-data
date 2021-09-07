import pymongo
from flask import Flask, render_template, request, url_for, redirect
from datetime import datetime, timedelta
from analysis import var_select, data_init, preprocess_text, analy_cluster_hier, analy_kmeans, lda_fit
from plot import plot_set
from scrap_data import update_count, tj_collect
import re
import pandas as pd
pd.options.mode.chained_assignment = None


app = Flask('job_ads_analysis')

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

try:
    mongo = pymongo.MongoClient(
        host='localhost',
        port=27017,
        serverSelectionTimeoutMS=1000
    )
    mongo.server_info()  # trigger exception if can not connect to db
except Exception as reason:
    print('ERROR - ', reason)

df_data = pd.DataFrame()
data_initial = dict()
df_cluster = dict()
topic_dict = dict()


@app.context_processor
def db_project():
    return dict(job_db_name='jobs', topic_db_name='topics')


# index for dashboard
@app.route('/')
def index():
    return redirect(url_for('search'))


@app.route('/search')
def search():
    db_job = mongo['jobs'] # Connect to jobs database
    db_topic = mongo['topics']  # Connect to topics database

    # Update collection doc count in index table
    update_count(dbase=db_job)
    update_count(dbase=db_topic)

    # Display all job tables available in mongodb
    coll_index_job = db_job['collection_index']
    coll_index_topic = db_topic['collection_index']
    job_list = [i for i in coll_index_job.find().sort([('last_date', -1)])]
    topic_list = [i for i in coll_index_topic.find().sort([('last_date', -1)])]
    return render_template('search.html', job_list=job_list, topic_list=topic_list)


# Scrap requested job offer
@app.route('/scrap_job', methods=['POST'])
def scrap_job():
    if request.form.get('collection_name') is None:  # New scraping
        if request.form.get('search_type') == 'Topic':
            db_name = 'topics'
        else:
            db_name = 'jobs'

        term = '-'.join(re.findall(r'\w+', request.form['term'])).lower()

        if request.form['location'] != '':
            location = '-'.join(re.findall(r'\w+', request.form['location'])).lower()
            loc = location.capitalize().replace('-', ' ')
            collection_name = '0'.join([term, location])
        else:
            location = False
            loc = '-'
            collection_name = term

        dbase = mongo[db_name]  # Connect to the database

        # Insert new job_location observation to collection index
        col_index = dbase['collection_index']

        # New search
        obs = {
            '_id': collection_name,
            db_name[:-1]: term.capitalize().replace('-', ' '),
            'loc': loc,
            'count': 0,
            'collection_name': collection_name,
            'first_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'last_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        col_index.insert_one(obs)

    else:  # Update function
        collection_name = request.form.get('collection_name')
        if request.form.get('db_name') == 'topics':
            db_name = 'topics'
        else:
            db_name = 'jobs'

        if request.form.get('location') == '-':
            term = collection_name
            location = False
        else:
            term = collection_name.split('0')[0]
            location = collection_name.split('0')[1]

        dbase = mongo[db_name]

    # Scrap the data
    col1 = dbase[collection_name]
    tj_collect(mongo_col=col1, position=term, job_country=location, limit_obs=10000)

    # Return to the search page
    return redirect(url_for('search'))


# Delete option - search page
@app.route('/delete', methods=["POST"])
def delete():
    # Connect to the database
    dbase = mongo[request.form['db_name']]  # imp_dbs(db_name=request.form['db_name'])

    # Delete the database
    collection_name = request.form['collection_name']
    dbase[collection_name].drop()

    # Delete the table information from the index
    observation_id = request.form['obs_id']
    dbase['collection_index'].remove({'_id': observation_id})

    return redirect(url_for('search'))


# Analysis option - search page redirect to dashboard page
@app.route('/dashboard', methods=["POST", "GET"])
def dashboard():
    global df_data
    global data_initial
    global df_cluster
    global topic_dict

    db_name = app.config.get('db_name')
    collection_name = app.config.get('collection_name')  # Indicate the job data to be analyzed

    if request.method == "POST" and request.form.get('collection_name') is not None:  # If it comes from search page
        # Extract database name
        app.config['db_name'] = request.form.get('db_name')
        db_name = request.form.get('db_name')

        # Extract collection name
        app.config['collection_name'] = request.form.get('collection_name')
        collection_name = request.form.get('collection_name')

        df_data = var_select(mongo[db_name][collection_name])
        df = df_data
        data_initial = dict()
        df_cluster = dict()
        topic_dict = dict()

    elif (request.method == "POST" and request.form.get('startDate') is not None
          and request.form.get('endDate') is not None):  # if it comes from dashboard after setting the date
        start_date = datetime.strptime(request.form.get('startDate'), "%m/%d/%Y").date().strftime('%Y-%m-%d')
        end_date = datetime.strptime(request.form.get('endDate'), "%m/%d/%Y").date().strftime('%Y-%m-%d')

        df = df_data[df_data['dateP'] >= start_date]
        df = df[df['dateP'] <= end_date]

    elif request.method == 'GET' and db_name is not None and collection_name is not None:
        df = df_data

    else:
        return redirect(url_for('search'))

    # Set max date and min date to calender
    min_datep = min(df['dateP']).date().strftime("%m/%d/%Y")
    max_datep = max(df['dateP']).date().strftime("%m/%d/%Y")

    # Data for plot in dashboard
    plot_db = plot_set(df=df, db=db_name, collection_name=collection_name)
    return render_template('dashboard.html', min_dateP=min_datep, max_dateP=max_datep,
                           plot_db=plot_db)


@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    global df_data
    global data_initial
    global df_cluster
    global topic_dict

    if request.method == 'GET' and len(data_initial) > 0:  # No new collection requested
        pass

    elif not df_data.empty and len(data_initial) == 0:  # New collection request
        df_data['description'] = preprocess_text(df_desc=df_data['description'])
        data_initial['data'] = data_init(processed_docs=df_data['description'], dict_filter=False)
        data_initial['data_filtered'] = data_init(processed_docs=df_data['description'], dict_filter=True)

    elif request.method == 'POST':  # Post with new filter for dictionary
        # Build dictionary, vocabulary and tf-idf
        no_below = int(request.form.get('no_below'))  # Min doc presented in
        no_above = float(request.form.get('no_above'))  # Fraction of total corpus size
        data_initial['data_filtered'] = data_init(processed_docs=df_data['description'], dict_filter=True,
                                                  no_below=no_below, no_above=no_above)
        df_cluster = dict()
        topic_dict = dict()

    elif request.method == 'GET' and df_data.empty:  # No collection selected
        return redirect(url_for('search'))

    return render_template('preprocess.html', data_initial=data_initial)


# Table of observations in selected database
@app.route('/table', methods=['GET'])
def job_table():
    collection_name = app.config.get('collection_name')
    if not df_data.empty:
        table_data = df_data[df_data['dateV'] > datetime.now()]
        table_data = table_data.where(pd.notnull(table_data), '-')
        table_data['disp_salary'] = table_data.apply(
            lambda x: str([x['salaryMin'], x['salaryMax']]) + '/' + str(x['unit']) if x['salaryMin'] != '-' or x[
                'salaryMax'] != '-' else '-', axis=1)
        table_data['disp_salary'] = table_data.apply(
            lambda x: str(x['value']) + '/' + str(x['unit']) if x['value'] != '-' and x[
                'disp_salary'] == '-' else x['disp_salary'], axis=1)

        table_data = table_data[['title', 'dateP', 'company', 'industry', 'region', 'disp_salary', 'url']]

        # Sort job offers by published date
        table_data.sort_values(by='dateP', ascending=False, inplace=True)
        table_data['dateP'] = table_data['dateP'].dt.date
        disp_col_name = collection_name.replace('-', ' ').replace('0', ' in ')
        return render_template('table.html', tjob=list(table_data.T.to_dict().values()), disp_col_name=disp_col_name)
    else:
        return redirect(url_for('search'))


@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    global df_data
    global data_initial
    global df_cluster

    if request.method == 'GET' and df_data.empty and len(data_initial) == 0:  # No database selected
        return redirect(url_for('search'))
    elif request.method == 'GET' and not df_data.empty and len(data_initial) == 0:  # No preprocessed data
        return redirect(url_for('preprocess'))

    if request.method == 'POST' or len(df_cluster) == 0:
        tf_idf = data_initial['data_filtered']['tf_idf']
        vocabulary = data_initial['data_filtered']['vocabulary']

        if request.method == 'POST':  # Not the first time processed
            num_cluster = int(request.form.get('num_cluster'))
            df_cluster = analy_kmeans(df=df_data, tf_idf=tf_idf, num_cluster=num_cluster, feature_names=vocabulary)

        else:
            num_cluster = 5
            # Hierarchical clustering
            analy_cluster_hier(tf_idf, df=df_data)
            df_cluster = analy_kmeans(df=df_data, tf_idf=tf_idf, num_cluster=num_cluster, feature_names=vocabulary)

        df_cluster = {'tables': list(df_cluster.T.to_dict().values()), 'num_cluster': num_cluster}
    else:
        pass
    print(df_cluster)
    return render_template('clustering.html', df_cluster=df_cluster, data_initial=data_initial)


@app.route('/topic_modeling', methods=['GET', 'POST'])
def topic_modeling():
    global topic_dict
    global df_data
    global data_initial

    if request.method == 'GET' and df_data.empty and len(data_initial) == 0:  # No database selected
        return redirect(url_for('search'))

    elif request.method == 'GET' and not df_data.empty and len(data_initial) == 0:  # No preprocessed data
        return redirect(url_for('preprocess'))

    if request.method == 'POST' or len(topic_dict) == 0:
        if request.method == 'POST':
            num_topic = int(request.form.get('num_topic'))
        else:
            num_topic = 10
        corpus = data_initial['data_filtered']['corpus']
        vocabulary = data_initial['data_filtered']['vocabulary']
        topic_dict = lda_fit(num_topic=num_topic, vocabulary=vocabulary, corpus=corpus, top_terms=20)
    else:
        pass
    return render_template('topic_modeling.html', data_initial=data_initial, topic_dict=topic_dict)


if __name__ == '__main__':
    # Declare the configuration, which will be set in search page
    app.config['collection_name'] = None
    app.config['db_name'] = None
    app.run()
