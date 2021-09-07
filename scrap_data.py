import re
import json
import time
import pymongo
from datetime import datetime
from selenium import webdriver


# Update index collection
def update_count(dbase):
    coll_index = dbase['collection_index']
    for i in coll_index.find():
        cont = dbase[i['collection_name']].count_documents({})
        if cont > 0:
            data = dbase[i['collection_name']].find().sort('datePosted', pymongo.ASCENDING)
            ads_date = [datetime.strptime(i['datePosted'].split('+')[0], '%Y-%m-%dT%H:%M').strftime('%Y-%m-%d %H:%M')
                        for i in data]
            coll_index.update_one({'_id': i['collection_name']},
                              {'$set': {'count': cont, 'first_date': ads_date[0], 'last_date': ads_date[-1]}})
        else:
            continue


def tj_collect(mongo_col, position, job_country=False, limit_obs=10000):
    # To solve automatic window close problem
    option = webdriver.ChromeOptions()
    option.add_experimental_option("detach", True)

    # Webdriver setting
    wd = webdriver.Chrome('chromedriver.exe', options=option)
    wd.implicitly_wait(30)

    # Define base url for the search
    if job_country:
        url_base = 'https://www.totaljobs.com/jobs/%s/in-%s?radius=30' % (position, job_country)
    else:
        url_base = 'https://www.totaljobs.com/jobs/%s?' % position

    # Add the page number to the base url
    num_page = 1
    url_page = url_base + '&page=%d' % num_page + '&sort=2'
    wd.get(url_page)
    wd.refresh()

    # Accept cookies
    if len(wd.find_elements_by_id('scrollableContent')) > 0:  # Accept cookies
        wd.find_element_by_id('ccmgt_explicit_accept').click()
    else:
        pass

    # The loop continue until some of the conditions are violated
    exc = 1
    n_obs = 0  # count the number of observations collected
    cont_rep_id = 0  # count the number of consecutive times requested job ads is already in the database
    no_term = 0  # count the number of consecutive times none of requested terms is in the description or title
    max_page = int(wd.find_elements_by_xpath("//a[@class='PageLink-sc-1v4g7my-0 gwcKwa']")[-1].text)

    try:
        while exc:
            # For each job offer in the page
            for job in wd.find_elements_by_xpath("//a[@data-at='job-item-title']"):  # for each job position in the page

                # For each job shown in the page, open a new tab
                job_url = job.get_attribute('href')
                job_id = job_url.split('job')[-1]

                # Check if all the term are in title if we are scrap job
                terms = position.split('-')
                cond = (all([re.search(r'\W+' + term + r'\W+|^' + term + r'|' + term + r'$',
                                       job.find_elements_by_tag_name('h2')[0].text, re.IGNORECASE) for term in terms]))
                job_in_db = mongo_col.find({'_id': job_id}, {}).count() > 0  # check if data is already in db
                if job_in_db or (mongo_col.database.name == 'jobs' and cond is False):
                    cont_rep_id += 1  # count the # of consecutive times requested job ads is already in the database
                    continue

                window2 = "window.open('" + job_url + "')"
                wd.execute_script(window2)  # Open a new tab
                wd.switch_to.window(wd.window_handles[1])  # Switch the tab
                time.sleep(5)

                # Check if the ads info is available
                try:
                    jobdata = json.loads(wd.find_element_by_id('jobPostingSchema').get_attribute('innerHTML'))
                    # Set _id with JobId of ads in totaljobs
                    jobdata['_id'] = wd.find_element_by_id('jobId').get_attribute('value')

                    # Salary
                    if len(wd.find_elements_by_xpath("//li[@class='salary icon']/div")) > 0:
                        jobdata['salary'] = wd.find_element_by_xpath("//li[@class='salary icon']/div").text

                    # Employment type
                    emp_type = 0
                    num_refresh = 0
                    while emp_type == 0:
                        wd.refresh()
                        time.sleep(5)
                        try:
                            if len(wd.find_elements_by_xpath("//li[@class='job-type icon']/div")) > 0:
                                jobdata['emp_type'] = wd.find_element_by_xpath("//li[@class='job-type icon']/div").text
                            emp_type = 1
                        except Exception as error_record:
                            print('Error occurred: %s, refresh' % error_record)
                            emp_type = 0
                            num_refresh += 1
                        if num_refresh >= 5:
                            emp_type = 0

                    # Close the job tab
                    wd.close()
                    wd.switch_to.window(wd.window_handles[0])
                except Exception as error_record:
                    print('Ads not available with error: %s' % error_record)
                    # Close the job tab
                    wd.close()
                    wd.switch_to.window(wd.window_handles[0])
                    continue

                # Check if the requested term is in the description or title
                if mongo_col.database.name == 'topics':
                    terms = [term.replace('-', ' ') for term in position.split('_')]
                    cond = (any([re.search(r'\W+' + term + r'\W+|^' + term + r'|' + term + r'$',
                                           jobdata['description'], re.IGNORECASE) for term in terms])
                            or
                            any([re.search(r'\W+' + term + r'\W+|^' + term + r'|' + term + r'$',
                                           jobdata['title'], re.IGNORECASE) for term in terms]))
                else:
                    terms = position.split('-')
                    cond = (all([re.search(r'\W+' + term + r'\W+|^' + term + r'|' + term + r'$',
                                           jobdata['title'], re.IGNORECASE) for term in terms]))

                if cond:
                    try:
                        # Insert data to MongoDB
                        mongo_col.insert_one(jobdata)
                        n_obs += 1
                        print(n_obs, jobdata['title'])
                        no_term = 0  # num of consecutive times none of requested terms is in the description/title
                        cont_rep_id = 0
                    except pymongo.errors.DuplicateKeyError:  # When the id is already in mongodb
                        print('Data already exists in database')
                        cont_rep_id += 1

                else:
                    no_term += 1

                if n_obs >= limit_obs:
                    exc = 0
                    continue

            # Check if there is a next page bottom
            if (cont_rep_id > 20  # count the number of consecutive times requested job ads is already in the database
                    or (no_term >= 10)  # 10 or more consecutive ads do not contain the searched term
                    or (num_page >= max_page)  # No more pages for the searched term
                    or (wd.find_element_by_tag_name('h1').text == 'Access Denied')):  # Access denied
                exc = 0
                print(
                    'Condition unaccomplished.',
                    'number of repeated ads: %d, number of consecutive no term ads: %d, num_page: %d, max_page: %d' %
                    (cont_rep_id, no_term, num_page, max_page))
                continue
            else:
                num_page += 1
                # Add the page number to the base url
                url_page = url_base + '&page=%d' % num_page + '&sort=2'
                wd.get(url_page)
                print('Page %d' % num_page)

                if len(wd.find_elements_by_xpath(
                        "//div[@id ='sec-if-container']")) > 0:  # If there appear a window of access denied
                    wd.get(url_base + '&page=1&sort=2')
                    time.sleep(6)
                    wd.get(url_page)
                else:
                    pass
    except Exception as error_reason:
        print('The program has been finished by a error: %s' % error_reason)

    wd.close()
