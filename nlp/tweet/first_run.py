#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : first_run.py
# @Author: sl
# @Date  : 2020/10/12 - 下午9:41
import re

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from util.constant import WORK_DIR

pd.set_option('display.max_colwidth', -1)

data_dir = "{}/data/test/tweet".format(WORK_DIR)

# nltk.download('stopwords')

stop = stopwords.words('english')
print(stop)


def print_text(data):
    print(data.head())


train_data = pd.read_csv("{}/{}".format(data_dir, 'train.csv'))
print_text(train_data)

test_data = pd.read_csv("{}/{}".format(data_dir, 'test.csv'))
test_data.head()

sample_submission = pd.read_csv("{}/{}".format(data_dir, 'sample_submission.csv'))
print_text(sample_submission)

train_data = train_data.drop(['keyword', 'location', 'id'], axis=1)
print_text(train_data)


# 文本常常包含许多特殊字符，这些字符对于机器学习算法来说不一定有意义。因此，我要采取的第一步是删除这些
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df


data_clean = clean_text(train_data, "text")
print_text(data_clean)

data_clean['text'] = data_clean['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print_text(data_clean)

X_train, X_test, y_train, y_test = train_test_split(data_clean['text'], data_clean['target'], random_state=0)

pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('nb', SGDClassifier()),
])
model = pipeline_sgd.fit(X_train, y_train)

y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))

submission_test_clean = test_data.copy()
submission_test_clean = clean_text(submission_test_clean, "text")
submission_test_clean['text'] = submission_test_clean['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
submission_test_clean = submission_test_clean['text']
submission_test_clean.head()
print_text(submission_test_clean)

submission_test_pred = model.predict(submission_test_clean)

id_col = test_data['id']
submission_df_1 = pd.DataFrame({
    "id": id_col,
    "target": submission_test_pred})
submission_df_1.head()
print_text(submission_df_1)

submission_df_1.to_csv("{}/{}".format(data_dir, 'submission_1.csv'), index=False)


police_filename = "{}/{}".format(data_dir, 'police_clean.csv')
police_data = pd.read_csv(police_filename)
police_data_id = police_data["IR_No"]
police_data_line = police_data["Main_Narrative"]


police_data_line_pred = model.predict(police_data_line)

police_data_df_1 = pd.DataFrame({
    "IR_No": police_data_id,
    "target(1:bad)": police_data_line_pred})


print_text(police_data_df_1)
police_data_df_1.to_csv("{}/{}".format(data_dir, 'police_data_df_1.csv'), index=False)


if __name__ == '__main__':
    pass
