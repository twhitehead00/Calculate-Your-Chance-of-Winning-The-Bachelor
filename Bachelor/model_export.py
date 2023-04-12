import pickle
import pandas as pd

with open('model.pkl', 'rb') as f:
    trained_model = pickle.load(f)

def predict(data):
    # code for making predictions using the trained_model
    return trained_model.predict(data)[0]

def preprocess(data):
    # code for preprocessing the data
    data['fir?'] = data.pop('FIR')
    data.pop('name')
    
    data = pd.DataFrame.from_dict([data])
    data = pd.get_dummies(data, columns=['hometown', 'note', 'job_category', 'race', 'fir?', 'joke_entrance'])
    
    columns = ['season', 'age', '1-on-1_week', 'hometown_INTERNATIONAL', 'hometown_NE', 'hometown_NW', 'hometown_SE', 'hometown_SW', 'note_0', 'note_DQ', 'note_quit', '0', 'job_category_CORPORATE', 'job_category_OTHER', 'job_category_POLITICS', 'job_category_TRADITIONAL', 'job_category_TRADES',
               'race_Asian', 'race_Black', 'race_Hispanic', 'race_Middle eastern', 'race_White', 'fir?_0', 'fir?_Yes', 'joke_entrance_Gimmick', 'joke_entrance_Regular']
    
    data = data.reindex(columns=columns, fill_value=0)
    data.columns = [clean_column_names(col) for col in data.columns]
    return data

# clean column names 
def clean_column_names(column_name):
    if column_name.startswith('race_'):
        return column_name.replace('race_', '')
    elif column_name.startswith('joke_entrance_'):
        return column_name.replace('joke_entrance_', '')
    elif column_name.startswith('fir?'):
        return column_name.replace('fir?', 'FIR')        
    elif column_name.startswith('1-on-1_week'):
        return column_name.replace('1-on-1_week', '1_on_1')  
    elif column_name.startswith('hometown_'):
        return column_name.replace('hometown_', '') 
    elif column_name.startswith('job_category_'):
        return column_name.replace('job_category_', '') 
    else:
        return column_name

