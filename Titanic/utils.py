# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:11:55 2018

Different algorithms for processing the Titanic data. Assumes the data is passed in as a pandas DataFrame with the following columns:
    
    PassengerId/Survived/Pclass/Name/Sex/Age/SibSp/Parch/Ticket/Fare/Cabin/Embarked

@author: Eric Cotner
"""

def process_data0(data):
    '''
    Most simple and straightforward way to process I can think of.
    
    Drops the PassengerId/Name/Ticket/Fare columns entirely because survival is unlikely to be based on them. Then drops the Cabin column because the vast majority of the data is missing. Finally, goes through and drops any other passenger who has missing data (a lot of people are missing entries in the Age column, but not too many; roughly 1/10 of the dataset). Also parses out the Survived column and uses that as the training labels.
    
    All columns are inherently numerical except Sex and Embarked. Sex can be considered a binary variable where (male, female) -> (0,1), and embarked will be treated as (C,Q,S) -> (0,1,2)
    '''
    import numpy as np
#    import pandas as pd
        
    # Drop the unwanted columns. Those that are left are:
    #   Survived/Pclass/Sex/Age/SibSp/Parch/Embarked
    data = data[['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked']].copy()
    
    # Map Sex/Embarked columns to numerical values
    sex = np.where(data['Sex'] == 'male', 0, 1)
    data['Sex'] = sex
    embarked = np.where(data['Embarked'] == 'C', 0, data['Embarked'])
    embarked = np.where(embarked == 'Q', 1, embarked)
    embarked = np.where(embarked == 'S', 2, embarked)
    data['Embarked'] = embarked
    data = np.array(data, dtype=float)
    
    # Drop all the examples which have missing data
    non_missing_data_row_idxs = []
    for row in range(data.shape[0]):
        if not np.any(np.isnan(data[row,:])):
            non_missing_data_row_idxs.append(row)
    data = data[non_missing_data_row_idxs,:]
    
    # Separate out input labels and data
    x = data[:,1:]
    y = data[:,0].astype(int)
    
    return x, y

'''========================================================================='''

def process_data1(data, test=False):
    '''
    Slightly more complicated process to deal with missing data. Basically just filling in the blanks.
    
    Drops the PassengerId/Name/Ticket/Fare columns entirely because survival is unlikely to be based on them. Then drops the Cabin column because the vast majority of the data is missing. Finally, goes through and drops any other passenger who has missing data (a lot of people are missing entries in the Age column, but not too many; roughly 1/10 of the dataset). Also parses out the Survived column and uses that as the training labels.
    
    All columns are inherently numerical except Sex and Embarked. Sex can be considered a binary variable where (male, female) -> (0,1), and embarked will be treated as (C,Q,S) -> (0,1,2)
    '''
    import numpy as np
#    import pandas as pd
        
    # Drop the unwanted columns. Those that are left are:
    #   Survived/Pclass/Sex/Age/SibSp/Parch/Embarked
    if not test:
        data = data[['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked']].copy()
    else:
        PassengerId = data['PassengerId'].copy().values
        data = data[['Pclass','Sex','Age','SibSp','Parch','Embarked']].copy()
    
    # Map Sex/Embarked columns to numerical values
    sex = np.where(data['Sex'] == 'male', 0, 1)
    data['Sex'] = sex
    embarked = np.where(data['Embarked'] == 'C', 0, data['Embarked'])
    embarked = np.where(embarked == 'Q', 1, embarked)
    embarked = np.where(embarked == 'S', 2, embarked)
    data['Embarked'] = embarked
    data = np.array(data, dtype=float)
    
    # Get indices of all rows which have missing data
    non_missing_data_row_idxs = []
    for row in range(data.shape[0]):
        if not np.any(np.isnan(data[row,:])):
            non_missing_data_row_idxs.append(row)
    
    # Make temp dataset without missing stuff to compute averages
    data_ = data[non_missing_data_row_idxs,:]
    data_ = np.mean(data_, axis=0)
    
    # Replace missing features with averages
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i,j]):
                data[i,j] = data_[j]
    
    # Separate out input labels and data
    if not test:
        x = data[:,1:]
        y = data[:,0].astype(int)
        return x, y
    else:
        return data, PassengerId

'''========================================================================='''

def process_data2(data, test=False):
    '''
    Fills in the blanks with averages. Also creates an extra set of features which is the person's last name.
    
    Drops the PassengerId/Ticket/Fare columns entirely because survival is unlikely to be based on them. Then drops the Cabin column because the vast majority of the data is missing. Finally, goes through and drops any other passenger who has missing data (a lot of people are missing entries in the Age column, but not too many; roughly 1/10 of the dataset). Also parses out the Survived column and uses that as the training labels.
    
    All columns are inherently numerical except Sex and Embarked. Sex can be considered a binary variable where (male, female) -> (0,1), and embarked will be treated as (C,Q,S) -> (0,1,2)
    '''
    import numpy as np
#    import pandas as pd
        
    # Drop the unwanted columns. Those that are left are:
    #   Survived/Pclass/Name/Sex/Age/SibSp/Parch/Embarked
    if not test:
        names = data['Name'].copy().values
        data = data[['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked']].copy()
    else:
        PassengerId = data['PassengerId'].copy().values
        names = data['Name'].copy().values
        data = data[['Pclass','Sex','Age','SibSp','Parch','Embarked']].copy()
    
    # Map Sex/Embarked columns to numerical values
    sex = np.where(data['Sex'] == 'male', 0, 1)
    data['Sex'] = sex
    embarked = np.where(data['Embarked'] == 'C', 0, data['Embarked'])
    embarked = np.where(embarked == 'Q', 1, embarked)
    embarked = np.where(embarked == 'S', 2, embarked)
    data['Embarked'] = embarked
    data = np.array(data, dtype=float)
    
    # Get indices of all rows which have missing data
    non_missing_data_row_idxs = []
    for row in range(data.shape[0]):
        if not np.any(np.isnan(data[row,:])):
            non_missing_data_row_idxs.append(row)
    
    # Make temp dataset without missing stuff to compute averages
    data_ = data[non_missing_data_row_idxs,:]
    data_ = np.mean(data_, axis=0)
    
    # Replace missing features with averages
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i,j]):
                data[i,j] = data_[j]
    
    # Create new set of features which is one-hot vector representing person's last name.
    names_dict = {}
    names_dimension = []
    for name in names:
        surname = name.split(',')[0].lower()
        if surname in names_dict:
            names_dimension.append(names_dict[surname])
        else:
            names_dict[surname] = len(names_dict)
            names_dimension.append(names_dict[surname])
    names_features = np.zeros(shape=[len(names_dimension), len(names_dict)])
    for i, j in enumerate(names_dimension):
        names_features[i,j] = 1
    # Append names features to data
    data = np.concatenate([data, names_features], axis=1)
    
    # Separate out input labels and data
    if not test:
        x = data[:,1:]
        y = data[:,0].astype(int)
        return x, y
    else:
        return data, PassengerId

'''========================================================================='''

def process_data3(data):
    '''
    Constructs feature vector column by column.
    '''
    import numpy as np
    import pandas as pd
    import re
    
#    # Detect any columns with missing data (hint, missing entries in Age [177], Cabin [687], and Embarked [2])
#    print('Missing data entries:')
#    for label in data:
#        nan_count = 0
#        for row in data[label]:
#            if type(row) in [np.dtype(s) for s in ['int64', 'int32', 'int', 'float64', 'float32', 'float']]:
#                if np.isnan(row):
#                    nan_count += 1
#        print('{}: {}'.format(label, nan_count))
#    raise Exception('just gonna throw an error here')
    
    # Create separate columns for easy manipulation of the features
    n_examples = data.shape[0]
    passenger_id = data['PassengerId'].copy().values # integer
    Y = data['Survived'].copy().values # integer
    pclass = data['Pclass'].copy().values # integer
    names = data['Name'].copy().values # string
    sex = data['Sex'].copy().values # string
    age = data['Age'].copy().values # float, missing fair number of entries
    siblings_spouses = data['SibSp'].copy().values # integer
    parents_children = data['Parch'].copy().values # integer
    ticket = data['Ticket'].copy().values # mixed string/integer
    fare = data['Fare'].copy().values # float
    cabin = data['Cabin'].copy().values # string, many missing entries
    embarked = data['Embarked'].copy().values # string, missing few entries
    
    # Now go through features (x) one by one:
    x = {}
    # Leave passenger_id as-is; it will not be used as it is totally arbitrary
    # Map pclass from (1,2,3) -> (-1,0,+1)
    x[0] = (pclass-2)
    
    # Get all unique names and turn each one into a binary vector
    names_dict = {}
    names_dimension = []
    for name in names:
        surname = name.split(',')[0].lower()
        if surname in names_dict:
            names_dimension.append(names_dict[surname])
        else:
            names_dict[surname] = len(names_dict)
            names_dimension.append(names_dict[surname])
    N = np.zeros(shape=[len(names_dimension), len(names_dict)])
    for i, j in enumerate(names_dimension):
        N[i,j] = 1.
#    x[1] = N
    
    # Also get each person's title
    titles = []
    title_count = {}
    p = re.compile(r'\w+\.')
    for name in names:
        title_ = re.search(p, name).group()
        title_count[title_] = title_count.get(title_, 0) + 1
        titles.append(title_)
    
    # Map Sex to binary value (male,female) -> (0,1)
    x[2] = np.where(sex == 'male', 0, 1)
    
    # Map embarkation point to binary vector
    x[3] = np.zeros([n_examples, 3])
    for n in range(n_examples):
        if embarked[n] == 'C':
            x[3][n,0] = 1.
        elif embarked[n] == 'Q':
            x[3][n,1] = 1.
        elif embarked[n] == 'S':
            x[3][n,2] = 1.
        else:
            x[3][n] = np.zeros(3)
    
    # Just use number of siblings/spouses/parents/children as features (and their sum) as a rough measure of group size
    x[4] = np.stack([siblings_spouses, parents_children, siblings_spouses + parents_children], axis=1)
    
    # Need some measure of how likely each person is to be in a group with each other
    S = np.zeros([n_examples, n_examples])
    for i in range(n_examples):
        for j in range(min(i+1, n_examples)):
            if np.all(N[i] == N[j]): # If they share the same last name, very likely to be in group
                S[i,j] += 1.0
            if np.all(x[3][i] != x[3][j]): # If they don't have the same embarkation, it's impossible to be in the same group
                S[i,j] += -10.
            if (ticket[i] == 'LINE') or (ticket[j] == 'LINE'): # Something's up with the LINE ticket, ignore it
                pass
            elif ticket[i] == ticket[j]: # If they share the exact same ticket number, it's certain they're in the same group
                S[i,j] += 10.
            elif abs(int(ticket[i].split()[-1]) - int(ticket[j].split()[-1])) <= 10: # If they have very similar ticket numbers, they might be in the same group
                S[i,j] += 0.5
            else:   # Radiacally different ticket numbers implies they aren't with each other
                S[i,j] += -0.2
            if pclass[i] != pclass[j]: # If they're in different passenger classes, almost certain they're not travelling together
                S[i,j] += -10.
            if (cabin[i] == cabin[j]) and (type(cabin[i]) != float) and (type(cabin) != float): # If they share the same cabin it's guaranteed they're travelling together
                S[i,j] += 10.
    S = S + S.T # Similarity matrix is symmetric, and we only computed half of it
    S = 1/(1+np.exp(-2*S)) # Squash entries to be between 0 and 1
#    x[5] = S
    
    # Now turn the similarity matrix into a matrix representing the probability of a given passenger being in a specific "group" by taking the eigendecomposition S = V.T*D*V, then keeping only the largest eigenvalues and reconstructing the feature vectors of S using these
    W, V = np.linalg.eigh(S)
    A = np.matmul(S, V[:,-600:])
    x[5] = (A - np.mean(A))/np.std(A)
    
    # Calculate average age for person given their title
    age_by_title = {}
    for i, title in enumerate(titles):
        if not np.isnan(age[i]):
            age_by_title[title] = age_by_title.get(title, []) + [age[i]]
    avg_age_by_title = {title: np.mean(age_by_title[title]) for title in age_by_title}
    for i, a in enumerate(age.copy()):
        if np.isnan(a):
            age[i] = avg_age_by_title[titles[i]]
    x[6] = age.copy()
    
    # Parse out the deck on which people have a cabin and turn into a binary vector
    cabin_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7}
    x[7] = np.zeros([n_examples, len(cabin_dict)])
    for i, c in enumerate(cabin):
        if type(c) == str:
            x[7][i,cabin_dict[c[0]]] = 1
        else:
            pass
    
    # Concatenate all the features together
    X = np.concatenate([x[n] if (x[n].ndim == 2) else np.expand_dims(x[n], axis=1) for n in x], axis=-1)
    
    # Save for future reference
    np.save('./X.npy', X)
    np.save('./Y.npy', Y)
    
    return X, Y, S

def scale_data(data, sparse_feature_idxs=[]):
    ''' Scales the data to zero mean and unit variance, unless some features are identified as sparse, then it just scales to unit variance. Assumes data is a rank-2 numpy array with batch dimension on axis 0. '''
    import numpy as np
    mean_list = []
    std_list = []
    for column in range(data.shape[1]):
        if column in sparse_feature_idxs:
            mean = 0.
        else:
            mean = np.average(data[:,column])
        std = np.std(data[:,column])
        assert std != 0, 'Zero variance detected in column {} while data scaling!'.format(column)
        mean_list.append(mean)
        std_list.append(std)
        data[:,column] = (data[:,column] - mean)/std
    return data, mean_list, std_list






if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    
    TRAIN_DATA_PATH = '../Datasets/Titanic/train.csv'
    TEST_DATA_PATH = '../Datasets/Titanic/test.csv'
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    all_data = pd.concat([train_data, test_data], axis=0)
    X, Y, S = process_data3(all_data)
    print('X.shape = {}, Y.shape = {}'.format(X.shape, Y.shape, S.shape))












