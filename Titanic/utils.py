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

def process_data3(data):
    '''
    Constructs feature vector column by column.
    '''
    import numpy as np
    import pandas as pd
    import re
    
#    # Detect any columns with missing data (hint, missing entries in Age [263], Cabin [1014], Embarked [2], Fare [1], and Survived [418] (just from the test set though))
#    print('Missing data entries:')
#    for label in data:
#        nan_count = 0
#        for row in data[label]:
#            if type(row) in [np.dtype(s) for s in ['int64', 'int32', 'int', 'float64', 'float32', 'float']]:
#                if np.isnan(row):
#                    nan_count += 1
#        print('{}: {}'.format(label, nan_count))
#    raise Exception('just gonna throw an error here to stop execution')
    
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
    
    # DATA CLEANING/IMPUTATION
    
    # Extract each passenger's title
    # Available titles are 'Master.', 'Jonkheer.', 'Mlle.', 'Rev.', 'Col.', 'Sir.', 'Dr.', 'Ms.', 'Major.', 'Lady.', 'Mme.', 'Miss.', 'Dona.', 'Capt.', 'Don.', 'Countess.', 'Mrs.', and 'Mr.'
    titles = []
    title_count = {}
    p = re.compile(r'\w+\.')
    for name in names:
        title_ = re.search(p, name).group()
        title_count[title_] = title_count.get(title_, 0) + 1
        titles.append(title_)
    
    # Calculate average age for person given their title, imputate missing values
    age_by_title = {}
    for i, title in enumerate(titles):
        if not np.isnan(age[i]):
            age_by_title[title] = age_by_title.get(title, []) + [age[i]]
    avg_age_by_title = {title: np.mean(age_by_title[title]) for title in age_by_title}
#    print(avg_age_by_title)
    for i, a in enumerate(age.copy()):
        if np.isnan(a):
            age[i] = avg_age_by_title[titles[i]]
    
    # Replace missing embarkation points with Southampton ('S')
    for i in range(len(embarked)):
        if type(embarked[i]) != str:
            if np.isnan(embarked[i]):
                embarked[i] = 'S'
    
    # Replace missing fares with average over same pclass
    fares_by_pclass = {1:[], 2:[], 3:[]}
    for i in range(len(fare)):
        if not np.isnan(fare[i]):
            fares_by_pclass[pclass[i]].append(fare[i])
    for i in range(len(fare)):
        if np.isnan(fare[i]):
            fare[i] = np.mean(fares_by_pclass[pclass[i]])
    
    print('Missing data entries:')
    for column, label in [(age, 'Age'), (cabin, 'Cabin'), (embarked, 'Embarked'), (fare, 'Fare'), (name, 'Name'), (parents_children, 'Parch'), (passenger_id, 'PassengerId'), (pclass, 'Pclass'), (sex, 'Sex'), (siblings_spouses, 'SibSp'), (Y, 'Survived'), (ticket, 'Ticket')]:
        nan_count = 0
        for row in column:
            if type(row) in [np.dtype(s) for s in ['int64', 'int32', 'int', 'float64', 'float32', 'float']]:
                if np.isnan(row):
                    nan_count += 1
        print('{}: {}'.format(label, nan_count))
    
    # FEATURE ENGINEERING
    
    # Now go through features (x) one by one:
    x = {}
    # Leave passenger_id as-is; it will not be used as it is totally arbitrary
    # Map pclass from (1,2,3) -> (-1,0,+1)
    pclass = pclass-2
    x[0] = pclass
    
    # Map Sex to binary value (male,female) -> (0,1)
    x[1] = np.where(sex == 'male', 0, 1)
    
    # Get all unique surnames and turn each one into a binary vector N[i,:]
    names_dict = {}
    names_dimension = []
    surnames = []
    for name in names:
        surname = name.split(',')[0].lower()
        surnames.append(surname)
        if surname in names_dict:
            names_dimension.append(names_dict[surname])
        else:
            names_dict[surname] = len(names_dict)
            names_dimension.append(names_dict[surname])
    N = np.zeros(shape=[len(names_dimension), len(names_dict)])
    for i, j in enumerate(names_dimension):
        N[i,j] = 1.
#    x[2] = N
    
    # Need some measure of how likely each person is to be in a group with each other
    S = np.zeros([n_examples, n_examples])
    for i in range(n_examples):
        for j in range(min(i+1, n_examples)):
            if (fare[i]==fare[j]) and (embarked[i]==embarked[j]) and (pclass[i]==pclass[j]):
                ticket_i = ticket[i].split()[-1]
                ticket_j = ticket[j].split()[-1]
                try:
                    ticket_i = int(ticket_i)
                    ticket_j = int(ticket_j)
                except:
                    ticket_i = 1e6*np.random.rand()
                    ticket_j = 1e6*np.random.rand()
                if ticket[i]==ticket[j]:
                    S[i,j] += np.inf
                elif abs(ticket_i-ticket_j) > 4:
                    S[i,j] += -1
                else:
                    if (type(cabin[i])==float) or (type(cabin[j])==float):
                        if surnames[i] == surnames[j]:
                            S[i,j] += np.inf
                        else:
                            S[i,j] += 1
                    elif cabin[i] == cabin[j]:
                        S[i,j] += np.inf
                    else:
                        S[i,j] += -np.inf
            else:
                S[i,j] += -np.inf
    S = S + S.T # Similarity matrix is symmetric, and we only computed half of it
    S = np.tanh(S) # Squash entries to be between 0 and 1
    
    # Group size will be the max of either 1) the number of siblings/spouses/parents/children, or 2) the sum across the similarity matrix
    group_size = np.maximum(siblings_spouses+parents_children, np.sum(S > 0.5, axis=1))
    x[3] = group_size
    
    # Identify people in couples
    couple = np.zeros(n_examples)
    for i in range(len(group_size)):
        if group_size[i] == 2:
            j = np.argmax(S[i,:])
            if sex[i] != sex[j]:
                couple[i] = 1
                couple[j] = 1
    x[4] = couple
    
    # Now turn the similarity matrix into a matrix representing the probability of a given passenger being in a specific "group" by taking the eigendecomposition S = V.T*D*V, then keeping only the largest eigenvalues and reconstructing the feature vectors of S using these
#    W, V = np.linalg.eigh(S)
#    A = np.matmul(S, V[:,-100:])
#    x[5] = (A - np.mean(A))/np.std(A)
    
    # Parse out the deck on which people have a cabin and turn into a binary vector
    cabin_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'T':7}
    x[6] = np.zeros([n_examples, len(cabin_dict)])
    for i, c in enumerate(cabin):
        if type(c) == str:
            x[6][i,cabin_dict[c[0]]] = 1
        else:
            pass
    
#    # Map embarkation point to binary vector
#    x[7] = np.zeros([n_examples, 3])
#    for n in range(n_examples):
#        if embarked[n] == 'C':
#            x[7][n,0] = 1.
#        elif embarked[n] == 'Q':
#            x[7][n,1] = 1.
#        elif embarked[n] == 'S':
#            x[7][n,2] = 1.
#        else:
#            x[7][n] = np.zeros(3)
    
    # Also add fare and age as features as well
    x[8] = fare
    x[9] = age
    
    # Normalize all the features
    for idx in x:
        mean = np.mean(x[idx], axis=0)
        std = np.std(x[idx], axis=0)
        x[idx] = (x[idx]-mean)/std
    
    # Concatenate all the features together
    X = np.concatenate([x[n] if (x[n].ndim == 2) else np.expand_dims(x[n], axis=1) for n in x], axis=-1)
    
    # Save for future reference
    np.save('./X.npy', X)
    np.save('./Y.npy', Y)
    
    return X, Y

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












