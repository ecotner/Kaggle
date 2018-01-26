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
    data = pd.read_csv(TRAIN_DATA_PATH)
    X_train, Y_train = process_data2(data)
    print(X_train)












