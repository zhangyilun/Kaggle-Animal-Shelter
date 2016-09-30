# import 
import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
import re

import warnings
warnings.filterwarnings('ignore')

# data
train = pd.read_csv("data/train_clean_3.csv")
test = pd.read_csv("data/test_clean_3.csv")

def update_sex(df):
    unique_sex = ['Neutered Male', 'Spayed Female', 'Intact Male', 'Intact Female', 'Unknown']
    df = df[df["SexuponOutcome"].isin(unique_sex)]
    df["Male"] = df["SexuponOutcome"].apply(lambda x: 1 if x.endswith("Male") else 0)
    df["Female"] = df["SexuponOutcome"].apply(lambda x: 1 if x.endswith("Female") else 0)
    df["SexType"] = df["SexuponOutcome"].apply(lambda x: x.split(" ")[0])
    df = pd.get_dummies(df,columns=["SexType"], prefix="SexType")
    # df = df.drop("SexuponOutcome",1)
    return df

def map_age(s):
    if type(s) == float:
        return -1
    if "year" in s:
        return int(s.split(" ")[0])*365
    elif "month" in s:
        return int(s.split(" ")[0])*30.5
    elif "week" in s:
        return int(s.split(" ")[0])*7
    elif "day" in s:
        return int(s.split(" ")[0])
    
def update_age(df):
    df["AgeuponOutcome"] = df["AgeuponOutcome"].apply(map_age)
    train_age_means_by_outcome = dict(train.groupby("OutcomeType")["AgeuponOutcome"].mean())
    # set value of 0 to outcome group mean
    for i in train[train.AgeuponOutcome == -1].index:
        df.ix[i,"AgeuponOutcome"] = train_age_means_by_outcome[df.ix[i,"OutcomeType"]]
    return df

# get unique_breed
'''
all_breeds = list(train["Breed"].apply(lambda x: x.split("/")))

unique_breed = []

for l in all_breeds:
    for s in l:
        if "," in l:
            sp = l.split(",")
            for i in l: 
                if i not in unique_breed:
                    unique_breed.append(i)
        else:
            if s not in unique_breed:
                unique_breed.append(s)
                
# get rid of the "_Mix"
unique_breed = [x[:-4] if "Mix" in x else x for x in unique_breed]
unique_breed = list(set(unique_breed))
'''

def map_breed(df):
    
    # number of "/" or "," in Breed
    df["crosses"] = df["Breed"].apply(lambda x: len(re.split(",|/",'Shetland Sheepdog Mix/Dog')))
    # check if it is a mix
    df["mix"] = df["Breed"].apply(lambda x: 1 if "Mix" in x else 0)
    # get rid of the "_Mix" at the end if there is one
    df["Breed"] = df["Breed"].apply(lambda x: x[:-4] if "Mix" in x else x)
    
    # dummy mapping for all unique breeds
    for b in unique_breed:
        df["Breed_"+b] = df["Breed"].apply(lambda x: b in x)
    # replace the spaces in column names with "_"
    df.columns = [x.replace(" ","_") for x in df.columns]
    
    '''
    # get all breed cols in train
    train_breed_only = train[[x for x in train.columns if x.startswith("Breed_")]]
    # get top 100 names
    train_top_100_names = train_breed_only.sum(axis=0).sort_values(ascending=False)[:100].index
    # get the rest column names
    train_else_names = train_breed_only.sum(axis=0).sort_values(ascending=False)[100:].index
    # get other cols of df
    df_else_cols = df[[x for x in df.columns if not x.startswith("Breed_")]]
    # combine other columns with columns of top 100 breeds
    df_new = pd.concat([df_else_cols, df[train_top_100_names]], axis=1)
    # add a new column for other breeds (binary)
    df_new["Breed_Other"] = df[train_else_names].sum(axis=1).apply(lambda x: 1 if x>0 else 0)
    '''
    
    # df_new = df_new.drop("Breed",1) 
    # return df_new
    return df

'''
# get unique color
all_color = list(train["Color"].apply(lambda x: x.split("/")))

unique_color = []
unique_description = []

for l in all_color:
    for s in l:
        if len(s.split(" ")) > 1:
            if s.split(" ")[1] not in unique_description:
                unique_description.append(s.split(" ")[1])
        if s not in unique_color:
            unique_color.append(s)

unique_color = list(set([x.split(" ")[0] for x in unique_color] + unique_description))
'''

def map_color(df):
    for c in unique_color:
        df["Color_"+c] = df["Color"].apply(lambda x: c in x)
    # number of mix colors
    df["Color_num_mix"] = df["Color"].apply(lambda x: len(re.split(",|/",x)))
    # df = df.drop("Color",1)
    return df

def map_animal(df):
    df["animal_is_dog"] = df["AnimalType"].apply(lambda x: "Dog" in x)
    # df = df.drop("AnimalType",1)
    return df

def clean_time(df):
    df["Year"] = df["DateTime"].apply(lambda x: x.split(" ")[0].split("-")[0])
    df["Month"] = df["DateTime"].apply(lambda x: x.split(" ")[0].split("-")[1])
    df["Day"] = df["DateTime"].apply(lambda x: x.split(" ")[0].split("-")[2])
    df["Weekday"] = df["DateTime"].apply(lambda x: parse(x.split(" ")[0]).weekday())
    df["Hour"] = df["DateTime"].apply(lambda x: x.split(" ")[1].split(":")[0])
    df["Minute"] = df["DateTime"].apply(lambda x: x.split(" ")[1].split(":")[1])
    # df = df.drop("DateTime",1)
    return df

def clean_name(df,df_type):
    
    # if there is a name
    df["has_name"] = df["Name"].apply(lambda x: type(x) != float)
    
    # name length
    # df["name_length"] = df["Name"].apply(lambda x: len(x) if type(x) != float else 0)
    
    # initial and map to int
    '''
    df["initial"] = df["Name"].apply(lambda x: x[0] if type(x) != float else "!")
    if df_type == "train":
        df["initial"][df["initial"].str.contains("'")] = "S"
        df["initial"][df["initial"].str.contains("0")] = "J"
        df["initial"][df["initial"].str.contains("3")] = "B"
        df["initial"][df["initial"].str.contains(" ")] = "J"
        df["initial"][df["initial"].str.contains(" ")] = pd.Series(["J","M"])
    else:
        df["initial"][df["initial"].str.contains("0")] = "!"
        df["initial"][df["initial"].str.contains("3")] = "!"
        df["initial"][df["initial"].str.contains("6")] = "!"
    D = {i:j for i,j in zip(list("!ABCDEFGHIJKLMNOPQRSTUVWXYZ"),list(range(27)))}
    df["initial"] = df["initial"].replace(D)
    '''
    
    # df = df.drop("Name",1)
    return df


train = update_sex(train)
test = update_sex(test)

#train = update_age(train)
#test = update_age(test)
#
#train_after_breed = map_breed(train)
#test_after_breed = map_breed(test)
#
#train = train_after_breed
#test = test_after_breed
#
#train = map_color(train)
#test = map_color(test)

train = map_animal(train)
test = map_animal(test)

#train = clean_time(train)
#test = clean_time(test)

#train = clean_name(train,"train")
#test = clean_name(test,"test")

# delete columns transformed
train = train.drop("SexuponOutcome",1)
#train = train.drop("Breed",1)
#train = train.drop("Color",1)
#train = train.drop("DateTime",1)
#train = train.drop("Name",1)
train = train.drop("AnimalType",1)

test = test.drop("SexuponOutcome",1)
#test = test.drop("Breed",1)
#test = test.drop("Color",1)
#test = test.drop("DateTime",1)
#test = test.drop("Name",1)
test = test.drop("AnimalType",1)

#train = train.drop("OutcomeSubtype",1)
#train = train.drop("AnimalID",1)

print(train.shape)
print(test.shape)

train.to_csv("data/train_clean_31.csv",index=False)
test.to_csv("data/test_clean_31.csv",index=False)

