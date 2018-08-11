# ------------------ INITIALIZE ---------------------

import os, errno

# clear rubbish
def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred
silentremove("analysis.pyc")
silentremove("loader.pyc")
silentremove("cross_val.pyc")
silentremove("util_ema.pyc")

# load private code
import analysis
import loader
import util_ema as ema
import cross_val as cv
import lookalike as la
import date_utils as dt

# load open-source library
import pickle
import numpy as np
import sklearn 
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import scipy
import datetime
from datetime import timedelta
import time
from scipy import stats
from sqlalchemy import create_engine
from copy import deepcopy
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone

cwd = os.getcwd()
ls = cwd.split("/")
del ls[-1]
cwd = "/".join(ls)
del ls[-1]
root = "/".join(ls)

# data fields
key_srl = "member_srl"
key_age = "age"
key_gender = "gender"
key_dy = "sale_basis_dy"
key_ts = "ts"
key_gmv = "gmv"
key_cat = "category"
key_rate = "rate"
key_days = "days"
key_views = "views"
key_dt = "dt"
key_label = "label"

# 45 categories
key_av = "av"
key_accessory = "accessory"
key_baby = "baby"
key_babyfashion = "babyfashion"
key_beauty = "beauty"
key_beautyappliance = "beautyappliance"
key_beverages = "beverages"
key_caraccessories = "caraccessories"
key_cellphone_acc = "cellphone_acc"
key_children = "children"
key_clothesothers = "clothesothers"
key_coffee_tea = "coffee_tea"
key_computer = "computer"
key_diapering = "diapering"
key_digital = "digital"
key_fashionshoes = "fashionshoes"
key_foreignbooks = "foreignbooks"
key_formula = "formula"
key_fresh_cold_frozen = "fresh_cold_frozen"
key_generalbooks = "generalbooks"
key_healthappliance = "healthappliance"
key_healthyfood = "healthyfood"
key_home = "home"
key_household = "household"
key_kidulttoys = "kidulttoys"
key_kitchen = "kitchen"
key_livingappliance = "livingappliance"
key_mealessentials = "mealessentials"
key_menclothes = "menclothes"
key_othermedia = "othermedia"
key_pc_videogame = "pc_videogame"
key_personalcare = "personalcare"
key_petcat = "petcat"
key_petdog = "petdog"
key_petothers = "petothers"
key_seasonalappliance = "seasonalappliance"
key_snacks = "snacks"
key_sportinggoods = "sportinggoods"
key_sportsclothes = "sportsclothes"
key_sportsshoes = "sportsshoes"
key_stationery = "stationery"
key_travel = "travel"
key_wipes = "wipes"
key_womenclothes = "womenclothes"
key_blank = "blank"

# merged categories
key_babycare = "babycare"
key_fashion = "fashion"
key_furniture = "furniture"
key_hygiene = "hygiene"

file_prefix = "cols"
sub_dir = "data"
path = os.path.join(cwd, sub_dir, file_prefix + ".csv")
df = pd.read_csv(path, sep = ",", nrows = 0, index_col=None)
df = df.set_index(key_srl)
cols = list(df.columns.values)

upper_cat = ["AV",
          "Accessory",
          "Baby",
          "Baby Fashion",
          "Beauty",
          "Beauty Appliance",
          "Beverages",
          "Car Accessories",
          "Cell Phone & Acc.",
          "Children",
          "Clothes Others",
          "Coffee & Tea",
          "Computer",
          "Diapering",
          "Digital",
          "Fashion Shoes",
          "Foreign Books",
          "Formula",
          "Fresh, Cold & Frozen",
          "General Books",
          "Health Appliance",
          "Healthy food",
          "Home",
          "Household",
          "Kidult Toys",
          "Kitchen",
          "Living Appliance",
          "Meal Essentials",
          "Men Clothes",
          "Other Media",
          "PC & Video Game",
          "Personal Care",
          "Pet Cat",
          "Pet Dog",
          "Pet Others",
          "Seasonal Appliance",
          "Snacks",
          "Sporting Goods","Sports Clothes",
          "Sports Shoes",
          "Stationery",
          "Travel",
          "Wipes",
          "Women Clothes",
          "null"]
map_u2l = {}
map_l2u = {}
for i in range(len(upper_cat)):
    map_u2l[upper_cat[i]] = cols[i]
    map_l2u[cols[i]] = upper_cat[i]

class_labls = ["c" + str(i) for i in range(19)]
class_map = {
    class_labls[0] : "Infant-Toddler Mom",
    class_labls[1] : "Male Softline",
    class_labls[2] : "Homemaker I",
    class_labls[3] : "Household Buyer",
    class_labls[4] : "Infant Mom",
    class_labls[5] : "Homemaker II",
    class_labls[6] : "Cat Owner",
    class_labls[7] : "Digital and Electronics",
    class_labls[8] : "Single Upwardly Mobile",
    class_labls[9] : "Drink and Meal Buyer",
    class_labls[10] : "Caffeinated Drink Buyer",
    class_labls[11] : "Female Softline",
    class_labls[12] : "Dog Owner",
    class_labls[13] : "Traveler",
    class_labls[14] : "Healthy Food Purchaser",
    class_labls[15] : "Food Purchaser",
    class_labls[16] : "Computers and Games",
    class_labls[17] : "Personal Apperance",
    class_labls[18] : "Toddler Mom"
}

# ------------------ CODE BEGINS ---------------------

file_prefix = "sample_100_dataset_v2_d7"
sub_dir = "data"

path = os.path.join(cwd, sub_dir, file_prefix + ".csv")
df = pd.read_csv(path, sep = ",")

nans = lambda df: df[df.isnull().any(axis=1)]

# drop bad features here
df = df.drop(["gpd"], axis = 1)

df = df.dropna()

# separate label into another dataframe
df_labl = df[[key_label]]
df_labl.head()

# extract input features
df_data = df.drop([key_dt, key_label], axis = 1)
df_data.head()

# choose subset of features to use
features = ["dpq", "dsl"]

# choose train-test split ratio
ratio = 0.9

# normalize
df_data = (df_data - np.mean(df_data)) / np.std(df_data)

df_data = df[features]

rows = len(df_data)
split = int(rows * ratio)
trn_data = df_data[:split]
trn_labl = df_labl[:split].astype(int)

tst_data = df_data[split:]
tst_labl = df_labl[split:].astype(int)

print("Dataset baseline: %.5f%%"%(analysis.Baseline(df_labl)))
print("Train baseline: %.5f%%"%(analysis.Baseline(trn_labl)))
print("Test baseline: %.5f%%"%(analysis.Baseline(tst_labl)))

# NEURAL NET
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.utils import np_utils

def Build_Model():
    nn = Sequential()
    nn.add(Dense(32, activation="relu", input_shape=(len(features),)))
    nn.add(Dense(32, activation="relu"))
    nn.add(Dense(1, activation="sigmoid"))
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return nn

k = 4
num_val_samples = len(trn_data) // k
num_epochs = 80
batch_size = 256
all_scores = []
all_val_loss_history = []
all_val_acc_history = []

for i in range(k):
    print('processing fold %i'%(i + 1))
    val_data = trn_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_labl = trn_labl[i * num_val_samples : (i + 1) * num_val_samples]
    
    prt_trn_data = np.concatenate([trn_data[:i * num_val_samples],
                                  trn_data[(i + 1) * num_val_samples:]], axis = 0)
    prt_trn_labl = np.concatenate([trn_labl[:i * num_val_samples],
                                  trn_labl[(i + 1) * num_val_samples:]], axis = 0)
    
    model = Build_Model()
    history = model.fit(prt_trn_data, prt_trn_labl, 
                        validation_data = (val_data, val_labl),
                        epochs = num_epochs, batch_size = batch_size, verbose = 0)
    
    all_val_loss_history.append(history.history['val_loss'])
    all_val_acc_history.append(history.history['val_acc'])
    
    val_l, val_a = model.evaluate(val_data, val_labl, verbose = 0)
    print("Fold %i val_loss %.5f, val_acc %.5f"%(i + 1, val_l, val_a))
    all_scores.append(val_a)

# FINAL MODEL
num_epochs = 80
batch_size = 256
threshold = 0.45
model = Build_Model()
history = model.fit(trn_data, trn_labl,
                    validation_data = (tst_data, tst_labl),
                    epochs = num_epochs, batch_size = batch_size, verbose = 0)

pred = model.predict(tst_data)
analysis.err_analyze(tst_labl.values, pred, threshold, verbose = True)
tprs, fprs, ts = analysis.discretize(tst_labl.values, pred, n = 100)