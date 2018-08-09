
import pandas as pd
import os
## Complete Training Set Info
# * no risk: 977884 (98.3%)
# * risky: 12122 (1.2%)
# * unknown: 4725 (0.5%)
# * total: 994731
# * 297 features

trn_file = "atec_anti_fraud_train.csv"
tst_file = "atec_anti_fraud_test_b.csv"

def Get_Empty_Frame(df, default = 0):
    member_srls = np.unique(df[key_srl])
    init = {key: default for key in cols}
    init[key_srl] = member_srls.astype(int)
    df_0 = pd.DataFrame(init)
    df_0 = df_0.set_index(key_srl)
    return df_0

def SQL_from_File(path):
    file_object  = open(path, "r")
    string = file_object.read()
    string = string.replace("\n", " ")
    string = string.replace("%", "%%")
    start = string.find("?")
    string= string[start+1:]
    
    return string

def String_to_File(string, path):
    text_file = open(path, "w")
    text_file.write(string)
    text_file.close()

def Read_SQL_from_File(path, engine):
    df = pd.read_sql_query(SQL_from_File(path), engine)
    return df
 
def String_from_List(values):
    sub_clause = "("
    i = 1
    for srl in values:
        if i == len(values):
            sub_clause = sub_clause + str(srl) + ")"
        else:
            sub_clause = sub_clause + str(srl) + ", "
        i = i + 1
    return sub_clause

def Get_Empty_Frame(df, cols, key_idx, idx):
    init = {key: 0 for key in cols}
    init[key_idx] = idx
    df_0 = pd.DataFrame(init, dtype = float)
    df_0 = df_0.set_index(key_idx)
    df_0.index = df_0.index.values.astype(int)
    return df_0

def drop_cols(df, cols_):
    cols = list(df.columns.values)
    for col in cols_:
        if col in cols:
            df  = df.drop([col], axis = 1)
    return df

def load_data(nrows = 994731, skip = 0):
    cols = pd.read_csv(trn_file, sep = ",", nrows = 1)
    cols = list(cols.columns.values)
    
    df = pd.read_csv(trn_file, sep = ",", skiprows = skip, nrows = nrows)
    df.columns = cols

    df = df[df.label != -1]
    
    cols_ = ["date", "id", "label"]

    labl = df["label"]
    data = drop_cols(df, cols_)
    return data, labl

def load_tst():
    df_tst = pd.read_csv(tst_file, sep = ",")
    tst_id = df_tst["id"]
    tst_data = df_tst.drop(["date", "id"], axis = 1)
    return tst_data, tst_id

def load_trn_dev(trn = 900000, dev = 94731):
    df_trn = pd.read_csv(trn_file, sep = ",", nrows = trn)
    df_dev = pd.read_csv(trn_file, sep = ",", skiprows = trn, nrows = dev)

    col_names = list(df_trn.columns.values)
    df_dev.columns = col_names

    # remove no label
    df_trn = df_trn[df_trn.label != -1]

    cols_ = ["date", "id", "label"]

    trn_labl = df_trn["label"]
    trn_data = drop_cols(df_trn, cols_)

    dev_labl = df_dev["label"]
    dev_data = drop_cols(df_dev, cols_)

    return trn_data, trn_labl, dev_data, dev_labl

def sample_trn_dev(trn = 10000, dev = 1000, random_state = 0):
    all_data, all_labl = load_data()
    all_data['label'] = all_labl
    sample = all_data.sample(trn + dev, random_state = random_state)
    
    df_trn = sample[:trn]
    df_dev = sample[trn:]

    # remove no label
    df_trn = df_trn[df_trn.label != -1]

    cols_ = ["date", "id", "label"]

    trn_labl = df_trn["label"]
    trn_data = drop_cols(df_trn, cols_)

    dev_labl = df_dev["label"]
    dev_data = drop_cols(df_dev, cols_)

    return trn_data, trn_labl, dev_data, dev_labl