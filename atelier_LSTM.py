import tensorflow as tf
import json
import requests
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error  # nouveau package : scikit-learn

########## API

endpoint = "https://min-api.cryptocompare.com/data/histoday"
res = requests.get(endpoint + "?fsym=BTC&tsym=USD&limit=500")


hist = pd.DataFrame(json.loads(res.content)["Data"])
copie_data = pd.DataFrame(json.loads(res.content)["Data"])
print("++++++++++")
print(copie_data)

hist = hist.set_index("time")

hist.index = pd.to_datetime(hist.index, unit="s")

# colonne cible
target_col = "close"

# maj API, suppression des 2 colonnes string
hist = hist.drop(["conversionType", "conversionSymbol"], axis=1)

###################### SAVE MODEL ######################

path = "C:\\Users\crypt\Desktop\\apocabot\deep_learning"

filename = tf.saved_model.Asset("test.txt")


@tf.function(input_signature=[])
def func():
    return tf.io.read_file(filename)


trackable_obj = tf.train.Checkpoint()
trackable_obj.func = func
trackable_obj.filename = filename
tf.saved_model.save(trackable_obj, path)

###################### FONCTIONS ######################

# séparation des données pour test,
# ici 20% du data set est inclus dans l'ensemble d'entrainement
# normalement il faut diviser en 3 (entrainement, validation, test - 60 / 20 /20 )


def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]

    test_data = df.iloc[split_row:]

    return train_data, test_data


train, test = train_test_split(hist, test_size=0.2)


####### graphique des données d'entrainement par rapport au test (affichagevia plt.show())


def line_plot(line1, line2, label1=None, label2=None, title="", lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel("prix [EUR]", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="best", fontsize=16)
    plt.show()


line_plot(train[target_col], test[target_col], "training", "test", title="")


def normalise_zero_base(df):
    return df / df.iloc[0] - 1


def normalise_min_max(df):
    return (df - df.min()) / (df.max() - df.min())


def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx : (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)

    return np.array(window_data)


# préparation de la data via la fonction de fenetre pour mouliner dans un format digestible par le réseau de neurones

# format de fenetre de 10 dans ce setup?


def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)

    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values

    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test


# création du réseau de neurones modèle LSTM


def build_lstm_model(
    input_data,
    output_size,
    neurons=128,
    activ_func="linear",
    dropout=0.2,
    loss="mse",
    optimizer="adam",
):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
    return model


# paramétrage du modèle

np.random.seed(42)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 128
epochs = 100
batch_size = 32
loss = "mse"
dropout = 0.2
optimizer = "adam"


##################### banc d'entrainement #####################

train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size
)
model = build_lstm_model(
    X_train,  # data entrée (deux entrée, shape 1, et 2 voir ligne 135)
    output_size=1,
    neurons=lstm_neurons,
    dropout=dropout,
    loss=loss,
    optimizer=optimizer,
)


history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True
)

##############################################################

targets = test[target_col][window_len:]  # équivalent Y test, les valeurs cibles du test


preds = model.predict(X_test).squeeze()  # plage de prédiction (timestamp?) VALEUR
# nouveau : cours d'ouverture bougie dont la cible est la cloture

print("--------------- ICI ----------------")
print(X_test)
print("--------------- ICI ----------------")

####### GRAPHIQUE FINAL #######
# sert a ajouter le cours visé à la variation projetée
preds = test[target_col].values[:-window_len] * (
    preds + 1
)  # 95 valeurs # valeur visée par la prédiction
preds = pd.Series(
    index=targets.index, data=preds
)  # formatage data pour affichage graphique
line_plot(targets, preds, "actual", "prediction", lw=3)
########################################


# changer le data set qui est en cloture par les ouvertures avec changement du windows len, changer le date set par une seule valeur
####################### PROCESSUS :
# un jeu de données en train
# un jeu de données en test, pour voir si le modèle est calibré,
# un jeu de modèle en "réel"

# sau eht
def evaluation():
    try:
        # print(dir(Donnees_X_abcisses_test)) checker si variable est __iter__

        loss, accuracy = model.evaluate(x=X_test, y=y_test)
        print(f"Précision du modèle ; loss : {loss} |,accuracy : {accuracy}")
    except Exception as e:
        print(e)


evaluation()
