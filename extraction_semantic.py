# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

from navec import Navec

"""
Извлекать реплики с приветствием – где менеджер поздоровался. 
Извлекать реплики, где менеджер представил себя. 
Извлекать имя менеджера. 
Извлекать название компании. 
Извлекать реплики, где менеджер попрощался.
"""


def get_sim_vectors(vector):
    sim_value = np.zeros(len(word_embedding.vocab.words))
    for i, w in enumerate(word_embedding.vocab.words):
        cos_sim = np.dot(vector, word_embedding[w]) / \
                  (np.linalg.norm(vector) * np.linalg.norm(word_embedding[w]))
        sim_value[i] = cos_sim
    sort_ids = np.argsort(sim_value)[::-1]
    vocab = np.array(word_embedding.vocab.words)
    sim_words = vocab[sort_ids]
    sim_value = sim_value[sort_ids]
    return sim_words, sim_value


def get_sim_words(word):
    sim_value = np.zeros(len(word_embedding.vocab.words))
    for i, w in enumerate(word_embedding.vocab.words):
        sim_value[i] = word_embedding.sim(word, w)
    sort_ids = np.argsort(sim_value)[::-1]
    vocab = np.array(word_embedding.vocab.words)
    sim_words = vocab[sort_ids]
    sim_value = sim_value[sort_ids]
    return sim_words, sim_value


def has_string_sim_word(string: str, list_of_check_words: [str], threshold: float) -> bool:
    words = string.lower().split()
    for w in words:
        if w not in word_embedding.vocab.words:
            continue
        for check_word in list_of_check_words:
            if word_embedding.sim(check_word, w) > threshold:
                return True
    return False


def get_phrase(df: pd.DataFrame, list_of_check_words: [str], threshold: float):
    """
    :param df: input DataFrame
    :param list_of_check_words: Each string in the list is a word, which we want to have in phrase
    :param threshold: threshold similarity of words embedding
    :return: column of booleans.
    True if row have a word synonymous to at least one check word. False is haven't
    """
    return df["text"].apply(has_string_sim_word, args=(list_of_check_words, threshold))


def get_word_from_phrase(phrase: str, list_of_check_words: [str], count=1):
    """
    :param phrase:
    :param list_of_check_words:
    :param count:
    :return:
    """
    words = phrase.split()
    max_sim_value = 0
    index_sim_word = None
    sim_word = None
    for i, w in enumerate(words):
        if w not in word_embedding.vocab.words:
            continue
        for check_word in list_of_check_words:
            sim = word_embedding.sim(check_word, w)
            if sim > max_sim_value:
                index_sim_word = i
                sim_word = w
                max_sim_value = sim
    return words[index_sim_word: index_sim_word + count]


def get_names(df: pd.DataFrame):
    df_with_names = df.drop(columns=["line_n", "role", "is_greet",
                                     "is_name", "is_goodbye", "is_company"])
    df_with_names.loc[:, "name"] = \
        df["text"].apply(get_word_from_phrase,
                         args=(["софья", "татьяна", "максим", "александр"],))
    return df_with_names


def get_company_names(df: pd.DataFrame):
    df_with_company_names = df.drop(columns=["line_n", "role", "is_greet",
                                             "is_name", "is_goodbye", "is_company"])
    df_with_company_names.loc[:, "name"] = \
        df["text"].apply(get_word_from_phrase, args=(["компания", "фирма"], 3))
    return df_with_company_names


def is_manager_polite(df: pd.DataFrame):
    """
    check does manager said greetings in start of dialog and goodbye at the end
    :param df:
    :return:
    """
    grouped = df.groupby("dlg_id", ).any().drop(columns=["line_n", "role", "text",
                                                         "is_name", "is_company"])
    grouped.loc[:, "is_polite"] = grouped[["is_greet", "is_goodbye"]].apply(all, axis=1)
    return grouped


if __name__ == "__main__":
    pd.set_option("display.max_columns", 10)
    pd.set_option('display.max_colwidth', 35)
    path = Path.cwd().parent.parent
    word_embedding = Navec.load(path / r'Saved_models/russian_word2vec/'
                                       r'navec_hudlit_v1_12B_500K_300d_100q.tar')

    input_df = pd.read_csv(path / "Datasets/test_task_BEWISE/test_data.csv")

    manager_df = input_df[input_df["role"] == "manager"]

    # Извлекать реплики с приветствием – где менеджер поздоровался.
    input_df.loc[:, "is_greet"] = get_phrase(manager_df[manager_df["line_n"] < 4],
                                             ["здравствуйте", "привет", "приветствую"], 0.55)

    # Извлекать реплики, где менеджер представил себя.
    input_df.loc[:, "is_name"] = get_phrase(manager_df[manager_df["line_n"] < 4],
                                            ["софья", "татьяна", "максим", "александр"], 0.6)

    # Извлекать реплики, где менеджер попрощался.
    index_of_last_phrases = manager_df["dlg_id"].drop_duplicates(keep="last").index
    last_phrases = input_df.iloc[index_of_last_phrases, :].copy()
    input_df.loc[:, "is_goodbye"] = get_phrase(last_phrases, ["свидания", "доброго"], 0.43)

    input_df.loc[:, "is_company"] = get_phrase(manager_df, ["компания", "фирма"], 0.48)
    
    input_df[["is_greet", "is_name", "is_goodbye", "is_company"]] = \
        input_df[["is_greet", "is_name", "is_goodbye", "is_company"]].fillna(False)

    # Извлекать имя менеджера.
    names = get_names(input_df[input_df["is_name"]])

    # Извлекать название компании.
    company_names = get_company_names(input_df[input_df["is_company"]])

    # Проверять требование к менеджеру:
    #   «В каждом диалоге обязательно необходимо поздороваться и попрощаться с клиентом»
    polite_dialog = is_manager_polite(input_df)
