# bewise_test_task

## Использованные библиотеки 

[Navec](https://natasha.github.io/navec/) Для использования предобученых скрытых представлений русских слов

[pathlib](https://docs.python.org/3/library/pathlib.html) Для доступа к файлам

[pandas](https://pandas.pydata.org/) Для обработки табличных данных

[numpy](https://numpy.org/) Для работы с числами 

## Основной принцип работы- сравнение слов по их скрытым представлениям

Каждое скрытое представление слова, в произнесённой менеджером фразе, сравнивается со скрытым представлением искомых слов. 
К примеру если нужно найти где во фразе менеджер поздоровался мы сравниваем слова из фразы со словами "здравствуйте", "привет", "приветствую".
В коде за это отвечает функция *get_phrase()*
