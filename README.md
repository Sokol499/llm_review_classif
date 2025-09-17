# LLM классификация отзывов

## Краткое описание

Проект — ноутбук для обучения и оценки LLM (в основном семейство RuBERT / другие эксперименты) на задаче классификации отзывов маркетплейсов по категориям товаров (+ класс "нет товара"). Цель: максимизировать Weighted F1 и обеспечить среднее время инференса на пример < 5 с.

---

## Содержание README (что есть в ноутбуке)

1. Подготовка окружения и установка зависимостей
2. Загрузка и предварительная обработка данных (clean\_text)
3. Разметка / кодировка меток (LabelEncoder)
4. Учёт дисбаланса классов — compute\_class\_weight + кастомный loss
5. Модель и обучение (Trainer, TrainingArguments)
6. Оценка качества (precision / recall / weighted F1)
7. Инференс на тесте + замер времени на пример
8. Сохранение лучших чекпоинтов
9. Рекомендации и идеи для улучшения

---

## Быстрый старт (локально / colab)

1. Склонируйте или откройте ноутбук `nlp_case_tbank_sirius_final.ipynb`.
2. Установите зависимости (в ячейке ноутбука уже есть):

```bash
!pip install -q bitsandbytes
!pip install -q accelerate transformers peft
# дополнительно (если нужно):
pip install datasets evaluate scikit-learn pandas torch emoji nbformat
```

3. Положите `train.csv` и `test.csv` в рабочую директорию (в ноутбуке читаются как `/content/train.csv` и `/content/test.csv`). Формат:

   * `train.csv`: столбцы `text`, `label`.
   * `test.csv`: столбец `text`.

4. Запустите ячейки последовательно: это выполнит очистку текста, подготовит датасеты, обучит модель и выполнит предсказание на тесте.

---

## Предобработка данных

* Функция `clean_text(t: str)` удаляет URL, HTML-теги, e-mail, телефонные номера, эмодзи, лишние пробелы и приводит текст к удобному виду. Регулярные выражения определены в ноутбуке (переменные `URL_RE`, `HTML_RE`, `EMAIL_RE`, `PHONE_RE`).
* Пропуски удаляются: `df_test = df_test[["text_clean"]].dropna().reset_index(drop=True)`.

---

## Как размечаются и кодируются метки

* Используется `LabelEncoder` из scikit-learn:

  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  train_df['label'] = le.fit_transform(train_df['label'])
  val_df['label'] = le.transform(val_df['label'])
  NUM_LABELS = len(le.classes_)
  ```
* `NUM_LABELS` вычисляется автоматически.

---

## Учёт дисбаланса классов

* Вычисляются веса классов через `compute_class_weight(class_weight='balanced', classes=np.arange(NUM_LABELS), y=train_df['label'].values)`.
* Веса передаются в кастомный `CrossEntropyLoss` в функции `custom_loss`, и этот loss присваивается `trainer.compute_loss = custom_loss`.

---

## Модель(и)

В ноутбуке присутствуют несколько экспериментов:

* `cointegrated/rubert-tiny2` (токенизатор и модель для SequenceClassification в одном эксперименте).
* `DeepPavlov/rubert-base-cased` — модель для финального инференса и/или обучения.
* Также есть ячейки с загрузкой `Qwen/Qwen2-7B-Instruct` (эксперименты с causal LM / quantization через bitsandbytes). Эти эксперименты оставлены в ноутбуке как альтернативы/эксперименты.

Вы можете менять модель, присвоив новую строку переменной `MODEL_NAME` (находится в ноутбуке).

---

## Параметры обучения (как в ноутбуке)

Отрывок `TrainingArguments` использует следующие значения (настройки в ноутбуке):

```python
training_args = TrainingArguments(
    output_dir="./rubert_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    push_to_hub=False,
    report_to="none"
)
```

* Trainer используется в стандартном виде: `Trainer(model=..., args=training_args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, compute_metrics=compute_metrics)`
* `compute_metrics` возвращает `f1_weighted`, `f1_macro`, `accuracy`, precision и recall.

---

## Оценка (метрики)

* Итоговая метрика — **Weighted F1** (в коде `f1_weighted = f1_score(labels, predictions, average='weighted')`).
* Также выводятся `accuracy`, `f1_macro`, `precision`, `recall`.

---

## Инференс и замер скорости

* Инференс на тестовом датасете реализован через `Trainer.predict(test_dataset)`.
* Замер времени на пример делается так:

```python
import time
start = time.perf_counter()
predictions = trainer.predict(test_dataset)
end = time.perf_counter()
total_time = end - start
num_samples = len(test_dataset)
avg_time_per_sample = total_time / num_samples
print(f"Всего времени: {total_time:.2f} сек")
print(f"Среднее на 1 пример: {avg_time_per_sample:.4f} сек")
```

В ноутбуке есть пример, который выводит значения `Всего времени` и `Среднее на 1 пример`.

---

## Сохранение результатов

* Лучшие модели сохраняются в папку `./rubert_results/best_model/...` (в ноутбуке показаны разные подпапки для разных loss'ов, например `cross_entropy` и `focal_loss`).
* Предсказания теста сохраняются в `df_test['label']` (и вы можете сохранить `df_test.to_csv('test_pred.csv', index=False)`).

---

## Где смотреть и что поправить (рекомендации)

1. **Упрощение / скриптизация**: вынести ключевые этапы (preprocess, train, eval, infer) в отдельные `.py`-скрипты для удобного запуска вне ноутбука.
2. **PEFT / LoRA**: ноутбук содержит установки для `peft` и `bitsandbytes`, но основной учебный цикл использует стандартный `Trainer` на Rubert. Для больших LLM (Qwen) рекомендую использовать LoRA/PEFT + 4-bit quantization (bitsandbytes) для экономии памяти и ускорения итераций.
3. **Data augmentation / pseudo-labeling**: для редких классов можно попробовать синтетическое увеличение или self-training.
4. **Кросс-валидация / stratified split**: для стабильной оценки weighted F1.
5. **Оптимизация inference**: пакетирование, TorchScript/ONNX и уменьшение `max_length`/batch size для ускорения. Проверить, укладывается ли среднее время на примере в 5 сек (но ноутбук уже считает среднее время).
6. **Logging & reproducibility**: зафиксировать seed, сохранить версию зависимостей, добавить small `requirements.txt`.

---

## Примерные команды (export / запуск)

* Установить зависимости (см. выше).
* Запустить ноутбук в Colab / локально.

Если хотите — я могу:

* Сгенерировать `requirements.txt` и `train.py`/`inference.py` на основе ноутбука.
* Сократить и очистить ноутбук, удалив экспериментальные ячейки и оставив чистую воспроизводимую версию.
* Подготовить Dockerfile для развёртывания модели и измерения latency.

---

## Контакты / заметки

README собран автоматически на основе содержимого `nlp_case_tbank_sirius_final.ipynb`. Если нужно, могу адаптировать README под одну конкретную конфигурацию (например только `DeepPavlov/rubert-base-cased`) и добавить точные команды для запуска (скрипты).

*Конец README*
