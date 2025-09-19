# LLM классификация отзывов

## Просмотр работы

Код данного проекта можно посмотреть по ссылке - https://colab.research.google.com/drive/1aPXUGpzWU6N6AcdkTUtGb0E1FqVD_aeV?usp=share_link

## Краткое описание

Ноутбук для обучения и оценки LLM (в основном семейство RuBERT / другие эксперименты) на задаче классификации отзывов маркетплейсов по категориям товаров (+ класс "нет товара"). Цель: максимизировать Weighted F1 и обеспечить среднее время инференса на пример < 5 с.

---

## Содержание README (что есть в ноутбуке)

1. Подготовка окружения и установка зависимостей
2. Загрузка и предварительная обработка данных
3. Авторазметка с помощью LLM (`Qwen/Qwen2-7B-Instruct`)
4. Аугментация синтетики на основе выданных данных (+чистка)
5. Бейзлайн + обучение более крупных моделей
6. Оценка качества (accuracy / precision / recall / weighted F1 / macro F1)
7. Инференс на тесте + замер времени на пример

---

## Предобработка данных

* Функция `clean_text(t: str)` удаляет URL, HTML-теги, e-mail, телефонные номера, эмодзи, лишние пробелы и приводит текст к удобному виду. Регулярные выражения определены в ноутбуке (переменные `URL_RE`, `HTML_RE`, `EMAIL_RE`, `PHONE_RE`).
* Пропуски удаляются: `df_test = df_test[["text_clean"]].dropna().reset_index(drop=True)`.

---

## Авторазметка с помощью LLM

В ноутбуке реализован эксперимент с использованием **Qwen/Qwen2-7B-Instruct** для автоматической разметки данных.

* Функция `run_llm` отправляет партию текстов в модель Qwen и возвращает предсказанные категории.
* Функция `aggregate_llm_votes` агрегирует несколько предсказаний (например, при варьировании промптов или temperature) и выбирает финальный класс на основе голосования.
* Такой подход помогает создать псевдоразметку (pseudo-labels), которую можно использовать для дообучения основной классификационной модели.

---

## Аугментация синтетических данных

В ноутбуке реализован модуль для генерации и фильтрации синтетических примеров на основе выданных категорий. Он помогает увеличить число примеров в редких классах и сбалансировать датасет. 

* Функция `keyword_label` — назначает метку по ключевым словам, связанным с каждой категорией.
* Функция `passes_keyword_check` — проверяет, соответствует ли сгенерированный текст ключевым признакам категории.
* Функция `passes_llm_check` — дополнительная валидация LLM для отбраковки нерелевантных примеров.
* Функция `generate_and_filter` — объединяет генерацию кандидатов и их фильтрацию, оставляя только те синтетические тексты, которые удовлетворяют обоим условиям (ключевые слова + проверка LLM).

---

## Модели

В ноутбуке присутствуют несколько экспериментов:

* В роли бейзлайна использовались модели `LogisticRegression` и `LinearSVC`
* `cointegrated/rubert-tiny2` - для промежуточного тестирования
* `DeepPavlov/rubert-base-cased` — модель для финального инференса и/или обучения.

---

## Учёт дисбаланса классов

1) CrossEntropyLoss with compute_class_weight
   
* Вычисляются веса классов через `compute_class_weight(class_weight='balanced', classes=np.arange(NUM_LABELS), y=train_df['label'].values)`.
* Веса передаются в кастомный `CrossEntropyLoss` в функции `custom_loss`, и этот loss присваивается `trainer.compute_loss = custom_loss`.


2) Focal Loss
   
* Этот loss также интегрируется в Trainer через переопределение trainer.compute_loss = focal_loss.
  
---

## Параметры обучения (как в ноутбуке)

`TrainingArguments` использует следующие значения:

```python
training_args = TrainingArguments(
    output_dir="./rubert_results",
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=7,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    load_best_model_at_end=False,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=True,
    push_to_hub=False,
    report_to="none",
    seed=42
)
```

---

## Оценка (метрики)

* Лучшая метрика **Weighted F1** получилась **= 0.74** на валидации 7 эпохи (Модель: `DeepPavlov/rubert-base-cased`; CrossEntropyLoss with compute_class_weight)
* Итоговая метрика — **Weighted F1**
* Также выводятся `accuracy`, `f1_macro`, `precision`, `recall`.
* Также после каждой тренировки добавлены **Confusion matrix**

---

## Инференс и замер скорости

* Инференс на тестовом датасете реализован через `Trainer.predict(test_dataset)`.
* Замер времени на пример делается так:

```python
import time
import numpy as np
import torch
start = time.perf_counter()
predictions = trainer.predict(test_dataset)
end = time.perf_counter()
total_time = end - start
num_samples = len(test_dataset)
avg_time_per_sample = total_time / num_samples
print(f"Всего времени: {total_time:.2f} сек")
print(f"Среднее на 1 пример: {avg_time_per_sample:.4f} сек")
```

---

## Сохранение результатов

* Лучшие модели сохраняются в папку `./rubert_results/best_model/...` (в ноутбуке показаны разные подпапки для разных loss'ов, например `cross_entropy` и `focal_loss`)
