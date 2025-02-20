Binoculars - это метод обнаружения текста, сгенерированного искусственным интеллектом. Метод работает в режиме zero-shot (не требует обучающих данных) и основан на том, что большинство языковых моделей имеют значительное пересечение в наборах данных для предварительного обучения.

## Что пытаемся сделать

- Улучшить метрики детекции по сравнению с базовой версией при помощи нескольких пар моделей
- Адаптировать для работы с русскоязычными текстами

## Что уже сделали
- Русскоязычный датасет для комплексного тестирования
- Реализованы скрипты для оценки эффективности различных пар моделей
- Проведено масштабное тестирование, подтверждающее надежность метода

## Установка и использование

```bash
$ git clone https://github.com/CoffeBank/Trinoculars.git
$ cd Trinoculars
$ pip install -e .
```

### Пример использования

```python
from binoculars import Binoculars

bino = Binoculars()
text = "Ваш текст для проверки"
result = bino.predict(text)
print(f"Результат проверки: {result}")
```

## Ограничения

Детектор предназначен для академических целей и требует человеческого контроля при использовании. Эффективность может варьироваться в зависимости от языка и контекста.

## Цитирование

```bibtex
@misc{hans2024spotting,
      title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text}, 
      author={Abhimanyu Hans and Avi Schwarzschild and Valeriia Cherepanova and Hamid Kazemi and Aniruddha Saha and Micah Goldblum and Jonas Geiping and Tom Goldstein},
      year={2024},
      eprint={2401.12070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
