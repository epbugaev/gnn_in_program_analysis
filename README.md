# Graph Neural Networks in Program Analysis
Курсовая работа студента БПМИ 213 ФКН НИУ ВШЭ Бугаева Егора: "Использованию графовых нейросетей для программного анализа".
Выполнена под руководством Шайхелисламова Данила Салаватовича, старшего преподавателя ФКН НИУ ВШЭ

## Описание проекта
В этой работе исследуется применения GNN для обнаружения уязвимостей в сгенерированном коде. В частности, изучается модель Devign и сравнивается с традиционным статическим анализатором Svace. По результатам работы GNN позволяют находить уязвимости в сгенерированном коде,
которые статический анализатор найти может не всегда. Svace обнаружил уязвимости в 12.5% программ на языке Java среди тех, которые компилируются, в то время как Devign смог найти уязвимости среди 16% программ, написанных на языке C. Кроме того, Devign
достиг Accuracy в 92% на валидационной выборке в датасете.

## Устройство кода
В текущем режиме main.py работает как анализатор генерируемого кода с помощью Devign, для переключение на анализ с помощью Svace или других программ необходимо расскоментировать соответствующие строки в main.py

## Требования для запуска
Необходимо создать папку devign в папке с данным репозиторием и клонировать туда содержимое GitHub модели https://github.com/epicosy/devign. 

## Используемые источники
Модель Devign - https://github.com/epicosy/devign
Датасет для обучения - https://dl.acm.org/doi/10.1145/3379597.3387501
Статический анализатор Svace, используемый как бейзлайн - https://www.ispras.ru/technologies/svace/