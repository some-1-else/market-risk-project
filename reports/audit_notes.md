# Audit Notes

## Что проверено

- Pipeline запускается end-to-end и пересоздает processed data, outputs, tables и figures.
- Random seed зафиксирован в `scripts/run_pipeline.py`.
- Абсолютных путей внутри `src/` и `scripts/` нет; пути строятся через `pathlib` от корня проекта.
- Риск-факторы соответствуют портфелю: акции, FX, процентная кривая; индексы и Brent используются как дополнительные индикаторы.
- PCA построена по дневным изменениям процентной кривой, первые 3 компоненты объясняют 91.36% дисперсии.
- VaR считается как положительная величина потерь: `-quantile(P&L, 1 - confidence)`.
- ES 97.5% считается как средняя потеря в 2.5% левом хвосте P&L.
- Backtesting использует условие пробоя `actual_pnl < -VaR`, знак P&L не перепутан.
- Kupiec test реализован как likelihood-ratio unconditional coverage test.
- Christoffersen test реализован через transition counts и conditional coverage statistic.
- ОФЗ переоцениваются через DCF будущих cash flows по интерполированной процентной кривой.

## Найденные и исправленные зоны

- Добавлены финальные таблицы для защиты: portfolio composition, risk-factor inventory, VaR/ES summary, Kupiec, Christoffersen, limitations.
- Добавлены графики для презентации: динамика риск-факторов, backtesting с пробоями, ошибки DCF-оценки ОФЗ.
- Улучшены histogram labels для малых значений доходностей.
- README обновлен как финальный pre-defense документ.
- Созданы `presentation_market_risk.md`, `speaker_notes.md`, `qa_for_defense.md`.

## Остаточные ограничения

- Системный `python3` в текущей среде не содержит `pandas`; код успешно проверен в Python runtime, где установлены `pandas` и `numpy`.
- ADF test не выполняется в текущем runtime из-за отсутствия `statsmodels`; таблица сохраняет соответствующую пометку.
- Нормальная модель остается baseline и не описывает тяжелые хвосты идеально.
- Bond analytics приближенная: нет НКД, clean/dirty price conventions, bid/ask и индивидуальных spread factors.

