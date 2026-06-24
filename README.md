# Market Risk Project 2

Финальная версия проекта для защиты по оценке рыночного риска портфеля на дату `2025-12-02`.

Портфель:

- 5 выпусков ОФЗ: по 10 млн руб. в каждый;
- 10 российских акций: по 1 млн руб. в каждую;
- валютная позиция: 100 млн руб. в USD и 100 млн руб. в EUR.

Метрики:

- VaR 99%;
- Expected Shortfall 97.5%;
- горизонты 1 и 10 торговых дней;
- backtesting 1-day VaR 99% за 2025 год.

## Как запустить

Из корня проекта:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_pipeline.py
```

Pipeline одной командой:

- читает `df_final.csv` и `ofz_final.csv`;
- очищает данные в `data/processed/`;
- строит риск-факторы и PCA;
- считает Monte Carlo VaR/ES;
- выполняет backtesting;
- пересоздает таблицы, графики и outputs.

Минимальные зависимости указаны в `requirements.txt`: `pandas`, `numpy`. Графики создаются как SVG без `matplotlib`, чтобы проект запускался в минимальном окружении. Если `python3 scripts/run_pipeline.py` падает с `ModuleNotFoundError: No module named 'pandas'`, нужно сначала выполнить установку зависимостей командой выше или запускать проект в окружении, где уже есть `pandas` и `numpy`.

## Структура проекта

```text
market-risk-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── notebooks/
├── src/
│   ├── data.py
│   ├── risk_factors.py
│   ├── models.py
│   ├── valuation.py
│   ├── simulation.py
│   ├── risk_metrics.py
│   ├── backtesting.py
│   └── utils.py
├── scripts/
│   └── run_pipeline.py
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── presentation_market_risk.md
│   ├── qa_for_defense.md
│   ├── slide_outline.md
│   └── speaker_notes.md
├── df_final.csv
├── ofz_final.csv
├── requirements.txt
└── README.md
```

## Входные данные

`df_final.csv` содержит 1 309 строк за период `2021-01-01` - `2025-12-31`:

- процентная кривая по тенорам от `0 years` до `30 years`;
- цены пяти ОФЗ;
- цены 10 акций;
- индексы IMOEX/RTSI;
- Brent;
- USD/RUB и EUR/RUB.

`ofz_final.csv` содержит 136 строк денежных потоков ОФЗ: даты выплат, купоны и погашения.

## Методология

Риск-факторы:

- акции: лог-доходности 10 акций;
- валюта: лог-доходности USD/RUB и EUR/RUB;
- ставки: дневные изменения процентной кривой;
- облигации: переоценка через DCF по сдвинутой процентной кривой;
- IMOEX, RTSI и Brent: используются в описательной статистике и корреляциях как рыночные индикаторы.

PCA:

- PCA строится по дневным изменениям ставок;
- используются первые 3 компоненты;
- 3 компоненты объясняют около 91.36% дисперсии изменений кривой.

Модель динамики:

- совместная многомерная нормальная модель;
- параметры оцениваются через исторические mean/cov;
- для горизонта `h`: `mean * h`, `cov * h`;
- random seed зафиксирован.

Оценка инструментов:

- акции: количество акций × цена;
- валюта: валютная сумма × курс;
- ОФЗ: DCF будущих cash flows по интерполированной процентной кривой.

Backtesting:

- для каждого дня 2025 года рассчитывается 1-day VaR 99%;
- фактический P&L берется на следующий торговый день;
- пробой: `actual P&L < -VaR`;
- считаются тесты Kupiec unconditional coverage и Christoffersen conditional coverage.

## Основные результаты

Дата риска: `2025-12-02`.

Позиции формируются по предыдущей торговой дате: `2025-12-01`.

| Горизонт | Компонент | VaR 99%, руб. | ES 97.5%, руб. |
| ---: | --- | ---: | ---: |
| 1 день | акции | 487 972 | 488 969 |
| 1 день | облигации | 659 585 | 665 077 |
| 1 день | валюта | 25 785 978 | 25 616 016 |
| 1 день | весь портфель | 25 708 258 | 25 594 324 |
| 10 дней | акции | 1 396 885 | 1 396 688 |
| 10 дней | облигации | 2 076 295 | 2 113 488 |
| 10 дней | валюта | 63 601 598 | 64 556 347 |
| 10 дней | весь портфель | 63 932 855 | 64 574 267 |

Главный вывод: риск портфеля в основном валютный. Это естественно, потому что валютный блок равен 200 млн руб., тогда как акции - 10 млн руб., а облигации - 50 млн руб.

## Backtesting 2025

| Компонент | Наблюдений | Пробои | Доля пробоев | Kupiec p-value | Christoffersen CC p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| акции | 260 | 1 | 0.38% | 0.2544 | 0.5203 |
| облигации | 260 | 1 | 0.38% | 0.2544 | 0.5203 |
| валюта | 260 | 0 | 0.00% | 0.0222 | 0.0733 |
| весь портфель | 260 | 0 | 0.00% | 0.0222 | 0.0733 |

Интерпретация:

- для акций и облигаций Kupiec не отвергает корректность частоты пробоев;
- по валюте и всему портфелю пробоев нет, поэтому модель выглядит консервативной на 2025 годе;
- ноль пробоев при 1% VaR на 260 наблюдениях возможен, но статистически выглядит слишком осторожно.

## Основные выходные файлы

Расчеты:

- `data/outputs/risk_metrics_var_es.csv`
- `data/outputs/backtest_2025_summary.csv`
- `data/outputs/backtest_2025_details.csv`
- `data/outputs/mc_pnl_h1.csv`
- `data/outputs/mc_pnl_h10.csv`

Таблицы для защиты:

- `reports/tables/portfolio_composition.csv`
- `reports/tables/risk_factor_inventory.csv`
- `reports/tables/risk_factor_descriptive_stats.csv`
- `reports/tables/pca_explained_variance.csv`
- `reports/tables/var_es_summary.csv`
- `reports/tables/backtesting_summary.csv`
- `reports/tables/kupiec_test.csv`
- `reports/tables/christoffersen_test.csv`
- `reports/tables/bond_pricing_errors.csv`
- `reports/tables/model_limitations.csv`

Графики для презентации:

- `reports/figures/risk_factor_dynamics_prices_fx.svg`
- `reports/figures/risk_factor_dynamics_rates.svg`
- `reports/figures/risk_factor_correlation.svg`
- `reports/figures/pca_explained_variance.svg`
- `reports/figures/pca_components.svg`
- `reports/figures/pnl_distribution_total_h1.svg`
- `reports/figures/pnl_distribution_total_h10.svg`
- `reports/figures/backtest_total_breaches.svg`
- `reports/figures/bond_pricing_errors.svg`

Материалы защиты:

- `reports/presentation_market_risk.md`
- `reports/speaker_notes.md`
- `reports/qa_for_defense.md`
- `reports/slide_outline.md`
- `reports/audit_notes.md`

## Ограничения модели

Модель не идеальная, и это важно проговорить на защите:

- нормальное распределение не описывает тяжелые хвосты и резкие скачки;
- Student-t/EVT не включены в baseline, хотя являются естественным улучшением;
- DCF для ОФЗ приближенный: не учитываются НКД, bid/ask, ликвидность, clean/dirty price conventions;
- в модели ОФЗ основным фактором риска является процентная кривая, но не отдельный спред каждого выпуска;
- backtesting за один год дает всего 260 наблюдений, а для 1% VaR ожидается около 2-3 пробоев;
- `statsmodels` недоступен в текущем runtime, поэтому ADF-таблица создается с пометкой, а не с полноценными p-values.

## Что смотреть перед защитой

1. `reports/presentation_market_risk.md` - готовая структура презентации.
2. `reports/speaker_notes.md` - текст выступления на 15-18 минут.
3. `reports/qa_for_defense.md` - возможные вопросы и ответы.
4. `reports/tables/var_es_summary.md` - ключевые VaR/ES.
5. `reports/tables/backtesting_summary.md` - итоги backtesting.
6. `reports/figures/backtest_total_breaches.svg` - главный график проверки VaR.
7. `reports/audit_notes.md` - краткий аудит реализации.

