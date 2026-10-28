# aggcv

`GridSearchCV` і `RandomizedSearchCV` з вибором правила агрегування результатів
внутрішніх розбиттів.

## Мотивація

Стандартний `GridSearchCV` обирає конфігурацію гіперпараметрів за середнім значенням
метрики на внутрішніх розбиттях. Проте це лише одне з можливих правил: за тією самою
матрицею оцінок (конфігурація × розбиття) можна обирати за одним розбиттям, медіаною,
середнім рангом, глобальним максимумом або голосуванням, і на малих вибірках ці правила
часто дають різні результати. `aggcv` реалізує шість правил (Single, Mean, Median, Rank,
Max, Vote) поверх стандартного механізму scikit-learn: моделі навчаються один раз, а
правило застосовується до `cv_results_`. Правила й розв'язання нічиїх відтворюють код
досліджень, для яких створено пакет (`run_experiments.py`; до пакета не входить).

## Встановлення

Вимоги: Python ≥ 3.13, scikit-learn ≥ 1.8, < 1.9, numpy, scipy, pandas.

Встановлення з кореня репозиторію:

```bash
pip install -e .
```

Збірка wheel і sdist (у каталог `dist/`) та встановлення з wheel:

```bash
pip install build
python -m build
pip install dist/aggcv-0.4.0-py3-none-any.whl
```

Пакет протестовано на Python 3.13 і 3.14 зі scikit-learn 1.8, numpy 2.4–2.5,
scipy 1.17–1.18 і pandas 3.1.

## Використання

### Варіант 1: `make_refit` (лише публічний API scikit-learn)

```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from aggcv import make_refit

search = GridSearchCV(Lasso(), {"alpha": [0.01, 0.1, 1.0]}, cv=10,
                      scoring="neg_root_mean_squared_error",
                      refit=make_refit("rank"))
search.fit(X, y)
search.best_params_      # за callable refit scikit-learn не заповнює best_score_
```

За кількох метрик (multimetric scoring) ім'я скорера передається другим аргументом:
`refit=make_refit("median", "mae")`.

### Варіант 2: підклас

```python
from aggcv import AggregatedGridSearchCV

search = AggregatedGridSearchCV(Lasso(), {"alpha": [0.01, 0.1, 1.0]}, cv=10,
                                scoring="neg_root_mean_squared_error",
                                aggregation="rank")
search.fit(X, y)
search.best_params_
search.best_score_          # агрегована оцінка обраної конфігурації (див. «Обмеження»)
search.split_scores_        # матриця (n_configs, n_splits)
search.aggregated_scores_   # агреговані оцінки всіх конфігурацій
```

`AggregatedRandomizedSearchCV` має ті самі додаткові параметри `aggregation` і
`refit_metric`. За кількох метрик вибір робиться за `refit_metric`, а якщо він
`None` — за метрикою, указаною в `refit` (напр. `refit="mae"`).

### Аналітичні утиліти

Обидві функції працюють із `cv_results_` уже навченого пошуку (стандартного
`GridSearchCV` / `RandomizedSearchCV` або підкласу з `aggcv`) і не перенавчають
внутрішні моделі.

```python
from aggcv import compare_strategies, selection_stability

compare_strategies(search)
#   strategy  best_index            params  aggregated_score
# 0   single          11  {'alpha': 5.62}            -24.05
# 1     mean           9  {'alpha': 1.78}            -24.45
# ...
```

`compare_strategies` повертає вибір кожного з шести правил у колонках `strategy`,
`best_index`, `params`, `aggregated_score`.

```python
selection_stability(search, n_permutations=50, n_values=(1, 2, 5, 10), random_state=2026)
```

`selection_stability` випадково переставляє розбиття (стовпці матриці) і для кожного N
обирає конфігурацію кожним правилом за першими N стовпцями. Результат містить по рядку
на кожну пару (правило, N) з колонками `full_index` (вибір правила на повному пулі),
`agreement` (частка перестановок, у яких вибір за N розбиттями збігся з ним) і
`n_distinct` (кількість різних обраних конфігурацій). За замовчуванням
N = 1, …, min(20, `n_splits`).

Обрані конфігурації можна також оцінити на зовнішніх даних. Тоді кожна *різна* обрана
конфігурація один раз перенавчається на `X`, `y` й оцінюється скорером пошуку, а до
результату додаються колонки `test_score_mean`, `test_score_std`, `full_test_score`:

```python
selection_stability(search, X=X_train, y=y_train, X_test=X_test, y_test=y_test)
```

За кількох метрик метрику задає аргумент `metric=` (за замовчуванням — `refit_metric`
або метрика з `refit`). Правило `single` використовує лише перший стовпець перестановки,
тому його рядки не залежать від N.

### Перемішані розбиття: `ReshuffledSearchCV`

У стандартному пошуку всі конфігурації оцінюються на спільних розбиттях. За Nagler et al.
(NeurIPS 2024), кожну конфігурацію можна оцінювати на власних розбиттях; найсильніший
ефект автори отримали для holdout (одного розбиття).

```python
from sklearn.model_selection import ShuffleSplit
from aggcv import ReshuffledSearchCV

# перемішаний holdout: одне власне розбиття на конфігурацію
search = ReshuffledSearchCV(Lasso(), {"alpha": [0.01, 0.1, 1.0]},
                            cv=ShuffleSplit(n_splits=1, test_size=0.25),
                            scoring="neg_root_mean_squared_error",
                            aggregation="single", random_state=2026)
search.fit(X, y)
search.split_seeds_     # зерно розбивача для кожної конфігурації
```

- `cv` — розбивач із параметром `random_state` (`ShuffleSplit`, `GroupShuffleSplit`,
  `StratifiedShuffleSplit`, `KFold(shuffle=True)`, `RepeatedKFold`); кількість розбиттів
  однакова для всіх конфігурацій. Власний `random_state` розбивача ігнорується.
- Зерна конфігурацій детерміновано виводяться з `random_state` пошуку, тож результат
  відтворюваний, а зерно конфігурації не залежить від розміру сітки.
- Доступні правила: `single`, `mean`, `median`, `max`. За одного розбиття всі чотири
  дають однаковий результат.
- `rank` і `vote` недоступні (`ValueError`): вони порівнюють конфігурації в межах одного
  розбиття, а тут стовпець `split{k}` у кожної конфігурації свій.
- Групи передаються звичайним способом: `fit(X, y, groups=groups)` або через metadata
  routing у вкладеній CV.
- Параметри: `estimator`, `param_grid`, `aggregation`, `scoring`, `cv`, `random_state`,
  `n_jobs`, `refit`, `error_score`, `verbose`, `return_train_score`. Атрибути після `fit`
  ті самі, що в `GridSearchCV` (`cv_results_` з тими самими ключами, `best_index_`,
  `best_params_`, `best_score_`, `best_estimator_`, `refit_time_`, `n_splits_`, `scorer_`,
  `multimetric_`), плюс `split_seeds_`, `split_scores_`, `aggregated_scores_`.
  `compare_strategies` і `selection_stability` працюють із ним без змін.

**Цикл оцінювання.** `ReshuffledSearchCV` не є підкласом `GridSearchCV`
(`BaseEstimator` + `MetaEstimatorMixin`). Для кожної конфігурації створюється копія `cv`
із власним зерном, розбиття обчислюються заздалегідь, а всі C × N пар (конфігурація,
розбиття) навчаються в одному `joblib.Parallel(n_jobs)`. Тому `n_jobs` розпаралелює
роботу й між конфігураціями, зокрема за одного розбиття (перемішаний holdout).
Результати не залежать від `n_jobs`.

**Невдалі навчання.** Пара, для якої `fit` завершився помилкою, отримує `error_score` у
`cv_results_` і `-inf` у `split_scores_` (незалежно від значення `error_score`), тому
конфігурацію з хоча б одним невдалим навчанням не може бути обрано. Повністю невдала
конфігурація **не** зупиняє пошук: `fit` видає одне попередження `FitFailedWarning` з
підсумком (кількість невдалих навчань, повністю невдалі конфігурації, повідомлення про
помилки). `ValueError` виникає, лише якщо невдалими є всі навчання всіх конфігурацій;
за `error_score="raise"` перший виняток передається далі. Невдале оцінювання (помилка
скорера) обробляється, як у scikit-learn: `UserWarning` і `error_score` для цієї пари.

**Metadata routing.** Сигнатура: `fit(X, y=None, *, groups=None, **fit_params)`. Без
routing групи передаються розбивачу, а всі інші параметри — у `fit` оцінювача (масиви
довжини `n_samples` розрізаються за індексами train), як у `GridSearchCV`. За
`enable_metadata_routing=True` маршрутизацію задає `get_metadata_routing` (оцінювач —
`fit`, розбивач — `split`, скорер — `score`): у вкладеній CV
`cross_validate(search, ..., params={"groups": g, "sample_weight": w})` групи доходять до
внутрішнього розбивача, а `sample_weight` — до оцінювача, якщо той його запитує
(`set_fit_request(sample_weight=True)`).

**Refit.** `refit=True` / `False` або, за кількох метрик, ім'я метрики вибору; фінальна
модель навчається на всіх `X`, `y`. За однієї метрики й `refit=False` конфігурація все
одно обирається (`best_index_`, `best_score_`), за кількох — ні (як у `GridSearchCV`).

**Імітація за наявною матрицею.** Якщо вже є матриця (конфігурація × розбиття), отримана
на спільному пулі розбиттів, перемішування можна імітувати без навчання моделей:

```python
from aggcv import reshuffle_scores, selection_stability

sub = reshuffle_scores(matrix, n_splits=10, random_state=2026)   # (n_configs, 10)
selection_stability(matrix, n_values=(1, 2, 5, 10), reshuffle=True)
```

`reshuffle_scores` дає кожній конфігурації власну випадкову підмножину з N стовпців;
`selection_stability(..., reshuffle=True)` робить те саме в кожній перестановці й рахує
`agreement` / `n_distinct` для `single`, `mean`, `median`, `max` (першим аргументом може
бути як навчений пошук, так і матриця). Це наближення: підмножини беруться з одного
пулу з M розбиттів, тому пара конфігурацій має в середньому N²/M спільних розбиттів
(1 за N = 10, M = 100), а за N = M усі конфігурації бачать увесь пул.

> **Застереження щодо `max`.** Якщо `n_configs × N ≥ M` (розмір пулу), імітація для
> правила `max` некоректна: разом конфігурації покривають увесь пул, тому `max` майже
> завжди знаходить найлегше розбиття пулу й обирає найкращу з конфігурацій, яким воно
> дісталося, — тобто поводиться як парний `max` на всьому пулі. За справжнього
> перемішування кожне розбиття дістається лише одній конфігурації. Через це імітація
> завищує стабільність `max` (на даних досліджень, elasticnet, N = 10: SD зовнішнього
> RMSE 0,0019 проти 0,0093 у парному режимі).

### Вкладена групова CV

Групи (напр. два ока одного пацієнта) мають потрапляти і в зовнішній, і у внутрішній
розбивач. Для цього потрібен metadata routing:

```python
import sklearn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), Lasso(max_iter=10_000))
inner_cv = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)

with sklearn.config_context(enable_metadata_routing=True):
    inner = AggregatedGridSearchCV(pipe, {"lasso__alpha": [0.01, 0.1, 1.0]},
                                   cv=inner_cv, aggregation="rank",
                                   scoring="neg_root_mean_squared_error")
    res = cross_validate(inner, X, y, cv=GroupKFold(5),
                         params={"groups": groups},
                         scoring="neg_root_mean_squared_error",
                         return_estimator=True)
```

Повний приклад на синтетичних даних — у `examples/demo.py`.

## Правила

Матриця оцінок: рядки — конфігурації (у порядку `cv_results_["params"]`), стовпці —
розбиття; більше значення = краще (конвенція скорерів scikit-learn).

| `aggregation` | У статті | У коді досліджень | Вибір | Нічия | `best_score_` |
|---|---|---|---|---|---|
| `single` | Single | `pipeline5_single_split` | найкраща оцінка на першому розбитті | перша конфігурація | оцінка на першому розбитті |
| `mean` | Mean | `pipeline6_mean_metric` | найкраще середнє за розбиттями | перша конфігурація | середнє |
| `median` | Median | `pipeline6_median_metric` | найкраща медіана за розбиттями | перша конфігурація | медіана |
| `rank` | Rank | `pipeline7_mean_rank` | найменший середній ранг; ранги в межах розбиття, 1 = найкраща, за рівних оцінок — середній ранг (`rankdata(method="average")`) | 1) менша дисперсія рангів (`ddof=0`); 2) краще середнє значення метрики; 3) перша конфігурація | **мінус середній ранг** |
| `max` | Max | `flow1_baseline` | конфігурація з глобальним максимумом матриці | переможець найранішого розбиття, на якому досягнуто максимуму; у межах розбиття — перша конфігурація | найкраща оцінка на одному розбитті |
| `vote` | Vote | `flow3_majority_vote` | найбільше перемог на окремих розбиттях (у межах розбиття за рівних оцінок перемагає перша конфігурація) | конфігурація, що першою стала переможцем у порядку розбиттів | **кількість голосів** |

`mean` дає ті самі `best_index_` і `best_score_`, що й стандартний `GridSearchCV`.

Невдалі навчання (NaN у `cv_results_`) перетворюються на `-inf`: конфігурація з хоча б
одним невдалим розбиттям не може перемогти за жодним правилом, а її агрегована оцінка
дорівнює `-inf`. Це узгоджується зі стандартним `GridSearchCV`, де середнє такої
конфігурації дорівнює NaN і вона отримує останній ранг. Без цього обмеження частково
невдала конфігурація могла б перемогти за `median`, `max` або `vote`.

## Обмеження

- **Приватний метод.** `AggregatedGridSearchCV` / `AggregatedRandomizedSearchCV`
  перевизначають приватний `BaseSearchCV._select_best_index`, звідси верхня межа версії
  scikit-learn (див. розділ «Залежність від приватного API scikit-learn»).
- **Значення `best_score_`.** Це агрегована оцінка за обраним правилом, а не обов'язково
  значення метрики: для `rank` — мінус середній ранг, для `vote` — кількість голосів.
  Не порівнюйте `best_score_` між правилами. Середнє значення метрики для обраної
  конфігурації: `search.cv_results_["mean_test_score"][search.best_index_]`.
- **`cv_results_` не змінюється.** `rank_test_score` завжди відповідає стандартному
  ранжуванню за середнім і може не збігатися з `best_index_`.
- **Metadata routing.** У вкладеній CV групи доходять до внутрішнього розбивача лише за
  `enable_metadata_routing=True` і `cross_validate(..., params={"groups": groups})`. Інакше
  груповий внутрішній розбивач завершиться помилкою, а негруповий мовчки розподілить
  спостереження однієї групи між train і validation.
- **Callable `refit` у підкласі.** Якщо передати в `AggregatedGridSearchCV` власний
  callable `refit`, вибір робить він: `aggregation` ігнорується, а `split_scores_`,
  `aggregated_scores_` і `best_score_` не заповнюються.
- **Кілька метрик без `refit`.** За кількох метрик і `refit=False` конфігурація
  обирається лише тоді, коли явно задано `refit_metric`; інакше, як і в стандартному
  `GridSearchCV`, атрибутів `best_index_`, `best_params_` і `best_score_` немає.
- **`ReshuffledSearchCV`: лише сітка.** Варіанта з випадковим пошуком
  (`param_distributions`) немає. Непарні `rank` / `vote` не підтримуються принципово:
  вони також втрачають блокування за розбиттям.
- **`ReshuffledSearchCV`: скорер.** `scoring` — рядок, callable, що повертає число, або
  list / tuple / dict для кількох метрик; callable, що повертає dict, і callable `refit` не
  підтримуються. `pre_dispatch` не налаштовується (використовується типове значення joblib).
- **`ReshuffledSearchCV`: невдалі навчання в кожній конфігурації.** Якщо в кожній
  конфігурації є невдалі навчання (але не всі навчання невдалі), обирається перша
  конфігурація в порядку сітки з агрегованою оцінкою `-inf` — так само, як у
  `GridSearchCV`, де в такому разі всі середні дорівнюють NaN. Про це сигналізує
  попередження `FitFailedWarning`.
- **`single` навчає моделі на всіх розбиттях.** Правило використовує лише перше
  розбиття, але пошук усе одно навчає моделі на всіх `n_splits`; щоб заощадити час,
  задайте `cv` з одним розбиттям.

## Залежність від приватного API scikit-learn

- **`AggregatedGridSearchCV` / `AggregatedRandomizedSearchCV`** перевизначають приватний
  `BaseSearchCV._select_best_index` — це єдине місце, де aggcv змінює поведінку
  scikit-learn через приватний метод. Саме тому залежність обмежено
  `scikit-learn>=1.8,<1.9`: метод не входить до публічного API й може змінитися без
  попередження. Верхню межу можна підняти, якщо на новій версії проходить
  `tests/test_aggcv_sklearn_private.py`.
- **`ReshuffledSearchCV`** не залежить від внутрішнього механізму `BaseSearchCV`.
  Він використовує лише три допоміжні приватні функції, кожну з яких покриває
  сторожовий тест у `tests/test_aggcv_sklearn_private.py`:
  `BaseEstimator._validate_params` (перевірка `_parameter_constraints`),
  `sklearn.utils.metaestimators._safe_split` (розрізання `X`, `y`, зокрема попередньо
  обчислених ядер) і `sklearn.utils.validation._check_method_params` (розрізання
  параметрів fit / score за індексами). Решта — публічний API: `clone`,
  `check_scoring` / `get_scorer`, `indexable`, `ParameterGrid`,
  `MetadataRouter` / `MethodMapping` / `process_routing`, `available_if`, `get_tags`,
  `sklearn.utils.parallel.Parallel` / `delayed` (обгортки joblib, що передають
  конфігурацію scikit-learn, зокрема `enable_metadata_routing`, у робочі процеси).
- **`make_refit`, `aggregate`, `compare_strategies`, `selection_stability`,
  `reshuffle_scores`** приватного API не використовують.

## Відмінності від коду досліджень

- **Повна нічия в `rank`.** Якщо у двох конфігурацій збігаються і середній ранг, і
  дисперсія рангів, і середнє значення метрики, `aggcv` обирає першу в порядку сітки
  (`cv_results_["params"]`), а код досліджень — першу за алфавітним порядком рядка
  `config_key` (`run_experiments.py`, `Pipeline7_MeanRankStrategy`). У `cv_results_`
  такого ключа немає, тому відтворити це правило неможливо. На даних досліджень ця
  відмінність жодного разу не вплинула на вибір.
- **Невдалі навчання.** Код досліджень пропускає конфігурацію лише на тому розбитті, де
  навчання не вдалося; `aggcv` повністю вилучає таку конфігурацію з вибору (див. вище).

**Відповідність результатам досліджень.** Вибір `aggcv` звірено зі збереженими
результатами досліджень для моделей lasso, elasticnet і rf: 5 зовнішніх фолдів,
3 метрики, 28 значень N від 1 до 100 і 6 правил — збігаються всі 7560 виборів. Для
xgboost звірку не виконано.

## Тести

```bash
python -m pytest -q
```

Регресійний тест на відповідність результатам досліджень зберігається в репозиторії
досліджень, оскільки потребує їхніх даних.
