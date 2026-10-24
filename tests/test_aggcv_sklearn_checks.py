"""Базові перевірки scikit-learn (estimator checks) для підкласів пошуку."""

from sklearn.linear_model import Ridge
from sklearn.utils.estimator_checks import parametrize_with_checks

from aggcv import AggregatedGridSearchCV, AggregatedRandomizedSearchCV

GRID = {"alpha": [0.1, 1.0]}

# Єдина перевірка, що не проходить; так само вона не проходить для стандартного
# GridSearchCV(Ridge(), ...) у scikit-learn 1.8.0, тобто це не властивість aggcv.
EXPECTED_FAILED = {
    "check_supervised_y_2d": "не проходить і для стандартного GridSearchCV",
}


@parametrize_with_checks(
    [
        AggregatedGridSearchCV(Ridge(), GRID, cv=3),
        AggregatedGridSearchCV(Ridge(), GRID, cv=3, aggregation="rank"),
        AggregatedGridSearchCV(Ridge(), GRID, cv=3, aggregation="vote"),
        AggregatedRandomizedSearchCV(Ridge(), GRID, cv=3, n_iter=2,
                                     aggregation="median", random_state=0),
    ],
    expected_failed_checks=lambda est: EXPECTED_FAILED,
)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)
