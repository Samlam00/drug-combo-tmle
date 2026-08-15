# Super Learner implementation (AI-generated)
# _fingerprint method ensures that OOF predictions are returned if predict() is called on the training matrix

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

def _fingerprint(X):
    """Cheap, order-sensitive fingerprint of a training matrix so predict() can
    recognize when it is being called on the exact rows used to fit and return
    cross-validated (out-of-fold) predictions instead of in-sample ones."""
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    return (X.shape, hash(X.tobytes()))

class SuperLearnerRegression(BaseEstimator, RegressorMixin):
    def __init__(self, cv=5, random_state=1):
        self.cv = cv
        self.random_state = random_state

        self.base_learners = [
            ('lm', make_pipeline(StandardScaler(), LinearRegression())),
            ('ridge_1', make_pipeline(StandardScaler(), Ridge(alpha=0.1))),
            ('ridge_2', make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
            ('lasso_1', make_pipeline(StandardScaler(), Lasso(alpha=0.001, random_state=random_state))),
            ('en_1', make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=random_state))),
            ('en_2', make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=random_state))),
            ('rf_shallow', RandomForestRegressor(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1
            )),
            ('rf_deep', RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1
            )),
            ('gb_1', GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=random_state
            )),
            ('gb_2', GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=random_state
            )),
            ('svr_rbf', make_pipeline(
                StandardScaler(),
                SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
            )),
            ('svr_poly', make_pipeline(
                StandardScaler(),
                SVR(kernel='poly', degree=2, C=0.5, epsilon=0.1, gamma='scale')
            )),
            ('knn_5', make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=5, weights='distance')
            )),
            ('knn_15', make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=15, weights='distance')
            )),
        ]

    def _fit_meta_weights(self, Z, y):
        n_learners = Z.shape[1]

        def objective(w):
            pred = Z @ w
            return np.mean((y - pred) ** 2)

        bounds = [(0, 1)] * n_learners
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        w0 = np.ones(n_learners) / n_learners

        return minimize(objective, w0, bounds=bounds, constraints=constraints).x

    def fit(self, X, y):
        y = np.asarray(y)

        cv = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        oof_preds = []
        self.fitted_learners_ = []

        for name, learner in self.base_learners:
            model = clone(learner)

            preds = cross_val_predict(model, X, y, cv=cv, method='predict')
            oof_preds.append(preds)

            model.fit(X, y)
            self.fitted_learners_.append((name, model))

        Z = np.column_stack(oof_preds)
        self.weights_ = self._fit_meta_weights(Z, y)

        # store the out-of-fold ensemble prediction and a fingerprint of the
        # training matrix, so predict() on this same X returns OOF values.
        self.oof_ = Z @ self.weights_
        self._train_fp = _fingerprint(X)

        return self

    def predict(self, X):
        # if called on the exact training matrix, return cross-fitted predictions
        if getattr(self, "_train_fp", None) is not None and _fingerprint(X) == self._train_fp:
            return self.oof_

        base_preds = []
        for name, model in self.fitted_learners_:
            preds = model.predict(X)
            base_preds.append(preds)

        base_preds = np.column_stack(base_preds)
        return base_preds @ self.weights_

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def get_learner_weights(self):
        return {
            name: weight
            for (name, _), weight in zip(self.fitted_learners_, self.weights_)
        }

class SuperLearnerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cv=5, random_state=1):
        self.cv = cv
        self.random_state = random_state

        # Diverse classifier library
        self.base_learners = [
            ('logreg_l2', make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, l1_ratio=0, solver='lbfgs', max_iter=2000, random_state=random_state)
            )),
            ('logreg_l1', make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.5, l1_ratio=1, solver='liblinear', max_iter=2000, random_state=random_state)
            )),
            ('rf_shallow', RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1
            )),
            ('rf_deep', RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state
            )),
            ('svc_rbf', make_pipeline(
                StandardScaler(),
                SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=random_state)
            )),
            ('knn_5', make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=5, weights='distance')
            )),
            ('knn_15', make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=15, weights='distance')
            )),
            ('gnb', GaussianNB())
        ]

    def _fit_meta_weights_binary(self, Z, y):
        n_learners = Z.shape[1]

        def objective(w):
            p = np.clip(Z @ w, 1e-15, 1 - 1e-15)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        bounds = [(0, 1)] * n_learners
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        w0 = np.ones(n_learners) / n_learners

        return minimize(objective, w0, bounds=bounds, constraints=constraints).x

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # convert labels to {0,1} in class order
        y_bin = (y == self.classes_[1]).astype(int)

        cv = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state
        )

        oof_preds = []
        self.fitted_learners_ = []

        for name, learner in self.base_learners:
            model = clone(learner)

            # OOF probability for the positive class (classes_[1]); pass y_bin so the
            # column indexing is unambiguous regardless of the original label dtype.
            proba = cross_val_predict(model, X, y_bin, cv=cv, method='predict_proba')[:, 1]
            oof_preds.append(proba)

            model.fit(X, y)
            self.fitted_learners_.append((name, model))

        Z = np.column_stack(oof_preds)
        self.weights_ = self._fit_meta_weights_binary(Z, y_bin)

        # store OOF ensemble P(class1) and a fingerprint of the training matrix,
        # so predict_proba() on this same X returns OOF probabilities.
        self.oof_proba1_ = np.clip(Z @ self.weights_, 1e-15, 1 - 1e-15)
        self._train_fp = _fingerprint(X)

        return self

    def predict_proba(self, X):
        # if called on the exact training matrix, return cross-fitted probabilities
        if getattr(self, "_train_fp", None) is not None and _fingerprint(X) == self._train_fp:
            p1 = self.oof_proba1_
            return np.column_stack([1 - p1, p1])

        base_probas = []
        for name, model in self.fitted_learners_:
            proba = model.predict_proba(X)[:, 1]
            base_probas.append(proba)

        base_probas = np.column_stack(base_probas)
        p1 = np.clip(base_probas @ self.weights_, 1e-15, 1 - 1e-15)
        p0 = 1 - p1

        return np.column_stack([p0, p1])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return np.where(proba >= 0.5, self.classes_[1], self.classes_[0])

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def get_learner_weights(self):
        return {
            name: weight
            for (name, _), weight in zip(self.fitted_learners_, self.weights_)
        }
