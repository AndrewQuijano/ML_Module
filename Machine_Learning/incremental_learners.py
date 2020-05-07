import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.linear_model import PassiveAggressiveRegressor, SGDRegressor
from joblib import dump
from sklearn.model_selection import KFold

from .misc import plot_grid_search


def incremental_clf_list(train_x, train_y, speed=True):
    kf = KFold(n_splits=10)
    percep = tune_perceptron(train_x, train_y, kf, speed)
    sgd_class = tune_sgd_clf(train_x, train_y, kf, speed)
    pa_classifier = tune_passive_aggressive_clf(train_x, train_y, kf, speed)
    # sgd_regress = tune_passive_aggressive_reg(train_x, train_y, kf, speed)
    # pa_regress = tune_passive_aggressive_reg(train_x, train_y, kf, speed)

    # Get Parameters now
    with open("results.txt", "a+") as fd:
        fd.write("[percep] Best Parameters: " + str(percep.best_params_) + '\n')
        fd.write("[sgd_class] Best Parameters: " + str(sgd_class.best_params_) + '\n')
        fd.write("[pa_classifier] Best Parameters: " + str(pa_classifier.best_params_) + '\n')
        # fd.write("[sgd_regress] Best Parameters: " + str(sgd_regress.best_params_) + '\n')
        # fd.write("[pa_regress] Best Parameters: " + str(pa_regress.best_params_) + '\n')

        fd.write("[percep] Training Score: " + str(percep.score(train_x, train_y)) + '\n')
        fd.write("[sgd_class] Training Score: " + str(sgd_class.score(train_x, train_y)) + '\n')
        fd.write("[pa_classifier] Training Score: " + str(pa_classifier.score(train_x, train_y)) + '\n')
        # fd.write("[sgd_regress] Training Score: " + str(sgd_regress.score(train_x, train_y)) + '\n')
        # fd.write("[pa_regress] Training Score: " + str(pa_regress.score(train_x, train_y)) + '\n')

    clf_list = [percep, sgd_class, pa_classifier ]
    # Be prepared....Sometimes it does fail!
    try:
        true_model = tune_bayes(train_x, train_y, kf, speed)
        with open("results.txt", "a+") as fd:
            fd.write("[bayes] Best Parameters: " + str(true_model.best_params_) + '\n')
            fd.write("[bayes] Training Score: " + str(true_model.score(train_x, train_y)) + '\n')
        clf_list.append(true_model)
    except ValueError:
        pass


# -------------------------Tune Classifier-----------------------------------------------------------
def tune_bayes(x, y, n_folds=10, slow=True):
    print("Tuning Multinomial Bayes...")
    c = np.arange(0.01, 1.3, 0.01)
    param_grid = {'alpha': c}
    model = MultinomialNB()
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'Multinomial_Bayes')
    print("Finished Tuning Multinomial Bayes...")
    dump(true_model, "./Classifiers/" + type(model).__name__ + ".joblib")
    return true_model


def tune_perceptron(x, y, n_folds=10, slow=True):
    print("Tuning Perceptron...")
    c = np.arange(0.01, 1.3, 0.01)
    param_grid = {'alpha': c}
    model = Perceptron(tol=1e-3, warm_start=True)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, "Perceptron")
    print("Finished Tuning Perceptron...")
    dump(true_model, "./Classifiers/" + type(model).__name__ + ".joblib")
    return true_model


def tune_sgd_clf(x, y, n_folds=10, slow=True):
    print("Tuning SGD Classifier...")
    c = np.arange(0.0001, 0.01, 0.00001)
    param_grid = {'alpha': c}
    model = SGDClassifier(warm_start=True, tol=1e-3)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'SGD_Classifier')
    print("Finished Tuning SGD Classifier...")
    dump(true_model, "./Classifiers/" + type(model).__name__ + ".joblib")
    return true_model


def tune_sgd_reg(x, y, n_folds=10, slow=True):
    print("Tuning SGD Regressor...")
    c = np.arange(0.0001, 0.01, 0.00001)
    param_grid = {'alpha': c}
    model = SGDRegressor(warm_start=True, tol=1e-3)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'SGD_Regression')
    print("Finished Tuning SGD Regressor...")
    dump(true_model, "./Classifiers/" + type(model).__name__ + ".joblib")
    return true_model


def tune_passive_aggressive_clf(x, y, n_folds=10, slow=True):
    print("Tuning Passive Aggressive Classifier...")
    c = np.arange(0.01, 1.6, 0.01)
    param_grid = {'C': c}
    model = PassiveAggressiveClassifier(warm_start=True, tol=1e-3)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'Passive_Aggressive_CLF')
    print("Finished Tuning Passive Aggressive Classifier...")
    dump(true_model, "./Classifiers/" + type(model).__name__ + ".joblib")
    return true_model


def tune_passive_aggressive_reg(x, y, n_folds=10, slow=True):
    print("Tuning Passive Aggressive Regression Classifier...")
    c = np.arange(0.01, 1.6, 0.01)
    param_grid = {'C': c}
    model = PassiveAggressiveRegressor(warm_start=True, tol=1e-3)
    if slow:
        true_model = GridSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    else:
        true_model = RandomizedSearchCV(model, param_grid, cv=n_folds, n_jobs=-1, error_score='raise', verbose=2)
    true_model.fit(x, y)
    if slow:
        plot_grid_search(true_model.cv_results_, c, 'Passive_Aggressive_Regression')
    print("Finished Tuning Passive Aggressive Regression...")
    dump(true_model, "./Classifiers/" + type(model).__name__ + ".joblib")
    return true_model
