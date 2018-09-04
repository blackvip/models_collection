import numpy as np
from functools import reduce
from sklearn.model_selection import KFold

class qiaofeng_kfold_stack:
    def __init__(self, train, train_target, test, model, score_func=None, kfolds=5, random_seed=9527, logger=None):
        self.train = train
        self.train_target = train_target
        self.test = test
        self.model = model
        self.score_func = score_func
        self.kfolds = kfolds
        self.random_seed = random_seed
        self.logger= logger
        self.skf = KFold(n_splits=self.kfolds, random_state=self.random_seed, shuffle=True)
        self.predict_test_kfolds = []
        self.predict_valid_kfolds = np.zeros((self.train.shape[0]))
    def model_fit(self, train, train_target):
        self.model.fit(train, train_target)
    def model_predict(self, dataset):
        return self.model.predict(dataset)
    def model_fit_predict(self, train, train_target, valid, valid_target):
        self.model_fit(train, train_target)
        predict_valid = self.model_predict(valid)
        predict_test = self.model_predict(self.test)
        return predict_valid, predict_test
    def clear_predicts(self):
        self.predict_test_kfolds = []
        self.predict_valid_kfolds = np.zeros((self.train.shape[0]))
    def model_train_with_kfold(self):
        self.clear_predicts()
        for (folder_index, (train_index, valid_index)) in enumerate(self.skf.split(self.train)):
            x_train, x_valid = self.train[train_index], self.train[valid_index]
            y_train, y_valid = self.train_target[train_index], self.train_target[valid_index]
            predict_valid, predict_test = self.model_fit_predict(x_train, y_train, x_valid, y_valid)
            self.predict_test_kfolds.append(predict_test)
            self.predict_valid_kfolds[valid_index] = predict_valid
            if self.logger != None:
                valid_score = self.score(y_valid, predict_valid)
                self.logger('Fold: %s, valid score: %s' % (folder_index, valid_score))
        self.logger('Full score: %s' % (self.score(self.train_target, self.predict_valid_kfolds)))
    def predict_test_mean(self):
        return reduce(lambda x,y:x+y, self.predict_test_kfolds)  / self.kfolds