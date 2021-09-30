from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import xgboost as xg
from sklearn.metrics import mean_squared_error

class Learner:
    def __init__(self, data) -> None:
        self.data=data
        self.modelsLearned = {}
        pass

    def initLearning(self,support, approximated):
        X = self.data[support]
        for yTarget in approximated:
            self.modelsLearned[f"{yTarget}|{support}"]=self.train(X,self.data[yTarget])
            print ("Model Recorded ....", f"{yTarget}|{support}" )
        pass

    def train(self, X, y, crossValidation=True):
        
        try:
            param_cv = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5],
            'scale_pos_weight' : [3,4,5]
            }
            folds = 5
            param_comb = 6
            skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)
            model = xg.XGBRegressor(objective ='reg:linear', n_estimators = 10, seed = 123)
            random_search = RandomizedSearchCV(model, param_distributions=param_cv, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )
            
            random_search.fit(X, y)
            model = random_search.best_estimator_
            return model
        except:
            pass

    def test(self, support, yTarget, X_test, y_true):
        # yTest = model.predict(X)
        y_pred = self.modelsLearned[f"{yTarget}|{support}"].predict(X_test)
        if type(y_true) == None:
            return -100
        else:
            return self.checkMetrics(y_pred, y_true)
        pass

    def checkMetrics(self, y_pred, y_true):
        return mean_squared_error(y_pred=y_pred, y_true=y_true)

    def reload(self, dataPath):

        pass

    def dump(self, outputPath):
        pass
