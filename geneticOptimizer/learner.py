from matplotlib.pyplot import sca
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, TimeSeriesSplit
# import xgboost as xg
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
class Learner:
    def __init__(self, data) -> None:
        self.data=data
        self.modelsLearned = {}
        self.scaler = MinMaxScaler()
        pass

    def initOneMapperLearning(self,support,yTarget):
        # import pdb; pdb.set_trace()
        df =  self.data
        scaledData=(df-df.mean())/df.std()
        # scaledData=(df-df.min())/(df.max()-df.min())

        # X = scaledData[support].values.reshape(-1,1)
        # y = scaledData[yTarget].values.reshape(-1,1)
        X = scaledData[support].values.reshape(-1,1)
        y = scaledData[yTarget].values.reshape(-1,1)
        # for yTarget in approximated:
        # import pdb; pdb.set_trace()
        model, score = self.train(X, y)
        # model, score = self.train(X.values.reshape(-1,1),self.data[yTarget].values.reshape(-1,1))
        self.modelsLearned[f"{yTarget}|{support}"]= model
        print ("Model Recorded ....", f"{yTarget} =  f( {support} )" )
        return score
        pass

    def train(self, X, y, crossValidation=True):
        
        try:
            # import pdb; pdb.set_trace()
            # param_cv = {
            # 'min_child_weight': [1, 5, 10],
            # 'gamma': [0.5, 1, 1.5, 2, 5],
            # 'subsample': [0.6, 0.8, 1.0],
            # 'colsample_bytree': [0.6, 0.8, 1.0],
            # 'max_depth': [3, 4, 5],
            # 'scale_pos_weight' : [3,4,5]
            # }
            param_cv ={
                "criterion" : ["mse", "mae"], #  "poisson"],
                'max_depth': [3, 4, 5],
                "min_samples_split" : [0.01, 0.05, 0.1, 0.2, 0.4 ]
            }
            folds = 5
            param_comb = 6
            skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)
            # skf = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
            # import pdb;pdb.set_trace()
            # model = xg.XGBRegressor(objective ='reg:linear', n_estimators = 10, seed = 123)
            model = DecisionTreeRegressor()
            # random_search = RandomizedSearchCV(model, param_distributions=param_cv, n_iter=param_comb, scoring='r2', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )
            # random_search.fit(X, y)
            # model = random_search.best_estimator_
            model.fit(X, y)
            # import pdb; pdb.set_trace()
            y_pred = model.predict(X)
            score = self.checkMetrics(y_pred, y_true=y)
            return model, score
        except:
            print ("------XXXXXXXXXXX------------")
            return None, 10000
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
