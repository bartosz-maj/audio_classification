def knn_val():
  # Generating all hyperparameter values 
  k_list = []
  for i in range(1, 100):
    k_list.append(i)
  
  param_grid_knn = {"n_neighbors" : k_list}

  # Computing accuracies for non-transformed data
  knn_initial = KNeighborsClassifier()
  knn = GridSearchCV(knn_initial, param_grid_knn ,cv = 5, scoring = "balanced_accuracy")

  knn.fit(X_train, y_train)

  return knn.best_score_, knn.best_params_


knn_best_score, knn_best_params = knn_val()
knn_best_score, knn_best_params

def gnb_val():
  # Defining hyperparameters
  parameters_gnb = [1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10, 1e-11, 1e-12, 1e-13]
  param_gnb = {"var_smoothing" : parameters_gnb}

  gnb_init = GaussianNB()
  gnb = GridSearchCV(gnb_init, param_gnb, scoring = "balanced_accuracy")
  gnb.fit(X_train, y_train)

  
  return gnb.best_score_, gnb.best_params_

gnb_best_score, gnb_best_params = gnb_val()
gnb_best_score, gnb_best_params

def logistic_regression_val():
  c = [100, 10, 1.0, 0.1, 0.01]
  log_reg_param = {"C" : c}

  logreg_init = LogisticRegression(max_iter = 300, class_weight = "balanced")
  logreg = GridSearchCV(logreg_init, log_reg_param, scoring = "balanced_accuracy")
  logreg.fit(X_train, y_train)

  return logreg.best_score_, logreg.best_params_

logreg_best_score, logreg_best_params = logistic_regression_val()
logreg_best_score, logreg_best_params

def random_forest():
  n_estimators = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
  max_features = ['auto', 'sqrt']
  max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
  max_depth.append(None)
  min_samples_split = [2, 5, 10] 
  min_samples_leaf = [1, 2, 4]
  bootstrap = [True, False]

  random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

  rf = RandomForestClassifier()
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 80, cv = 5, scoring = "balanced_accuracy")
  rf_random.fit(X_train, y_train)


  return rf_random.best_score_, rf_random.best_params_

rf_random_best_score, rf_random_best_params = random_forest()
rf_random_best_score, rf_random_best_params

def ada_boost():
  ada_param_grid = {"n_estimators" : [10, 50, 100, 500],
                    "learning_rate" : [0.0001, 0.001, 0.01, 0.1, 1.0]}
  
  ada_init = AdaBoostClassifier()
  ada = GridSearchCV(estimator=ada_init, param_grid=ada_param_grid, cv=5, scoring = "balanced_accuracy")
  ada.fit(X_train, y_train)

  return ada.best_score_, ada.best_params_

ada_best_score, ada_best_params = ada_boost()
ada_best_score, ada_best_params

def voting_classifier():
  
  knn = KNeighborsClassifier(n_neighbors = 33)

  logreg = LogisticRegression(C = 100)

  gnb = GaussianNB(var_smoothing = 1e-05)

  rf = RandomForestClassifier(n_estimators = 1000,
                              min_samples_split = 5, 
                              min_samples_leaf = 4, 
                              max_features = "auto", 
                              max_depth = 110, 
                              bootstrap = True)

  ada = AdaBoostClassifier(learning_rate = 1.0, n_estimators = 500)
  
  vc = VotingClassifier(estimators = [("knn", knn), ("logreg", logreg), ("gnb", gnb),("rf", rf), ("ada", ada)], voting = "hard")
  vc.fit(X_train, y_train)


  return cross_val_score(vc, X_train, y_train, cv=5)

vc_score = voting_classifier()
sum(vc_score)/len(vc_score)

knn = KNeighborsClassifier(n_neighbors = 33)

logreg = LogisticRegression(C = 100)

gnb = GaussianNB(var_smoothing = 1e-05)

rf = RandomForestClassifier(n_estimators = 1000,
                            min_samples_split = 5, 
                            min_samples_leaf = 4, 
                            max_features = "auto", 
                            max_depth = 110, 
                            bootstrap = True)

ada = AdaBoostClassifier(learning_rate = 1.0, n_estimators = 500)
  
vc = VotingClassifier(estimators = [("knn", knn), ("logreg", logreg), ("gnb", gnb),("rf", rf), ("ada", ada)], voting = "hard")
vc.fit(X_train, y_train)
print(vc.score(X_train, y_train))
print(vc.score(X_val, y_val))

y_pred = vc.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

knn.fit(X_train, y_train)
print(knn.score(X_train, y_train))
print(knn.score(X_val, y_val))

y_pred = knn.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
print(rf.score(X_val, y_val))

y_pred = rf.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))

