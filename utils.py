from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

DATA_DIR = Path('images')

def random_images_of_gender(samples, gender):
    """Fetches "samples" size list of random images of male / female class"""
    img_idxs = np.random.choice(50000, size=samples)  # random indices for random images
    path_arr =  np.array(list(Path(globals()['DATA_DIR'] / gender).iterdir()))[img_idxs]
    
    return [np.asarray(Image.open(filename)) for filename in path_arr]

def show_imgs(imgs, title='Title'):
    '''Fast plotter to inspect the data first time'''
    fig, axes = plt.subplots(2, 8, figsize=(30, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.set_title(f'Image size is {imgs[i].shape[0]}x{imgs[i].shape[1]}')
    plt.suptitle(title, size=40);
    plt.show();
   
def clf_proba_pipeline(models, X_train, X_test, y_train, y_test):
    clfs = (model for model in models)  # gen of classificators and their names
    y_pred_arr = {model: None for model in models}  # gen of probabilities and classificators names
    for clf in clfs:
        algo = models.get(clf)
        algo.fit(X_train, y_train)
        y_pred_proba = algo.predict_proba(X_test)
        y_pred_arr[clf] = y_pred_proba
    
    return y_pred_arr

def threshhold_clf_controller(models, proba_arr, y_test):
    y_pred_arr = {model: None for model in models}  # y_pred for each model
    accuracy_arr = {model: None for model in models}  # 100 accuracy values for each y_pred 
    for model in y_pred_arr:
        y_pred_arr[model] = np.array([np.where(proba_arr.get(model)[:, 1] > f, 1, 0)\
                           for f in np.linspace(0, 1, 100)], dtype='int64') # threshhold round of proba values for 100 vals in range(0, 1)
        accuracy_range = np.array([accuracy_score(y_test, y_pred_arr[model][i]) for i in range(100)], dtype='float32')  # array of 100 accuracy scores relative to 100 thresholds
        accuracy_arr[model] = accuracy_range
    
    return accuracy_arr

def GS_pipeline(X_train, X_test, y_train, y_test, 
                       model, param_grid, cv=5, scoring_fit='neg_mean_squared_error',
                       scoring_test=accuracy_score):
    '''Implements Grid Search on passed models with passed params and metrics'''
    print('Current model:',  model)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=0
    )
    fitted_model = gs.fit(X_train, y_train)
    best_model = fitted_model.best_estimator_
    
    y_pred = fitted_model.predict(X_test)
    score = scoring_test(y_pred, y_test)
    
    return [best_model, y_pred, score]
