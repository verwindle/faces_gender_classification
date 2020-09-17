import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly_express as px


def hist_plotly(x, title_, x_label, y_label, x_name):
    fig = go.Figure()
    for _, name in zip(x, x_name): 
        fig.add_trace(go.Histogram(x=_,\
                               name=name, opacity=.9))
        
    fig.update_layout(
        xaxis_title_text=x_label, # xaxis label
        yaxis_title_text=y_label, # yaxis label
        bargap=0.1, # gap between bars of adjacent location coordinates
        title={ # title of plot
        'text': title_,
        'font': dict(size=50),
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    )
    fig.show()

def scatter_plotly(x, y, mode, name, title_, x_label, y_label, fill=None, marker_color=None):
    fig = go.Figure()
    
    for x_, y_, name_ in zip(x, y, name):
        fig.add_trace(go.Scattergl(
                    x=x_,
                    y=y_,
                    mode=mode,
                    name=name_,
                    fill=fill,
                    marker_color=marker_color
                    ))


    fig.update_layout(
        xaxis_title_text=x_label, # xaxis label
        yaxis_title_text=y_label, # yaxis label
        title={ # title of plot
        'text': title_,
        'font': dict(size=50),
        'y':1.0,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
    )

    fig.show()
    
def eigen_faces(pca_model, gender, cmap='cool'):
    number_of_eigenfaces = len(pca_model.components_)
    eigen_faces = pca_model.components_.reshape((number_of_eigenfaces, 128, 128))

    cols = 10
    rows = int(number_of_eigenfaces / cols)
    fig, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(30, 8))  # eigen faces plot grid
    axarr = axarr.flatten()
    for i in range(number_of_eigenfaces):
        axarr[i].imshow(eigen_faces[i], cmap=cmap)
        axarr[i].set_xticks([])  # erase plot ticks 
        axarr[i].set_yticks([])  # erase plot ticks
        axarr[i].set_title(f'Component {i+1}')  # ax title
    plt.suptitle(f'{number_of_eigenfaces} {gender.capitalize()} Eigen Faces', fontsize=40);  # title
