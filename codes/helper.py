from keras.preprocessing import image
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display
import numpy as np
import os 

def get_wrong_pic_list(model, preprocess_input, decode_predictions, top_number=50):    
    pic_list = [] 

    # loop through all the files and make predictions
    for root, dirs, files in os.walk("train/", topdown=True):
    #     for name in log_progress(files[:i], name="Progress"):
        for name in log_progress(files, name="Progress"):
            if name[0] == 'c' or name[0] == 'd':    # exclude system hidden file like '.DS_Store' 
                image_path = root + "/" + name
                img = image.load_img(image_path, target_size=(224, 224)) 
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)
                decode_preds = decode_predictions(preds, top=top_number)[0]
                is_good_picture = False      # assume it is a wrong picture

                # loop through all predictions and
                # check if the predicition contains cats or dogs
                for i in decode_preds:
                    if not is_good_picture: 
                        if i[0] in dogs or i[0] in cats:
                            is_good_picture = True
                    else:
                        # break the loop if cat or dog is found
                        break

                # if there is no dog or cat in the picture, then it is a bad picture
                if not is_good_picture:
                    pic_list.append(image_path)
                    
    print(str(len(pic_list)) + " wrong pictures are found!")                
    return pic_list




# from: https://github.com/alexanderkuk/log-progress
def log_progress(sequence, every=None, size=None, name='Items'):

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


def showHistory(history):
    import matplotlib.pyplot as plt

    # list all data in history
    # print(history.history.keys())
    # print(history.history['acc'])

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()