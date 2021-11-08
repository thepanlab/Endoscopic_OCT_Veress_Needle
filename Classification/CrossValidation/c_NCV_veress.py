'''
Veress Needle Classification, Nested Cross-Validation
Justin Reynolds
CNN to classify images of fat, muscle, abdominal space, and intestine.
'''
import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import numpy as np
import os
import pickle
from time import perf_counter
from timeit import default_timer as timer

keras.backend.clear_session()
np.random.seed(0)
tf.random.set_seed(0)

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"])
        )
'''
Architectures: ResNet50, InceptionV3, Xception, NasNetLarge
'''
# ResNet50
def create_ResNet50_model(input_shape, n_classes):
    #ResNet50
    base = keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling=None)
    pool = keras.layers.GlobalMaxPooling2D()(base.output)
    #pool = keras.layers.GlobalAveragePooling2D()(base.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(pool)
    model = keras.models.Model(inputs=base.inputs, outputs=output)
    return model
# InceptionV3
def create_InceptionV3_model(input_shape, n_classes):
    base = keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights=None)
    pool = keras.layers.GlobalMaxPooling2D()(base.output)
    #pool = keras.layers.GlobalAveragePooling2D()(base.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(pool)
    model = keras.models.Model(inputs=base.inputs, outputs=output)
    return model
# Xception
def create_Xception_model(input_shape, n_classes):
    base = keras.applications.Xception(input_shape=input_shape, include_top=False, weights=None)
    pool = keras.layers.GlobalMaxPooling2D()(base.output)
    #pool = keras.layers.GlobalAveragePooling2D()(base.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(pool)
    model = keras.models.Model(inputs=base.inputs, outputs=output)
    return model
# NasNetLarge
def create_NasNetLarge_model(input_shape, n_classes):
    base = keras.applications.NASNetLarge(input_shape=input_shape, include_top=False, weights=None)
    pool = keras.layers.GlobalMaxPooling2D()(base.output)
    #pool = keras.layers.GlobalAveragePooling2D()(base.output)
    output = keras.layers.Dense(n_classes, activation="softmax")(pool)
    model = keras.models.Model(inputs=base.inputs, outputs=output)
    return model

# main driver function
if __name__ == '__main__':
    # strings for import
    #path2data = '/ccs/home/jreynolds/21summer/veress/classification/data/preprocessed_data/v20210803/'
    path2data = '/gpfs/alpine/bif121/proj-shared/veress/classification/data/preproc_v20210803/'
    fin_images =    'c_images_1D_20210803.npy'
    fin_labels =    'c_labels_20210803.npy'
    fin_vid =       'c_vid_20210803.npy'
    fin_filenames = 'c_filenames_20210803.npy'

    try:
        testvid = int(sys.argv[1])
        valvid = int(sys.argv[2])
        arch_idx = int(sys.argv[3])
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <testvid>  <valvid> <arch_idx>")
    if arch_idx == 0:
        arch = 'ResNet50'
    elif arch_idx == 1:
        arch = 'InceptionV3'
    elif arch_idx == 2:
        arch = 'Xception'
    elif arch_idx == 3:
        arch = 'NasNetLarge'
    else:
        print("ERROR - options include 0 (ResNet50), 1 (InceptionV3), or 2 (Xception)\ntry again...")
        exit()

    # strings for export
    #export_path = "/gpfs/alpine/bif121/proj-shared/veress/classification/v20210805_0/S"+str(testvid)+"/" # summit
    export_root_path = "/gpfs/alpine/bif121/proj-shared/veress/classification/" # summit
    version =          "v20210811/CV"+str(testvid)+"/"
    export_path = export_root_path+version
    # Hyperparameters
    n_epochs = 20 # convert to input arg
    batch_size = 32 # convert to input arg
    #my_metrics = ['mape', 'mae', 'mse'] # regression
    my_metrics = ['accuracy'] # classfication

    # input the data
    with open(path2data+fin_labels, 'rb') as f:
        labels_str = np.load(f) # input labels
    with open(path2data+fin_filenames, 'rb') as f:
        filenames = np.load(f) # input filenames
    with open(path2data+fin_vid, 'rb') as f:
        vid = np.load(f) # input VID
    with open(path2data+fin_images, 'rb') as f:
        images_1D = np.load(f) # input images

    vid_unique = np.unique(vid) # get unique subjects
    n_classes = len(np.unique(labels_str)) # get number of label classes
    images_1D = images_1D[..., np.newaxis] # satisfying tf needs
    # convert labels in str format to int
    labels = np.copy(labels_str)
    labels[labels == "fat"] = 0
    labels[labels == "muscle"] = 1
    labels[labels == "abdominal space"] = 2
    labels[labels == "intestine"] = 3
    labels=labels.astype(int)

    data_mat=list(zip(filenames, vid, labels, images_1D)) # zip the imported data

    # Get X_test and y_test
    test_img_list = [data_mat[i][3] for i in range(len(data_mat)) if data_mat[i][1] == testvid]
    test_dist_list = [data_mat[i][2] for i in range(len(data_mat)) if data_mat[i][1] == testvid]
    test_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] == testvid]
    # validation data
    val_img_list = [data_mat[i][3] for i in range(len(data_mat)) if data_mat[i][1] == valvid]
    val_dist_list = [data_mat[i][2] for i in range(len(data_mat)) if data_mat[i][1] == valvid]
    val_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] == valvid]
    # training data
    #train_img_list = [data_mat[i][3] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid and data_mat[i][1] in vid_unique]
    train_img_list = [data_mat[i][3] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid]
    #train_dist_list = [data_mat[i][2] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid and data_mat[i][1] in vid_unique]
    train_dist_list = [data_mat[i][2] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid]
    #train_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid and data_mat[i][1] in vid_unique]
    train_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid]

    # convert from list to np.array
    X_train = np.array(train_img_list)
    y_train = np.array(train_dist_list)
    X_val = np.array(val_img_list)
    y_val = np.array(val_dist_list)
    X_test = np.array(test_img_list)
    y_test = np.array(test_dist_list)

    # get the unique subject eids (insurance prep)
    unique_train_vid = np.unique(train_check)
    unique_val_vid = np.unique(val_check)
    unique_test_vid = np.unique(test_check)

    # Confirming integrity of splits
    # insurance covering x and y are same length
    if len(X_train) != len(y_train):
        print("ERROR - length mismatch len(X_train)=", len(X_train), ", len(y_train)=", len(y_train))
        exit()
    if len(X_val) != len(y_val):
        print("ERROR - length mismatch len(X_val)=", len(X_val), ", len(y_val)=", len(y_val))
        exit()
    if len(X_test) != len(y_test):
        print("ERROR - length mismatch len(X_test)=", len(X_test), ", len(y_test)=", len(y_test))
        exit()
    # insurance covering it belongs in the train/val/test set
    for i in range(len(train_check)):
        if train_check[i] == testvid or train_check[i] == valvid:
            print("ERROR - train set contamination, train_check[", i, "]=", train_check[i], " belongs elsewhere.")
            exit()
    for i in range(len(val_check)):
        if val_check[i] != valvid:
            print("ERROR - validation set contamination, val_check[", i, "]=", val_check[i], " belongs elsewhere.")
            exit()
    for i in range(len(test_check)):
        if test_check[i] != testvid:
            print("ERROR - validation set contamination, test_check[", i, "]=", test_check[i], " belongs elsewhere.")
            exit()

    image_shape = X_train[0].shape # all images are the same size

    ### Configure Arch ###
    # model architecture configuration according to argv[2],
    # declare callbacks, and initialize the optimizer.
    if arch_idx == 0:
        # ResNet50
        print("* S%d - V%d - %s *" %(testvid, valvid, arch))
        opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
        model = create_ResNet50_model(image_shape, n_classes)
    elif arch_idx == 1:
        # InceptionV3
        print("* S%d - V%d - %s *" %(testvid, valvid, arch))
        opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
        model = create_InceptionV3_model(image_shape, n_classes)
    elif arch_idx == 2:
        # Xception
        print("* S%d - V%d - %s *" %(testvid, valvid, arch))
        opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
        model = create_Xception_model(image_shape, n_classes)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    #time_cb = TimeHistory()
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=my_metrics) # compile model
    start_time = perf_counter() # start clock
    # train model
    history = model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_val, y_val), epochs=n_epochs, callbacks=[early_stopping_cb])
    stop_time = perf_counter() # stop clock
    train_time = stop_time-start_time # calculate time
    y_eval = model.evaluate(X_val, y_val) # evaluate with X_val
    y_preds = model.predict(X_val) # providing predictions on validation set
    # display results
    print("\n*** Classification Results ***")
    print("S"+str(testvid)+"-V"+str(valvid)+"-"+str(arch))
    print("export path:       "+str(export_path))
    print("train time:        "+str(train_time)+" seconds ("+str(train_time/60.0)+" mins)")
    print("y_eval (accuracy): "+str(y_eval))
    # save results
    # export model (.h5)
    model.save(export_path+"S"+str(testvid)+"_V"+str(valvid)+"_"+str(arch)+".h5")
    with open(export_path+"cv_time_S"+str(testvid)+"_V"+str(valvid)+"_"+str(arch)+".npy", 'wb') as f:
        np.save(f, np.array(train_time))
    # export y_preds (.npy)
    with open(export_path+"cv_y_preds_S"+str(testvid)+"_V"+str(valvid)+"_"+str(arch)+".npy", 'wb') as f:
        np.save(f, np.array(y_preds))
    # export y_eval (.npy)
    with open(export_path+"cv_y_eval_S"+str(testvid)+"_V"+str(valvid)+"_"+str(arch)+".npy", 'wb') as f:
        np.save(f, np.array(y_eval))
    # export training history (.pickle)
    outfile = open(export_path+"cv_trainhist_S"+str(testvid)+"_V"+str(valvid)+"_"+str(arch)+".pickle", "wb")
    pickle.dump(history.history, outfile)
    outfile.close()
    #with open(export_path+"cv_y_eval_S"+str(testvid)+"_V"+str(valvid)+"_"+str(arch)+".pickle", 'wb') as p:
    #    pickle.dump(history.history, p)

    del model # get rid of the model
