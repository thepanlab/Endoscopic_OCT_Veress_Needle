'''
Veress Needle - Classification
Cross-Testing Script for Summit
8/11/2021
Justin Reynolds

Determines the winning model in each testing fold.
Trains a new model using 28K train images and the 
same configuration of the winning model for the 
specified (args) testing fold. 
'''

import sys
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
from scipy.stats import sem
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

''' 
Architectures: ResNet50, InceptionV3, Xception
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


if __name__ == '__main__':
    try:
        testvid = int(sys.argv[1])
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <testvid>")
    #testvid = 5
    arch_dict = {0: 'ResNet50', 1: 'InceptionV3', 2: 'Xception'}
    percentsign='%'

    # strings for import
    path2data = '/gpfs/alpine/bif121/proj-shared/veress/classification/data/preproc_v20210803/' # summit
    fin_images =    'c_images_1D_20210803.npy'
    fin_labels =    'c_labels_20210803.npy'
    fin_vid =       'c_vid_20210803.npy'
    fin_filenames = 'c_filenames_20210803.npy'

    # strings for export
    export_root_path = "/gpfs/alpine/bif121/proj-shared/veress/classification/" # summit
    version =          "scratch/CT_results/"
    export_path = export_root_path+version

    ## Input
    with open(path2data+fin_labels, 'rb') as f:
        labels_str = np.load(f) # input labels 
    with open(path2data+fin_filenames, 'rb') as f:
        filenames = np.load(f) # input filenames
    with open(path2data+fin_vid, 'rb') as f:
        vid = np.load(f) # input VID (i.e. subject ID [1-8])
    with open(path2data+fin_images, 'rb') as f:
        images_1D = np.load(f) # input images

    ## Organize the data
    print("---- S"+str(testvid)+" ----")
    vid_unique = np.unique(vid) # get unique subjects
    n_classes = len(np.unique(labels_str)) # get number of label classes
    images_1D = images_1D[..., np.newaxis] # satisfying tf needs
    # convert labels in str format to int
    # 0: fat, 1: muscle, 2: abd_space, 3: intestine
    labels = np.copy(labels_str) 
    labels[labels == "fat"] = 0
    labels[labels == "muscle"] = 1
    labels[labels == "abdominal space"] = 2
    labels[labels == "intestine"] = 3
    labels=labels.astype(int)
    data_mat=list(zip(filenames, vid, labels, images_1D)) # zip the imported data

    ## Partition into train and test sets
    bool_train = vid != testvid # bools for trainers
    bool_test = vid == testvid  # bools for testers
    # training sets
    X_train=images_1D[bool_train]
    y_train=labels[bool_train]
    vid_train=vid[bool_train]
    filenames_train=filenames[bool_train]
    # testing sets
    X_test=images_1D[bool_test]
    y_test=labels[bool_test]
    vid_test=vid[bool_test]
    filenames_test=filenames[bool_test]
    # partition stats
    train_frac = float(len(X_train)/(len(X_train)+len(X_test))*100)
    test_frac = float(len(X_test)/(len(X_train)+len(X_test))*100)
    # display some info about the partitioning
    print("training: size=%s, subjects=%s, %.1f%s of total"%(str(X_train.shape), str(np.unique(vid_train)), float(train_frac), percentsign))
    print("testing:  size=%s, subjects=%s, %.1f%s of total"%(str(X_test.shape), str(np.unique(vid_test)), float(test_frac), percentsign))
    ## Get the average cross-validation accuracy for each model 
    arch_avg_acc_temp_list = []
    arch_sem_temp_list = []
    path2data = '/gpfs/alpine/bif121/proj-shared/veress/classification/scratch/'
    # loop through architectures
    print('S%d avg validation accuracies:'%testvid)
    for a in arch_dict:
        # loop through validation folds
        this_val_acc_list = []
        for valvid in vid_unique:
            # skip when valvid is same as testvid
            if valvid == testvid:
                continue
            # read from file
            with open(path2data+'CV'+str(testvid)+'/cv_y_eval_S'+str(testvid)+'_V'+str(valvid)+'_'+arch_dict[a]+'.npy', 'rb') as f:
                # get cv_y_eval
                y_eval = np.load(f) # grab the validation accuracy
            this_val_acc_list.append(y_eval[1]) # store validation accuracy in list
        arch_avg_acc_temp_list.append(np.mean(this_val_acc_list)) # compute avg cross-validation score for current model
        mysem = sem(this_val_acc_list) # compute the std. err. of the mean of the cross-validation accuracy scores
        arch_sem_temp_list.append(mysem)
        print('\t- %s: %.4f%s ± %.4f%s'%(arch_dict[a], np.mean(this_val_acc_list), percentsign, mysem, percentsign))

    ## Select a winning model determined by the best average cross-validation score
    winning_acc = max(arch_avg_acc_temp_list)
    winning_idx = arch_avg_acc_temp_list.index(winning_acc)
    winning_sem = arch_sem_temp_list[winning_idx]
    winning_arch = arch_dict[winning_idx]
    print("\tWINNER: %s, %.4f%s ± %.4f%s"%(winning_arch, winning_acc, percentsign, winning_sem, percentsign))

    ## Preparing for new model
    # Hyperparameters 
    n_epochs = 20 # convert to input arg
    batch_size = 32 # convert to input arg
    loss='sparse_categorical_crossentropy' # classification
    my_metrics = ['accuracy'] # classfication
    lr=0.01
    nesterov=True
    mom=0.9
    dr=0.01
    image_shape = X_train[0].shape # all images are the same size
    n_classes = len(np.unique(labels)) # get number of label classes

    ## Configure architecture based on winner
    if winning_arch == arch_dict[0]:
        # ResNet50
        print("* S%d %s *" %(testvid, winning_arch))
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=nesterov, decay=dr)
        model = create_ResNet50_model(image_shape, n_classes)
    elif winning_arch == arch_dict[1]:
        # InceptionV3
        print("* S%d %s *" %(testvid, winning_arch))
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=nesterov, decay=dr)
        model = create_InceptionV3_model(image_shape, n_classes)
    elif winning_arch == arch_dict[2]:
        # Xception
        print("* S%d %s *" %(testvid, winning_arch))
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=nesterov, decay=dr)
        model = create_Xception_model(image_shape, n_classes)

    # assemble callbacks (or not)
    #early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    #time_cb = TimeHistory()
    model.compile(loss=loss, optimizer=opt, metrics=my_metrics) # compile model
    start_time = perf_counter() # start clock
    # train new model according to winner
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs)
    stop_time = perf_counter() # stop clock
    ct_train_time = stop_time-start_time # calculate time
    y_eval = model.evaluate(X_test, y_test) # evaluate model on test set !!! 

    start_interp_time = perf_counter() # start clock
    y_preds = model.predict(X_test) # get predictions on test set
    stop_interp_time = perf_counter() # stop clock
    time_interp_all = stop_interp_time - start_interp_time # interp time for all test images
    time_interp_per_photo = time_interp_all / float(len(X_test)) # interp time for single image
    time_interp_duo_temp=(time_interp_all, time_interp_per_photo)

    # display results
    print("\n*** %d cross-testing classification results (%d-class) ***"%(testvid, n_classes))
    print("export path:       "+str(export_path))
    print("train time:        "+str(ct_train_time)+" seconds ("+str(ct_train_time/60.0)+" mins)")
    print("val_loss:     "+str(y_eval[0]))
    print("val_accuracy: "+str(y_eval[1]))
    print("\n%.2f secs (%.2f mins) to interpret all %d test images "%(time_interp_all, float(time_interp_all/60.0), len(X_test)))
    print("and %.2f secs per photo (.2f mins) "%(time_interp_per_photo, float(time_interp_per_photo/60.0)))

    ## Export results
    # save the model (.h5)
    model.save(export_path+"S"+str(testvid)+"_"+str(winning_arch)+"_ct_winner.h5")
    # save train time (.npy)
    with open(export_path+"S"+str(testvid)+"_traintime.npy", 'wb') as f:
        np.save(f, np.array(ct_train_time))
    # save interp time (.npy)
    with open(export_path+"S"+str(testvid)+"_interptime.npy", 'wb') as f:
        np.save(f, np.array(time_interp_duo_temp))
    # save y_preds (.npy)
    with open(export_path+"S"+str(testvid)+"_y_preds.npy", 'wb') as f:
        np.save(f, np.array(y_preds))
    # save y_eval (.npy)
    with open(export_path+"S"+str(testvid)+"_y_eval.npy", 'wb') as f:
        np.save(f, np.array(y_eval))
    # save training history (.pickle)
    outfile = open(export_path+"S"+str(testvid)+"_trainhist.pickle", "wb")
    pickle.dump(history.history, outfile)
    outfile.close()
