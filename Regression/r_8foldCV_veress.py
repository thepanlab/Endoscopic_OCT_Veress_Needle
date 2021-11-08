'''
Veress Needle Regression
8-fold cross-validation
Train on 7 samples and evaluate on the remaining sample.
'''
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"
import pandas as pd
import numpy as np
import time
from time import perf_counter 
from timeit import default_timer as timer
from scipy.stats import sem
import pickle

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# ResNet50 model
def create_ResNet50_model(input_shape):
    #ResNet50
    base = keras.applications.resnet50.ResNet50(
        include_top=False, 
        weights=None, 
        input_tensor=None, 
        input_shape=input_shape, 
        pooling=None
    )
    pool = keras.layers.GlobalMaxPooling2D()(base.output)
    output = keras.layers.Dense(1, activation="linear")(pool)
    model = keras.models.Model(inputs=base.input, outputs=output)
    return model

# ResNet50V2 model
def create_ResNet50V2_model(input_shape):
    #ResNet50V2
    base = keras.applications.ResNet50V2(
        include_top=False, 
        weights=None, 
        input_tensor=None, 
        input_shape=input_shape, 
        pooling=None
    )
    pool = keras.layers.GlobalMaxPooling2D()(base.output)
    output = keras.layers.Dense(1, activation="linear")(pool)
    model = keras.models.Model(inputs=base.input, outputs=output)
    return model

def create_InceptionV3_model(input_shape):
    base = keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights=None)
    pool = keras.layers.GlobalMaxPooling2D()(base.output)
    output = keras.layers.Dense(1, activation="linear")(pool)
    model = keras.models.Model(inputs=base.inputs, outputs=output)
    return model

def create_Xception_model(input_shape):
    base = keras.applications.Xception(input_shape=input_shape, include_top=False, weights=None)
    pool = keras.layers.GlobalMaxPooling2D()(base.output)
    output = keras.layers.Dense(1, activation="linear")(pool)
    model = keras.models.Model(inputs=base.inputs, outputs=output)
    return model

# NasNetLarge
def create_NasNetLarge_model(input_shape):
    base = keras.applications.NASNetLarge(input_shape=input_shape, include_top=False, weights=None)
    x = keras.layers.GlobalMaxPooling2D()(base.output)
    x = keras.layers.Dense(1, activation="linear")(x)
    model = keras.models.Model(inputs=base.inputs, outputs=x)
    return model

# MobileNetV2
def create_MobileNetV2_model(input_shape):
    base = keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    x = keras.layers.GlobalMaxPooling2D()(base.output)
    x = keras.layers.Dense(1, activation="linear")(x)
    model = keras.models.Model(inputs=base.inputs, outputs=x)
    return model

if __name__ == '__main__':
    my_gpu=0
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[my_gpu], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

    np.random.seed(0)
    tf.random.set_seed(0)
    # import details
    path2data='/home/jreynolds/21summer/veress/regression/data/export_preprocess_20210803/'
    fn_filenames='filenames_20210803.npy'
    fn_vid='vid_20210803.npy'
    fn_distgroup='distgroup_20210803.npy'
    fn_idx='idx_20210803.npy'
    fn_label='labels_20210803.npy'
    fn_img='images_20210803.npy'
    # export details
    export_root='/home/jreynolds/21summer/veress/regression/data/export_model_results/'
    model_version='8fold_v20210823'
    this_export_path=export_root+model_version+'/'
    architecture_dict = {0: 'ResNet50', 1: 'InceptionV3', 2: 'Xception'}
    results_log={}
    n_epochs=20
    batch_size=32
    my_metrics = ['mape', 'mae', 'mse']
    with open(path2data+fn_filenames, 'rb') as f:
        _filenames=np.load(f)
    with open(path2data+fn_vid, 'rb') as f:
        _vid=np.load(f)
    with open(path2data+fn_distgroup, 'rb') as f:
        _distgroup=np.load(f)
    with open(path2data+fn_idx, 'rb') as f:
        _idx=np.load(f)
    with open(path2data+fn_label, 'rb') as f:
        _label=np.load(f)
    with open(path2data+fn_img, 'rb') as f:
        _img=np.load(f)
    vid_unique=np.unique(_vid)
    _img = _img[..., np.newaxis]
    data_mat=list(zip(_filenames, _vid, _distgroup, _idx, _label, _img))
    master_y_eval_dict={}
    master_y_preds_dict={}
    master_y_test_dict={}
    master_traintimes_dict={}
    master_interptimes_dict={}
    for testvid in vid_unique:
        test_bool = _vid == testvid
        X_test = _img[test_bool]
        y_test = _label[test_bool]
        train_bool = _vid != testvid
        X_train = _img[train_bool]
        y_train = _label[train_bool]
        fold_y_eval = {}
        fold_y_test = {}
        fold_y_preds = {}
        fold_traintimes = {}
        fold_interptimes = {}
        for arch in architecture_dict:
            img_shape=X_train[0].shape
            # Get the appropriate model architecture with the if-elif-elif,
            #  declare callbacks, and initialize the optimizer.
            if arch == 0:
                # ResNet50
                print("** S"+str(testvid)+" "+str(architecture_dict[arch])+" **")
                model = create_ResNet50_model(img_shape)
            elif arch == 1:
                # InceptionV3
                print("** S"+str(testvid)+" "+str(architecture_dict[arch])+" **")
                model = create_InceptionV3_model(img_shape)
            elif arch == 2:
                # Xception
                print("** S"+str(testvid)+" "+str(architecture_dict[arch])+" **")
                model = create_Xception_model(img_shape)
            opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
            model.compile(loss=keras.losses.MeanAbsolutePercentageError(), optimizer=opt, metrics=my_metrics)
            my_trainers=[i for i in _vid if i != testvid]
            unique_trainers=np.unique(my_trainers)
            print("%d train images from subjects %s"%(len(X_train), np.unique()))
            print("%d test images from subject %d"%(len(X_test), testvid))
            print("--- train ---")
            # Fit the model
            t1_start = perf_counter() 
            history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)
            t1_stop = perf_counter()
            train_time = t1_stop - t1_start
            print("---- S%d %s Testing Performance ----"%(testvid, architecture_dict[arch]))
            y_eval = model.evaluate(X_test, y_test)
            t2_start = perf_counter()
            y_preds = model.predict(X_test)
            t2_stop = perf_counter()
            interp_time = t2_stop - t2_start
            # store results in dict
            fold_y_eval[architecture_dict[arch]] = y_eval
            fold_y_preds[architecture_dict[arch]] = y_preds
            fold_y_test[architecture_dict[arch]] = y_test
            fold_traintimes[architecture_dict[arch]] = train_time
            fold_interptimes[architecture_dict[arch]] = interp_time
            print("---- S%d %s results ----"%(testvid, architecture_dict[arch]))
            print("y_eval: ", y_eval)
            print("train_time:  %.4f secs"%train_time)
            print("interp_time: %.4f secs"%interp_time)
            print("\n")
            # end architecture loop
        master_y_eval_dict[testvid]=fold_y_eval
        master_y_preds_dict[testvid]=fold_y_preds
        master_y_test_dict[testvid]=fold_y_test
        master_traintimes_dict[testvid]=fold_traintimes
        master_interptimes_dict[testvid]=fold_interptimes
        # end subject ID loop
    print("master_y_eval_dict:")
    print(master_y_eval_dict)
    print("master_traintimes_dict:")
    print(master_traintimes_dict)
    print("master_interptimes_dict:")
    print(master_interptimes_dict)
    # export the results
    model.save("%sf8_model_S%d_%s.h5"%(this_export_path, testvid, architecture_dict[arch]))
    of = open(this_export_path+"f8_y_eval.pickle", "wb")
    pickle.dump(master_y_eval_dict, of)
    of.close()
    of = open(this_export_path+"f8_y_preds.pickle", "wb")
    pickle.dump(master_y_preds_dict, of)
    of.close()
    of = open(this_export_path+"f8_y_tests.pickle", "wb")
    pickle.dump(master_y_test_dict, of)
    of.close()
    of = open(this_export_path+"f8_traintimes.pickle", "wb")
    pickle.dump(master_traintimes_dict, of)
    of.close()
    of = open(this_export_path+"f8_master_time.pickle", "wb")
    pickle.dump(master_interptimes_dict, of)
    of.close()
