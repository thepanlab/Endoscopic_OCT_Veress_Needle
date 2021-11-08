'''
Veress Needle Regression 
Justin Reynolds
Nested CrossValidation and CrossTesting
CNN models to estimate the distance from the
veress needle to the intestine. 
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

np.random.seed(0)
tf.random.set_seed(0)
my_gpu = 0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[my_gpu], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

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
    base_model_empty = keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling=None)
    avg = keras.layers.GlobalAveragePooling2D()(base_model_empty.output)
    output = keras.layers.Dense(1, activation="linear")(avg)
    model_resnet50 = keras.models.Model(inputs=base_model_empty.input, outputs=output)
    return model_resnet50

def create_InceptionV3_model(input_shape):
    base = keras.applications.InceptionV3(input_shape=input_shape, include_top=False, weights=None)
    x = keras.layers.GlobalMaxPooling2D()(base.output)
    x = keras.layers.Dense(1, activation="linear")(x)
    model = keras.models.Model(inputs=base.inputs, outputs=x)
    return model

def create_Xception_model(input_shape):
    base = keras.applications.Xception(input_shape=input_shape, include_top=False, weights=None)
    x = keras.layers.GlobalMaxPooling2D()(base.output)
    x = keras.layers.Dense(1, activation="linear")(x)
    model = keras.models.Model(inputs=base.inputs, outputs=x)
    return model

def create_NasNetLarge_model(input_shape):
    base = keras.applications.NASNetLarge(input_shape=input_shape, include_top=False, weights=None)
    x = keras.layers.GlobalMaxPooling2D()(base.output)
    x = keras.layers.Dense(1, activation="linear")(x)
    model = keras.models.Model(inputs=base.inputs, outputs=x)
    return model

if __name__ == '__main__':
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
    model_version='v20210803'
    this_export_path=export_root+model_version+'/'
    export_path_train=export_root+model_version+'/CV_results/'
    export_path_test=export_root+model_version+'/CT_results/'

    architecture_dict = {0: 'ResNet50', 1: 'InceptionV3', 2: 'Xception'}
    results_log={}

    n_epochs=2
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

    total_fold_time=0.0
    start_time = time.time()
    cv_master_mape = {}
    cv_master_mae = {}
    cv_master_time = {}
    cv_master_mean_mape_and_sem = {}
    cv_master_mean_mae_and_sem = {}

    print("Veress Needle Regression")
    print("\timport data path: "+path2data)
    print("\texport data path: "+this_export_path)
    print("\tnumber of images: "+str(len(_img)))
    print("\tsubject list:     "+str(vid_unique))
    print("\n")
    # Cross-testing loop
    for testvid in vid_unique:
        arch_mean_mape_dict = {}
        arch_mean_mae_and_sem_dict = {}
        arch_mean_mape_and_sem_dict = {}
        mape_dict_arch={}
        mae_dict_arch={}
        time_dict_arch={}
        # Split training, validation, and testing sets
        X_test_list = [data_mat[i][5] for i in range(len(data_mat)) if data_mat[i][1] == testvid and data_mat[i][1] in vid_unique]
        y_test_list = [data_mat[i][4] for i in range(len(data_mat)) if data_mat[i][1] == testvid and data_mat[i][1] in vid_unique]
        test_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] == testvid and data_mat[i][1] in vid_unique]
        X_test=np.array(X_test_list)
        y_test=np.array(y_test_list)
        unique_test_vid = np.unique(test_check)
        # Get training images without validation set for best model in current fold
        X_train_all_list = [data_mat[i][5] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] in vid_unique]
        y_train_all_list = [data_mat[i][4] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] in vid_unique]
        train_all_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] in vid_unique]
        X_train_all = np.array(X_train_all_list)
        y_train_all = np.array(y_train_all_list)
        unique_train_all_vid = np.unique(train_all_check)
        # loop through the architecture configurations
        for arch in architecture_dict:

            print("\nstarting S"+str(testvid)+" "+str(architecture_dict[arch])+"...")
            cv_time_list = []
            cv_mape_list = []
            cv_mae_list=[]
            train_time_list = []
            # nested cross-validation loop
            for valvid in vid_unique:
                # skip when validation set subject == test set subject
                if valvid == testvid:
                    train_time_list.append(0.0)
                    print("---> skipping validation with valvid("+str(valvid)+") == testvid("+str(testvid)+")")
                    continue
                print("\t ************")
                print("\t ** S"+str(testvid)+", V"+str(valvid)+" **")
                # partition validation data
                X_val_list = [data_mat[i][5] for i in range(len(data_mat)) if data_mat[i][1] == valvid and data_mat[i][1] in vid_unique]
                y_val_list = [data_mat[i][4] for i in range(len(data_mat)) if data_mat[i][1] == valvid and data_mat[i][1] in vid_unique]
                val_check = [data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] == valvid and data_mat[i][1] in vid_unique]
                # partition training data
                X_train_list = [data_mat[i][5] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid and data_mat[i][1] in vid_unique]
                y_train_list = [data_mat[i][4] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid and data_mat[i][1] in vid_unique]
                train_check=[data_mat[i][1] for i in range(len(data_mat)) if data_mat[i][1] != testvid and data_mat[i][1] != valvid and data_mat[i][1] in vid_unique]
                # convert validation and training list to np.array
                X_val=np.array(X_val_list)
                y_val=np.array(y_val_list)
                unique_train_vid = np.unique(train_check)

                X_train=np.array(X_train_list)
                y_train=np.array(y_train_list)
                unique_val_vid = np.unique(val_check)
                # confirm training split
                trainers=[]
                for i in range(len(data_mat)):
                    if data_mat[i][1] != testvid and data_mat[i][1] != valvid:
                        trainers.append(data_mat[i][1])

                img_shape=X_train[0].shape
                # Get the appropriate model architecture with the if-elif-elif, 
                #  declare callbacks, and initialize the optimizer. 
                if arch == 0:
                    # ResNet50
                    print("\t ** ResNet50 **")
                    model = create_ResNet50_model(img_shape)
                elif arch == 1:
                    # InceptionV3
                    print("\t ** InceptionV3 **")
                    model = create_InceptionV3_model(img_shape)
                elif arch == 2:
                    # Xception
                    print("\t ** Xception **")
                    model = create_Xception_model(img_shape) 
                # display summary of splits
                print("\t --- split summary ---")
                print("\t training set:   S"+str(unique_train_vid)+", "+str(len(X_train_list))+" images")
                print("\t validation set: S"+str(valvid)+", "+str(len(X_val_list))+" images")
                print("\t testing set:    S"+str(testvid)+", "+str(len(X_test_list))+" images")
                # Optimizer
                opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
                # Compile the model
                model.compile(loss=keras.losses.MeanAbsolutePercentageError(), optimizer=opt, metrics=my_metrics)
                print("--- train ---")
                # Fit the model
                t1_start = perf_counter() 
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=batch_size)
                t1_stop = perf_counter()
                train_time = t1_stop-t1_start
                train_time_list.append(train_time)
                # Evaluate 
                print("--- Validate - S%d - V%d - %s ---" %(testvid, valvid, architecture_dict[arch]))
                my_model_val = model.evaluate(X_val, y_val)
                # append validation score 
                cv_mape_list.append(my_model_val[1]) # mapes
                cv_mae_list.append(my_model_val[2]) # maes
                # end nested cross-validation loop
            my_mean_mape = np.mean(cv_mape_list)
            my_mean_mae = np.mean(cv_mae_list)
            my_sem = sem(cv_mape_list)
            my_sem2 = sem(cv_mae_list)
            my_duo = (my_mean_mape, my_sem)
            my_duo2 = (my_mean_mae, my_sem2)
            arch_mean_mape_dict[arch] = my_mean_mape 
            arch_mean_mape_and_sem_dict[arch] = my_duo
            arch_mean_mae_and_sem_dict[arch] = my_duo2
            mape_dict_arch[architecture_dict[arch]] = cv_mape_list
            mae_dict_arch[architecture_dict[arch]] = cv_mae_list
            time_dict_arch[architecture_dict[arch]] = train_time_list
            print("\n--- done with cross validation - S%d %s" %(testvid, str(architecture_dict[arch])))
            print("\narch_mean_MAPE_and_sem_dict:")
            for i in arch_mean_mape_and_sem_dict:
                print("\t%s: %.4f ± %.4f" %(architecture_dict[i], arch_mean_mape_and_sem_dict[i][0], arch_mean_mape_and_sem_dict[i][1]))
            print("arch_mean_MAE_and_sem_dict:")
            for i in arch_mean_mae_and_sem_dict:
                print("\t%s: %.4f ± %.4f" %(architecture_dict[i], arch_mean_mae_and_sem_dict[i][0], arch_mean_mae_and_sem_dict[i][1]))
            avg_cv_time = float(sum(train_time_list)/(len(train_time_list)-1.0))
            print("\navg train time for "+str(int(len(vid_unique)-2))+"-fold CV S"+str(testvid)+"-"+str(architecture_dict[arch])+": "+str(avg_cv_time))
            # end architecture configuration loop
        print("\n--- done with cross validation S%d - all archs" %testvid)
        cv_master_mape[testvid] = mape_dict_arch
        cv_master_mae[testvid] = mae_dict_arch
        cv_master_time[testvid] = time_dict_arch
        cv_master_mean_mape_and_sem[testvid] = arch_mean_mape_and_sem_dict
        cv_master_mean_mae_and_sem[testvid] = arch_mean_mae_and_sem_dict
        # Using the config (i.e. resnet50, inceptionV3, xception) with the
        # lowest average CV score in this testing fold. 
        best_arch_mean = 999999
        best_arch=''
        for i in arch_mean_mape_dict:
            if arch_mean_mape_dict[i] < best_arch_mean:
                best_arch_mean = arch_mean_mape_dict[i]
                best_arch = architecture_dict[i]
        print("--- finding best S%d configuration" %(testvid))
        print("arch_mean_dict:")
        for i in arch_mean_mape_dict:
            print("%s: %.4f" %(architecture_dict[i], arch_mean_mape_dict[i]))
        print("\nWINNER -- best_arch=", best_arch, ", best_arch_mean=", best_arch_mean)
        print("\n--- training new %s model with all %d train images for S%d fold" %(str(best_arch), int(len(X_train_all)), testvid)) 
        # Train a new model based off the best of the models in the current testing fold. 
        if best_arch == architecture_dict[0]:
            print(" ** ResNet50 **")
            time_cb = TimeHistory()
            opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
            model = create_ResNet50_model(img_shape)
        elif best_arch == architecture_dict[1]:
            print(" ** InceptionV3 **")
            time_cb = TimeHistory()
            opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
            model = create_InceptionV3_model(img_shape)
        elif best_arch == architecture_dict[2]:
            print(" ** Xception **")
            time_cb = TimeHistory()
            opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, decay=0.01)
            model = create_Xception_model(img_shape)
        print(" ** testvid = "+str(testvid)+" **")
        print(len(X_train_all), "train images from subjects", unique_train_all_vid)
        print(len(X_test), "test images from subject", unique_test_vid)
        # Compile the new model
        model.compile(loss=keras.losses.MeanAbsolutePercentageError(), optimizer=opt, metrics=my_metrics)
        # Fit the new model without validation data
        history = model.fit(X_train_all, y_train_all, epochs=n_epochs, batch_size=batch_size, callbacks=[time_cb])
        times = time_cb.times
        print(f'total time for %d epochs is %.3f secs or %.3f mins' % (n_epochs, sum(times), sum(times)/60.0))
        print(f'average time per epoch is %.3f secs or %.3f mins' % (np.mean(times), np.mean(times)/60.0))
        print("--- test new model on unseen S%s ---" %str(testvid))
        test_eval = model.evaluate(X_test, y_test)
        print("\n")
        y_preds = model.predict(X_test)
        model.save(this_export_path_test+'ct_model_S%s_%s.h5' %(str(testvid), str(best_arch))) # creates a HDF5 file 'my_model.h5'
        my_history = history.history
        hist_df = pd.DataFrame(history.history)
        print("\nTraining history of new model")
        print(hist_df)
        print("\nExporting results for ct fold", testvid)
        print("\ttraining history")
        outfile1 = open(export_path_test+"ct_S"+str(testvid)+"_trainhist_"+str(best_arch), "wb")
        pickle.dump(my_history, outfile1)
        outfile1.close()
        print("\ty_preds")
        outfile2 = open(export_path_test+"ct_S"+str(testvid)+"_y_preds_"+str(best_arch), "wb")
        pickle.dump(y_preds, outfile2)
        outfile2.close()
        print("\ty_test")
        outfile3 = open(export_path_test+"S"+str(testvid)+"_y_test_"+str(best_arch), "wb")
        pickle.dump(y_test, outfile3)
        outfile3.close()
        print("\ttimes")
        outfile4 = open(export_path_test+"S"+str(testvid)+"_test_times_"+str(best_arch), "wb")
        pickle.dump(times, outfile4)
        outfile4.close()
        print("\tX_test")
        outfile5 = open(export_path_test+"S"+str(testvid)+"_X_test_"+str(best_arch), "wb")
        pickle.dump(X_test, outfile5)
        outfile5.close()
        print("\ttest evaluation")
        outfile6 = open(this_export_path_test+"S"+str(testvid)+"_testEval_"+str(best_arch), "wb")
        pickle.dump(test_eval, outfile6)
        outfile6.close()
        total_fold_time = total_fold_time + (time.time() - start_time)
        print("\n#-#-#-#-#-#-# Done with S%s" %str(testvid))
        print(f'#-#-#-#-#-#-# time: %.2f seconds, %.2f mins, %.2f hrs' %(total_fold_time, total_fold_time/60.0, total_fold_time/3600.0))
        print("#-#-#-#-#-#-# best_arch:  ", best_arch)
        print("#-#-#-#-#-#-# test_score: ", test_eval[1], ", ", test_eval[2])
        print("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#\n")
        print("CV RESULTS\n")
        print("cv_master_mape:")
        print(cv_master_mape)
        print("cv_master_mae:")
        print(cv_master_mae)
        print("cv_master_mean_mape_and_sem:")
        print(cv_master_mean_mape_and_sem)
        print("cv_master_mean_mae_and_sem:")
        print(cv_master_mean_mae_and_sem)
        print("cv_master_time:")
        print(cv_master_time)
        print("\n")
        # end cross-testing loop

    print("exporting all training results")
    of = open(export_path_train+"cv_master_mape", "wb")
    pickle.dump(cv_master_mape, of)
    of.close()
    of = open(export_path_train+"cv_master_mae", "wb")
    pickle.dump(cv_master_mae, of)
    of.close()
    of = open(export_path_train+"cv_master_mean_mape_and_sem", "wb")
    pickle.dump(cv_master_mean_mape_and_sem, of)
    of.close()
    of = open(this_export_path_train+"cv_master_mean_mae_and_sem", "wb")
    pickle.dump(cv_master_mean_mae_and_sem, of)
    of.close()
    of = open(export_path_train+"cv_master_time", "wb")
    pickle.dump(cv_master_time, of)
    of.close()
    of = open(export_path_train+"GUIDE_arch_idx_key_mapping", "wb")
    pickle.dump(architecture_dict, of)
    of.close()
