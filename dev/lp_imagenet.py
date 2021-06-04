import numpy as np
import pandas as pd

from time import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

import pickle

from utils import nn_score, input_transformer_gaussian, batch_normal_kernel
from sampling_tools import ImportanceSplittingLpBatch
import tensorflow as tf
from math import ceil
import os
import envir_def 
import sys, getopt

nb_samples = envir_def.NB_TEST_SAMPLES
LOG_DIR = envir_def.EFFICIENT_STAT_PATH+"logs/ImageNet/"
DATA_PATH = envir_def.EFFICIENT_STAT_PATH+"data/ImageNet/"
DIM = 3*224*224
gaussian_gen = lambda N: np.random.normal(size =(N,DIM))
print(f"Number of GPU used: {len(tf.config.list_physical_devices('GPU'))}")

X_test, y_test = [], []
tests_pkl = os.path.join(DATA_PATH, 'imagenet_test.pkl')
tests = []
if os.path.exists(tests_pkl):
        print('.pkl with files exists, loading from it')
        with open(tests_pkl, 'rb') as testsf:
            tests = pickle.load(testsf)

else:
    val_path = DATA_PATH+'val'
    data_gen = ImageDataGenerator()
    seed=3
    test_generator = data_gen.flow_from_directory(val_path,
                                                            target_size=(224, 224),
                                                            seed=seed,
                                                            batch_size=32)
    i = 0
    for test in test_generator:
        #print('test batch {}; img {}'.format(test[0].shape, test[0][0].shape))
        #print('correct batch {}'.format(i, test[1].shape))
        img_batch = np.array([img_to_array(x) for x in test[0]])

        #correct_labels = decode_predictions(test[1])
        # correct label clabel is the same as top1 predicted label plabel
        idx = 0
        for clabel in test[1]:
            label = np.argmax(clabel)
            if i >= nb_samples:
                print('Wrote {} samples to {}'.format(nb_samples, tests_pkl))
                with open(tests_pkl, 'wb') as testsf:
                    pickle.dump(tests, testsf)
                break
            tests.append([label, img_batch[idx]])
            print('added to test set image {} label {}'.format(i, label))
            idx += 1
            i += 1


for t in tests:
    X_test.append(t[1])
    y_test.append(t[0])
X_test = np.array(X_test)
y_test= np.array(y_test)

n_repeat = 10
N=2
p_c=10**-15
T=20
alpha=1-10**-2
epsilon_range = [0.015,0.03,0.06]
nb_examples = 5
if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv,"n:N:T:p:a:e:",["n_repeat=","p_c=","T=","N=","alpha=","epsilon_range=","epsilon=", "nb_examples="])

    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        
        sys.exit(2)
    for opt, arg in opts:
        print(f"opt:{opt}, arg:{arg}")
        if opt in ('-n', "--n_repeat"):
          n_repeat= int(arg)
        elif opt in ("-N", "--N"):
            N = int(arg)
        elif opt in ("-a", "--alpha"):
            alpha= arg
        elif opt in ("-T", "--T"):
            T= int(arg)
        elif opt in ("-p", "--p_c"):
            p_c= float(arg)
        elif opt=="--epsilon_range":
            epsilon_range = [float(e) for e in  arg.strip('[').strip(']').split(',')]
        elif opt in ("--epsilon","-e"):
            epsilon_range = [float(arg)]
        elif opt=="--nb_examples":
            nb_examples=int(arg)


name_method = f"Last Particle|N={N}|p_c={p_c}|T={T}"
print(f"Starting simulation with {name_method} on ImageNet.")
label_pred = lambda logits:np.argmax(logits,1)

net_eps_img_results = []
count = 1

TOTAL_RUN = n_repeat*len(epsilon_range)*2
avg_=0
i=0
list_models = [ResNet50, MobileNet]
model_names= ["ResNet50","MobileNet"]
preprocessors = [resnet_preprocess, mobilenet_preprocess]

for i in range(len(list_models)):
    model = list_models[i](weights='imagenet')
    x_test_n = np.array(X_test[:100], copy=True)
    x_test_n = preprocessors[i](x_test_n)
    model_name = model_names[i]
    t=time()
    

    logits = model.predict(x_test_n)
    def model_infer(X, by_batches=True, batch_size=None):
        w = np.array(X, copy=True)
        preds= model.predict(preprocessors[i](w))
        del w
        return preds
    t1 = time()-t
    
    y_pred_test = np.argmax(logits,1)
    
    test_correct = y_pred_test==y_test[:100]
    print(f'Accuracy of model {model_names[i]} on test data:{test_correct.mean()}')
    true_indices = np.where(y_pred_test==y_test[:100])[0]
    
    nb_systems = nb_examples
    big_batch_size= 128
    x_test, Y_test = X_test[true_indices][:nb_examples], y_test[true_indices][:nb_examples]
    for j in range(len(epsilon_range)):
        epsilon = epsilon_range[j]
        input_transform= lambda X,idx: input_transformer_gaussian(X,X_original = x_test[idx],epsilon=epsilon)
        total_score = lambda X,idx: nn_score(input_transform(X,idx), Y_test[idx], model_infer, several_labels=True, clipping_in=False)
    
        input_transform_big = lambda X: input_transformer_gaussian(X,X_original = np.repeat(x_test,N,axis=0), epsilon=epsilon)

        total_score_big = lambda X: nn_score(input_transform_big(X), np.repeat(Y_test,N),model=model_infer, several_labels=True, clipping_in=False)
        aggr_results = []
        for k in range(n_repeat):
            count+=1
            print(f'Run {k} on network {model_name} for epsilon={epsilon} started... ')
            t0= time()
            p_est, s_out = ImportanceSplittingLpBatch(gen =gaussian_gen, nb_system=nb_systems, s=1.5,d=DIM, kernel_b = batch_normal_kernel,h_big = total_score_big, h=total_score,N=N, tau=0, p_c=p_c, T=T,
    alpha_test = alpha, verbose = 0,check_every=5, accept_ratio=0.25,  reject_thresh = 0.99, reject_forget_rate =0.9, gain_forget_rate=0.9, fast_d=3)
            t1=time()-t0
            local_result = [t1, s_out['Cert'], s_out['Calls'],p_est]
            aggr_results.append(local_result)
            avg_ = (count-1)/count*avg_ + (1/count)*t1
            print(f'Run {k} on network {model_name} for epsilon={epsilon} finished... \n (RUN {count}/{TOTAL_RUN}, Avg. time per run:{avg_}) ')
            eta_ = ((TOTAL_RUN-count)*avg_)/3600
            hours = int(eta_)
            minutes = (eta_*60) % 60
            seconds = (eta_*3600) % 60

            print("ETA %d:%02d.%02d" % (hours, minutes, seconds))
            


        aggr_results = np.array(aggr_results)
        p_estimates = np.vstack(aggr_results[:,-1])
        cert_ = np.vstack(aggr_results[:,1])
        avg_compute_time, avg_calls = aggr_results[:,0].mean(), int(aggr_results[:,2].mean())
        std_compute_time,  std_calls = aggr_results[:,0].std(), aggr_results[:,2].std()
        avg_pest = p_estimates.mean(axis=0)
        std_pest = p_estimates.std(axis=0)
        avg_cert = cert_.mean(axis =0) 
        CI = s_out['CI_est']
        for l in range(nb_examples):
            net_eps_img_result = {'network name': model_name, 'epsilon':epsilon, 'Image index (MNIST Test)':true_indices[l], 'Original label':Y_test[l] , 'Compute time':t1/nb_examples, 'Certified': bool(int(s_out['Cert'][l])), 'method': name_method, 'Calls': int(s_out['Calls']/nb_examples), 'P_est': p_est[l], 'CI': s_out['CI_est'][l], 'Std P_est':std_pest[l], 'Computie time std':std_compute_time ,'Calls std': std_calls, 'Avg. Compute time':avg_compute_time/nb_examples,'Avg. number of calls':ceil(avg_calls/nb_examples), 'Frequency Certified': avg_cert[l],'Avg. P_est':avg_pest[l]}
            net_eps_img_results.append(net_eps_img_result)
         

lp_imagenet_results = pd.DataFrame(net_eps_img_results)
save_file = LOG_DIR+f"LP_N_{N}_pc_{-int(np.log10(p_c))}_T_{T}_n_{n_repeat}_ImageNet_results.csv"
with open(save_file, "w+") as file:
     lp_imagenet_results.to_csv(file)





