import numpy as np
import scipy.stats as stat
import tensorflow as tf



def nn_score(X, original_label, model, clipping_in=True ,clipping_out=False, mean=None, std=None, to_torch=False, 
several_labels=False, by_batches=False, batch_s=32):
    if clipping_in:
        a,b = 0,1
        if mean is not None and mean!=0:
            a,b = (a-mean), (b-mean)
        if std is not None and std!=0:
            a,b = a/std, b/std
        X = np.clip(X, a_min=a,a_max=b)    

        
    if len(X.shape)>1:
        logits = model(X, by_batches=by_batches, batch_size=batch_s)
    else:
        logits = model(np.array([X]))
    if to_torch:
        logits = logits.cpu().detach().numpy()
    if not several_labels:
        score = np.max(np.delete(logits,[original_label], axis=1), axis =1)- logits[:,original_label]
    else:
        original_labels = original_label
        batch_size = X.shape[0]
        logits_original = logits[np.arange(batch_size),original_labels]
        logits[np.arange(batch_size),original_labels] = -10e15
        score = np.max(logits,axis=1)-logits_original
    if clipping_out:
        score = np.clip(score, a_min=-np.inf, a_max =0)

    return score



def graph_model_converter(model, variable_scope=None):
    sess = tf.compat.v1.get_default_session()
    graph = model.graph
    output_tensor = graph.get_tensor_by_name([model.op.name][0] + ':0')
    input_name = 'x:0'
    if variable_scope is not None:
        input_name = variable_scope+'/'+input_name
    def model_out(x, batch_size =32, by_batches=False,verbose=0):
        if not by_batches:
            if len(x.shape)>1:
                logits = sess.run(output_tensor, feed_dict = {input_name: x})
            else:
                logits = sess.run(output_tensor, feed_dict = {input_name: x.reshape((1,)+x.shape)})
        else:
            logits_ = []
            nb_batches=  x.shape[0]//batch_size + 1 if (x.shape[0]%batch_size)!=0 else x.shape[0]//batch_size
            if verbose>=1:
                 print(f"nb_batches={nb_batches}")
            for j in range(nb_batches):
                logits = sess.run(output_tensor, feed_dict = {input_name: x[j*batch_size:(j+1)*batch_size]})
                logits_.append(logits)
            logits = np.concatenate(logits_,0)
        return logits
    
    return model_out


def input_transformer_eran(X, X_original, mean=0.0, std=0.0, eps=0.1,from_gaussian=True, clipping=True,norm='L_inf', quantization = False):
    if from_gaussian:
        if norm=='L_inf':
            X=eps*(2*stat.norm.cdf(X)-1)
            if std is not None and std!=0:
                X=X/std
        else:
            raise RuntimeError("This norm is not supported yet, please choose a norm among: \n {'L_inf'}")
    else:
        raise RuntimeError("This distribution is not supported yet, please choose a norm among: \n {'Gaussian'}")
    
    if len(X_original.shape)<len(X.shape):
        res = X+X_original[None,:]
    else:
        res = X+X_original
    if clipping:
        a,b = 0,1
        if mean is not None and std!=0:
            a,b = (a-mean)/std, (b-mean)/std
        res = np.clip(res, a_min=a,a_max=b)    
    if quantization==True and std==0:
        res = np.floor(res*255)/255
    return res


def input_transformer_gaussian(X,X_original,epsilon):
    X = 255*(2*stat.norm.cdf(X)-1)
    imgs = np.clip(X_original + epsilon*X.reshape(X_original.shape),a_min=0, a_max=255)
    return imgs




def dichotomic_search(f, a, b, thresh=0, n_max =50):
    """Implementation of dichotomic search of minimum solution for an increasing function
        Args:
            -f: decreasing function
            -a: lower bound of search spaceq
            -b: upper bound of search space
            -thresh: threshold such that if f(x)>0, x is considered to be a solution of the problem
    
    """
    low = a
    high = b
     
    i=0
    while i<n_max:
        i+=1
        if f(low)>=thresh:
            return low, f(low)
        mid = 0.5*(low+high)
        if f(mid)>thresh:
            high=mid
        else:
            low=mid

    return high, f(high)



gaussian_to_sym_uniform = lambda X: 2*stat.norm.cdf(X)-1 

normal_kernel =  lambda x,s : (x + s*np.random.normal(size = x.shape))/np.sqrt(1+s**2)

def batch_normal_kernel(X,s_batch):
    tilde_X = np.random.normal(size = X.shape)
    s_batch = s_batch.reshape(-1)
    X = (X+s_batch[:,None]*tilde_X)/np.sqrt(1+s_batch[:,None]**2)
    return X

