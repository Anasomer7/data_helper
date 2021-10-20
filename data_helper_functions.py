import pandas as pd, numpy as np, random,os, shutil
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import sklearn
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import wandb
import yaml
from IPython import display as ipd

from glob import glob
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
class CFG:
    wandb         = True
    competition   = 'petfinder' 
    _wandb_kernel = 'awsaf49'
    debug         = False
    exp_name      ='effnetb4-baseline' # name of the experiment, folds will be grouped using 'exp_name'
    
    # USE verbose=0 for silent, vebose=1 for interactive, verbose=2 for commit
    verbose      = 1 if debug else 0
    display_plot = True

    device = "TPU" #or "GPU"

    model_name = 'efficientnet_b4'

    # USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
    seed = 42

    # NUMBER OF FOLDS. USE 2, 5, 10
    folds = 5
    
    # FOLDS TO TRAIN
    selected_folds = [0, 1, 2, 3, 4]

    # IMAGE SIZE
    img_size = [512, 512]

    # BATCH SIZE AND EPOCHS
    batch_size  = 32
    epochs      = 12

    # LOSS
    loss      = 'RMSE'
    optimizer = 'Adam'

    # CFG.augmentATION
    augment   = True
    transform = False

    # TRANSFORMATION
    fill_mode = 'nearest'
    rot    = 10.0
    shr    = 5.0
    hzoom  = 30.0
    wzoom  = 30.0
    hshift = 30.0
    wshift = 30.0

    # FLIP
    hflip = True
    vflip = False

    # CLIP [0, 1]
    clip = False

    # LEARNING RATE SCHEDULER
    scheduler   = 'exp' # Cosine

    # Dropout
    drop_prob   = 0.75
    drop_cnt    = 10
    drop_size   = 0.05

    #bri, contrast
    sat  = [0.7, 1.3]
    cont = [0.8, 1.2]
    bri  =  0.15
    hue  = 0.05

    # TEST TIME CFG.augmentATION STEPS
    tta = 1
    
    tab_cols    = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
                   'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
    target_col  = ['Pawpularity']
def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)
    tf.random.set_seed(SEED)
    print('seeding done!!!')
seeding(CFG.seed)

def build_decoder(with_labels=True, target_size=CFG.img_size, ext='jpg'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.reshape(img, [*target_size, 3])

        return img
    
def build_augmenter(with_labels=True, dim=CFG.img_size):
    def augment(img, dim=dim):
        img = tf.image.random_flip_left_right(img) if CFG.hflip else img
        img = tf.image.random_flip_up_down(img) if CFG.vflip else img
        img = tf.image.random_hue(img, CFG.hue)
        img = tf.image.random_saturation(img, CFG.sat[0], CFG.sat[1])
        img = tf.image.random_contrast(img, CFG.cont[0], CFG.cont[1])
        img = tf.image.random_brightness(img, CFG.bri)
        img = tf.clip_by_value(img, 0, 1)  if CFG.clip else img         
        img = tf.reshape(img, [*dim, 3])
        return img
    
    def augment_with_labels(img, label):    
        return augment(img), label
    
    return augment_with_labels if with_labels else augment

def build_dataset(paths, labels=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir="", drop_remainder=False):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)
    
    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTO)
    return ds

def display_batch(batch, size=2):
    imgs, tars = batch
    plt.figure(figsize=(size*5, 5))
    for img_idx in range(size):
        plt.subplot(1, size, img_idx+1)
        plt.title(f'{CFG.target_col[0]}: {tars[img_idx].numpy()[0]}', fontsize=15)
        plt.imshow(imgs[img_idx,:, :, :])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
    

def RMSE(y_true, y_pred):
    loss = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(tf.subtract(y_true, y_pred))))
    return loss

def get_lr_callback(batch_size=8, plot=False):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * REPLICAS * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        elif CFG.scheduler=='exp':
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        elif CFG.scheduler=='cosine':
            decay_total_epochs = CFG.epochs - lr_ramp_ep - lr_sus_ep + 3
            decay_epoch_index = epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            cosine_decay = 0.5 * (1 + math.cos(phase))
            lr = (lr_max - lr_min) * cosine_decay + lr_min
        return lr
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(np.arange(CFG.epochs), [lrfn(epoch) for epoch in np.arange(CFG.epochs)], marker='o')
        plt.xlabel('epoch'); plt.ylabel('learnig rate')
        plt.title('Learning Rate Scheduler')
        plt.show()

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

        
def gen_gradcam_heatmap(img, model, last_conv_layer_name='top_conv', pred_index=0):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer('top_conv').output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
    
def get_gradcam(img, model,  alpha=0.4, show=False):

    heatmap = gen_gradcam_heatmap(img, model, last_conv_layer_name='top_conv', pred_index=0)
    img     = img[0]
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors  = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = cv2.resize(jet_heatmap, dsize=(img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
#     superimposed_img = np.uint8(superimposed_img*255.0)

    # Display Grad CAM
    if show:
        plt.imshow(superimposed_img)
        
    return superimposed_img


def wandb_init(fold):
        config = {k:v for k,v in dict(vars(CFG)).items() if '__' not in k}
        yaml.dump(config, open(f'/kaggle/working/config fold-{fold}.yaml', 'w'),)
        config = yaml.load(open(f'/kaggle/working/config fold-{fold}.yaml', 'r'), Loader=yaml.FullLoader)
        run    = wandb.init(project="petfinder-public",
                   name=f"fold-{fold}|dim-{CFG.img_size[0]}|model-{CFG.model_name}",
                   config=config,
                   anonymous=anonymous,
                   group=CFG.exp_name
                        )
        return run

def log_wandb(fold):
    "log best result and grad-cam for error analysis"
    
    valid_df = df.loc[df.fold==fold].copy()
    valid_df['pred'] = oof_pred[fold].reshape(-1)
    valid_df['diff'] =  abs(valid_df.Pawpularity - valid_df.pred)
    valid_df    = valid_df[valid_df.fold == fold].reset_index(drop=True)
    vali_df     = valid_df.sort_values(by='diff', ascending=False)
    
    noimg_cols  = ['Id', 'fold', 'Subject Focus','Eyes','Face','Near','Action','Accessory','Group',
                    'Collage','Human','Occlusion','Info','Blur',
                   'Pawpularity', 'pred', 'diff']
    # select top and worst 10 cases
    gradcam_df  = pd.concat((valid_df.head(10), valid_df.tail(10)), axis=0)
    gradcam_ds  = build_dataset(gradcam_df.image_path, labels=None, cache=False, batch_size=1,
                   repeat=False, shuffle=False, augment=False)
    data = []
    for idx, img in enumerate(gradcam_ds):
        gradcam = get_gradcam(img, model)
        row = gradcam_df[noimg_cols].iloc[idx].tolist()
        data+=[[*row, wandb.Image(img.numpy()[0]), wandb.Image(gradcam)]]
    wandb_table = wandb.Table(data=data, columns=[*noimg_cols,'image', 'gradcam'])
    wandb.log({'best_rmse':oof_val[-1], 
               'best_rmse_tta':rmse,
               'best_epoch':np.argmin(history.history['val_rmse']),
               'viz_table':wandb_table})
    

        
    
    



        
        
    