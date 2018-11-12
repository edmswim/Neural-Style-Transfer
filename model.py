import numpy as np
from PIL import Image
import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

tf.enable_eager_execution()

content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)









def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # VGG networks are trained on image with each channel normalized by mean = [103.939, 116.779, 123.68]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # need to reverse the channels since VGG networks are trained on images with channels BGR
    x = x[:, :, ::-1]

    # make sure rgb values are between [0,255]
    x = np.clip(x, 0, 255).astype('uint8')
    return x







def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    length = max(img.size)
    scale = max_dim/float(length)

    # Image.ANTIALIAS (a high-quality downsampling filter)
    img = img.resize((int(round(img.size[0]*scale)), int(round(img.size[1]*scale))), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension 
    img = np.expand_dims(img, axis=0)
    return img




def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    # shape of this is (5,)
    style_outputs = [vgg.get_layer(name).output for name in style_layers]

    # shape of this is (1,)
    content_outputs = [vgg.get_layer(name).output for name in content_layers]

    # shape of this is (6,)
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)



def get_feature_representations(model, content_path, style_path):
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # num_style_layers: because that is associated with style in get_model()
    # length is 5
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
   
    # :num_style_layers because that is associated with content in get_model()
    # length is 1
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def create_gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])

    # this converts the (3,1) tensor into a (2,) tensor
    # the first input of a will be the product of first two input from input_tensor
    a = tf.reshape(input_tensor, [-1, channels])

    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_content_loss(curr_content, target):
    return tf.reduce_mean(tf.square(curr_content - target))


def get_style_loss(curr_style, gram_target):
    gram_curr_style = create_gram_matrix(curr_style)
    return tf.reduce_mean(tf.square(gram_curr_style - gram_target))



def compute_loss(model, loss_weights, curr_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights

    # this curr_image will be constantly updated
    model_curr_img_outputs = model(curr_image)

    curr_img_style_output_features = model_curr_img_outputs[:num_style_layers]
    curr_img_content_output_features = model_curr_img_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, curr_style in zip(gram_style_features, curr_img_style_output_features):
        style_score += weight_per_style_layer * get_style_loss(curr_style[0], target_style)

    # Accumulate content losses from all layers 
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, curr_content in zip(content_features, curr_img_content_output_features):
        content_score += weight_per_content_layer * get_content_loss(curr_content[0], target_content)

    # multiply score by weights
    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score 
    return loss, style_score, content_score


def compute_gradient(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]

    #.compute gradient of loss function with respect to image for backward pass
    return tape.gradient(total_loss, cfg['curr_image']), all_loss


def neural_style_transfer(content_path, 
                       style_path,
                       num_iterations,
                       content_weight=1e3, 
                       style_weight=1e-2): 

    model = get_model() 
    # don't need to train any layers of the model
    for layer in model.layers:
        layer.trainable = False
  
    # Get the style and content feature representations (from our specified intermediate layers) 
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [create_gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img(content_path)
    init_image = tfe.Variable(init_image, dtype=tf.float32)

    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    # Store our best result
    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'curr_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    global_start = time.time()

    # VGG networks are trained on image with each channel normalized by mean = [103.939, 116.779, 123.68]
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   

    for i in range(num_iterations):
        grads, all_loss = compute_gradient(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])

        # clips tensor values to a specified min and max
        # Any values less than clip_value_min are set to clip_value_min. 
        # Any values greater than clip_value_max are set to clip_value_max
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        print("Iterations: {}".format(i))
        
        if loss < best_loss:
            # Update best loss and best image from total loss. 
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())
   
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    return best_img, best_loss 


content_path = 'cornell.jpg'
style_path = 'van_gogh_self_portrait.jpg'

final_output, best_loss = neural_style_transfer(content_path, style_path, num_iterations=1000)
Image.fromarray(final_output).save("cornell_output_pointillism.jpeg")







