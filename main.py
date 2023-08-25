import numpy as np
import tensorflow as tf
import sys
import tqdm

# Image preprocessing / deprocessing utilities

def preprocess_image(image_path, img_nrows, img_ncols):
    """
    Preprocesses image for EfficientNetB1
    """
    img = tf.keras.preprocessing.image.load_img(image_path,
                                                target_size=(img_nrows, img_ncols))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)

    return tf.convert_to_tensor(img)

def deprocess_image(tensor, img_nrows, img_ncols):
    """
    Converts tensor to image

    Since preprocess_input from EfficientNetB1 is just a pass through function,
    we don't need to do anything special here
    """
    tensor = tensor.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    tensor[:, :, 0] += 103.939
    tensor[:, :, 1] += 116.779
    tensor[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    tensor = tensor[:, :, ::-1]
    tensor = np.clip(tensor, 0.0, 255.0).astype('uint8')

    return tensor

# Util function to compute the style transfer loss

def gram_matrix(tensor):
    """
    Computes the Gram matrix of the input tensor
    """
    # (h, w, c) -> (c, h, w)
    tensor = tf.transpose(tensor, (2, 0, 1))
    # (c, h, w) -> (c, h * w)
    features = tf.reshape(tensor, (tf.shape(tensor)[0], -1))
    # (c, h * w) x (h * w, c) -> (c, c)
    gram = tf.matmul(features, tf.transpose(features))

    return gram

def style_loss(style, combination, img_nrows, img_ncols):
    """
    Computes the style loss
    """
    # gram matrices
    S = gram_matrix(style)
    C = gram_matrix(combination)

    # channels and size
    channels = 3
    size = img_nrows * img_ncols

    # calculating the style loss
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

def content_loss(base, combination):
    """
    Computes the content loss
    """
    # calculating the content loss
    return tf.reduce_sum(tf.square(combination - base))

def total_variation_loss(tensor, img_nrows, img_ncols):
    """
    The function calculates the components of the total variation loss.
    It calculates the squared differences between neighboring pixels in the
    horizontal and vertical directions.
    These differences are calculated for each color channel of the input tensor.
    """

    # The tensor a represents the squared differences between pixels in adjacent rows
    #, except for the last row.
    a = tf.square(
        tensor[:, :img_nrows - 1, :img_ncols - 1, :] - tensor[:, 1:, :img_ncols - 1, :])

    # The tensor b represents the squared differences between pixels in adjacent columns
    #, except for the last column.
    b = tf.square(
        tensor[:, :img_nrows - 1, :img_ncols - 1, :] - tensor[:, :img_nrows - 1, 1:, :])

    # This line computes the actual total variation loss by summing up the powered
    #added tensors a and b
    return tf.reduce_sum(tf.pow(a + b, 1.25))


def main(base_image_path_input, style_image_path_input, destination_path_input):

    try:
        base_image_path = base_image_path_input
        base_image = tf.keras.preprocessing.image.load_img(base_image_path_input)

        style_image_path = style_image_path_input
        style_image = tf.keras.preprocessing.image.load_img(style_image_path)

        destination_path = destination_path_input
    except:
        print("Invalid image path")
        return
    
    # weights of the loss components
    variation_weight = 1e-6
    style_weight = 1e-6
    content_weight = 2.5e-8

    # setting the dimensions of the generated picture
    width, height = base_image.size
    img_nrows = 400
    # protecting aspect ratio
    img_ncols = int(width * (img_nrows / height))

    # building the model

    model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # getting the outputs of intermediate layers
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # setting up a model that returns the activation values for every layer in
    # EfficientNetB1 given an input image
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=outputs_dict)

    # choosing the layers that will be used for content and style loss
    
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]

    content_layer = "block5_conv2"

    def compute_loss(combination_image, base_image, style_reference_image):
        """
        This function calculates the total loss for a neural style transfer optimization.
        It combines content loss, style loss, and total variation loss to form a single
        loss value that guides the optimization process.
        """
        # concatenating the base image, style image and combination image
        input_tensor = tf.concat(
            [base_image, style_reference_image, combination_image], axis=0
        )
        # getting the outputs of intermediate layers
        features = feature_extractor(input_tensor)

        # initializing the loss
        loss = tf.zeros(shape=())

        # adding the content loss
        layer_features = features[content_layer]

        # base image features are the first features
        base_image_features = layer_features[0, :, :, :]
        # combination image features are the third features
        combination_features = layer_features[2, :, :, :]

        # calculating the content loss
        loss = loss + content_weight * content_loss(
            base_image_features, combination_features
        )

        # adding the style loss
        for layer_name in style_layers:
            # getting the style features
            layer_features = features[layer_name]

            # style reference features are the second features
            style_reference_features = layer_features[1, :, :, :]
            # combination features are the third features
            combination_features = layer_features[2, :, :, :]

            # calculating the style loss
            sl = style_loss(style_reference_features, combination_features, img_nrows, img_ncols)
            # adding the style loss
            loss += (style_weight / len(style_layers)) * sl

        # adding the total variation loss
        loss += variation_weight * total_variation_loss(combination_image, img_nrows, img_ncols)

        return loss
    
    @tf.function
    def compute_loss_and_grads(combination_image, base_image, style_reference_image):
        # using GradientTape to update the gradients
        with tf.GradientTape() as tape:
            # calculating the loss
            loss = compute_loss(combination_image, base_image, style_reference_image)
        # calculating the gradients
        grads = tape.gradient(loss, combination_image)

        return loss, grads
    
    # the style transfer loop

    # defining Adam optimizer with exponential learning rate decay of 0.9
    optimizer = tf.keras.optimizers.SGD(
        tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=200.0, decay_steps=100, decay_rate=0.96
        )
    )

    # preparing the base image
    base_image = preprocess_image(base_image_path, img_nrows, img_ncols)
    # preparing the style image
    style_reference_image = preprocess_image(style_image_path, img_nrows, img_ncols)
    # initializing the combination image as a tensor version of base image
    combination_image = tf.Variable(preprocess_image(base_image_path, img_nrows, img_ncols))

    # number of iterations
    iterations = 1000

    # type of the style image
    OUT = "results"
    STYLE = "street"
    BASE = "yl"

    for i in tqdm.tqdm(range(1, iterations+1), desc="Style Transfer Progress"):
        # computing the loss and gradients
        loss, grads = compute_loss_and_grads(
            combination_image, base_image, style_reference_image
        )
        # applying the gradients to the combination image
        optimizer.apply_gradients([(grads, combination_image)])

        if i % 10 == 0:
            print(f"Iteration {i}: loss={loss:.2f}")

    # saving the current combination image
    img = deprocess_image(combination_image.numpy(), img_nrows, img_ncols)
    fname = destination_path
    print(f"Image saved at {fname}")
    tf.keras.preprocessing.image.save_img(fname, img)


# get the base, style and destination image paths from the command line
if __name__ == "__main__":
    base_image_path_input = sys.argv[1]
    style_image_path_input = sys.argv[2]
    destination_path_input = sys.argv[3]
    main(base_image_path_input, style_image_path_input, destination_path_input)