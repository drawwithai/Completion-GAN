# Completion GAN

GAN model used to try to complete hand sketchs, using Tensorflow 2 

## Dependencies

- Tensorflow 2
- Numpy
- Mathplotlib
- IPython

## Usage

	$ main batch_size [images_folder]
	
    batch_size : 
        Number of images to take at once for training. 
        Must be tweaked according to available ram memory.
        Higher is better.
        
    images_folder :
        If given, it will train on images in given folder.
        Load the dataset oneline45 if not given.

## Tweak

Generator and Discriminator models are separated in their own files. Each has a set of tweaking constants one can modify. For now, discriminator seems good, but generator doesn't success to complete nor preserve images. It might be a problem with training datas or a problem with the model itself.

### Generator

It take three inputs of ( masked image, mask, random noise ) each of shape 256x256x1
And output an image with the same shape

The model consist of two facing funnels, with a kind of blender in the midle

The first part consist of cascading convolutions layers to compress image to a more abstract form.
The second part is a pipe of dilated convolution layers, the idea was to propagate global image features to prepare the reconstruction.
The last part is a cascade of deconvolution layers, converting abstract features to a new image.

Noise input should be moved further inside the model instead of with the image and the mask.

Loss function is a bit weird.
It's a linear combinaison of three losses :

- preceipt_loss : measure efficiency at fooling the discriminator. Computed with binarycrossentropy
- context_loss : try to measure if the non masked area is preserved. binarycrossentropy
- fillrate : try to punish the generator if he doesn't try to draw in the masked area. 

### Discriminator

Actualy, a simple convolution funnel followed by dense layers
It takes a 256x256x1 normalised image as input 
and output a single value, <0 if predicted fake, >0 if real

The loss function is binary crossentropy, summ of loss on real and fake images

## Continuation

As said before, this GAN isn't successfull, here are few clues to continue this project :

### Have a bigger dataset : 

For now, traning occurs with a ~200 images dataset. It's way to few...

### Change inputs layout : 

Generator's model takes noise as secondary input in order to obtain a new sketch each time, but this input happens at the same point as image and mask input, which is not optimal, model have to learn to separate image from the noise before it can learn how to complete sketches.

One should try to move theses inputs, for exemple, having one funnel which will reduce the masked image and the mask together to abstract features, then introduce some noise, mash features and put the growing funnel to generate final image.

### Change generator's layers :

As input and output funnels are deeply related, it could be usefull to change the simple convolution / deconvolution to a more advanced pooling / unpooling layout.

In a same way, blending stage's layers could be changed to something else than dilated convolutions.