The images are 3 channels because that is how the pix2pix model would output them.
You will have to convert it into a 1 channel image. Just take the average across the
3 channels should do.

I have provided a script that will convert the 3 channel image to a 1 channel image. There will be two functions:
1. convert_directory
    args: 
        source directory
        target directory
    description:
        Will convert all images in the source directory and save them in the target directory
2. convert_image
    args:
        a PIL Image object
    description will convert the pass in PIL Image object and convert it to a 1 channel image and return the result

DEPENDENCIES:
    PIL (pillow)