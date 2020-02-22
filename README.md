# ASCIIArt

This project was part of the course MO446 Computer Vision at University of Campinas.
Detailed information(related state of the art, algorithms explanation, experiments, results and discussion) about this project can be found at [report](report_ASCIIArt.pdf).

### Abstract from report
An ASCII art can be described as a set of characters, defined by the ASCII standard, arranged as a bidimensional matrix that aims to reproduce an image. ASCII Art stemmed from the lack of graphics ability from early printers, and nowadays it is commonly used to represent pseudo gray-scale images in text-based messages. In the present work, we describe an ASCII Art pipeline to obtain an effective representation of an input image in grayscale and color using only characters of the ASCII standard. This pipeline is comprised of a procedure of either image upsampling or downsampling as required, which is carried out with our own implementation of image convolution using several filters like the median filter. Afterward, each pixel of the resultant image is compared with ASCII characters based on its density in order to find the most effective match. This procedure is performed aiming to obtain the best representation of the original image.
