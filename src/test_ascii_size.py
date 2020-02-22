from match_ascii import *
from matplotlib import pyplot as plt

alphabet = ['#','@','%','=','*',':','-','.',' '] # given by the professor

img_list = ['i-2-a', 'i-2-b', 'i-2-c', 'i-2-d', 'i-2-e']

sizes = [50, 90, 150, 210]

nst = [s.FILTER_GAUSSIAN, s.FILTER_MEDIAN, s.FILTER_BILINEAR_INTERPOLATION]

# sizes experiment: for this experiment we only are considering alph_3 
# for each image
print('Size experiments:')
char = 97
for img_name in img_list:
    img_path = '../input/ascii_art/' + img_name + '.jpg'
    img = load_img_np(img_path)
    print('processing:', img_path)
    k = 0 
    for size in sizes:
        #for q in range(len(alphabets)):
        l = 97
        for ts in range(len(nst)):
            img_color, img_gray = Asciify(img, alphabet, size, nst[ts])
            cv2.imwrite('../output/size/o-4-'+ chr(char) + '-' + str(k) + '-' + chr(l) + '-0.png', img_color)
            cv2.imwrite('../output/size/o-4-'+ chr(char) + '-' + str(k) + '-' + chr(l) + '-1.png', img_gray)
            l += 1
        k += 1
    char += 1
