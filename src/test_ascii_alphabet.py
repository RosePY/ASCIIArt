from match_ascii import *
from matplotlib import pyplot as plt

alph_1 = [',', '"', '!', '1', '<', '=', 'V', 'U', '5', 'R', 'B', 'W', '@'] # equal sample 13 chars
alph_2 = [' ','.', ',', '`', '!', '\\', '1', '<', '(', '=', 'V', '4', '2', 'E', '0', 'R', 'B', 'W', '&', '#', '@'] # equal sample 21 chars
alph_3 = ['#','@','%','=','*',':','-','.',' '] # given by the professor
alph_4 = ['C','O','M','P','U','T','E','R','V','I','S','N','F','o','c','i','m','p','.'] # personalized

img_list = ['i-2-a', 'i-2-b', 'i-2-c', 'i-2-d', 'i-2-e']

alphabets = [''.join(i) for i in [alph_1, alph_2, alph_3, alph_4]]

sizes = [50, 200]

# alphabets experiment
print('Alphabets experiments')
char = 97
for img_name in img_list:
    img_path = '../input/ascii_art/' + img_name + '.jpg'
    img = load_img_np(img_path)
    print('processing:', img_path)
    k = 0 
    for alphabet in alphabets:
        m = 97
        for size in sizes:
            img_color, img_gray = Asciify(img, alphabet, size, s.FILTER_GAUSSIAN)
            cv2.imwrite('../output/alphabet/o-4-'+ chr(char) + '-' + str(k) + '-' + chr(m) + '-0.png', img_color)
            cv2.imwrite('../output/alphabet/o-4-'+ chr(char) + '-' + str(k) + '-' + chr(m) + '-1.png', img_gray)
            m += 1
        k += 1
    char += 1
