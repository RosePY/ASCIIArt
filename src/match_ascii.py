from utils import *
from sampling import Sampling as s

##Resize image just with the width###
def s_res_width(img, width, smooth_type):
    ratio_scale=(width*1.0)/img.shape[1]
    height=int(ratio_scale*img.shape[0])
    nimg = s.smooth_resize(img, height,width, smooth_type)
    return nimg


def RankingAsciiChars(alphabet):
    accumulator=[]
    frameSize=14
    for letter in alphabet:
        frame = np.ones((frameSize,frameSize))*255
        charSize=cv2.getTextSize(letter,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)[0]
        pos_x=(frameSize-charSize[0])//2
        pos_y=(frameSize+charSize[1])//2
        cv2.putText(frame, letter, (pos_x, pos_y-1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        #plt.imshow(frame, cmap='gray')
        #plt.show()
        accumulator.append((letter, np.sum(frame != 255), (pos_x,pos_y)))#np.count_nonzero(frame==0)))
    accumulator.sort(key=lambda tup: tup[1])
    return accumulator


def Asciify(img,alphabet_string,n_width,smooth_type,):
    fs=14
    alphabet = [i for i in alphabet_string]
    rankingAlphabet = RankingAsciiChars(alphabet)
    #Histogram equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if n_width != img.shape[1]:
        res_img = s_res_width(img, n_width, smooth_type)
        res_img_gray = s_res_width(img_gray, n_width, smooth_type)
    else:
        res_img = img
        res_img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x = res_img_gray.shape[0]
    y = res_img_gray.shape[1]
    new_img_gray = np.ones((x*fs,y*fs),np.uint8)*255
    new_img_color = np.zeros((x*fs,y*fs,img.shape[2]),np.uint8)
    rankingAlphabetgray = rankingAlphabet[::-1]
    
    interval_size=255//(len(alphabet))
    
    for i in range(x):
        for j in range(y):
            position=int(res_img_gray[i,j]//interval_size)
            if position >= (len(alphabet)):
                position=position-1
            chargray=rankingAlphabetgray[position][0]
            charcol=rankingAlphabet[position][0]
            cv2.putText(new_img_gray[i*fs:(i+1)*fs,j*fs:(j+1)*fs], chargray,rankingAlphabetgray[int(position)][2] , cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
            cv2.putText(new_img_color[i*fs:(i+1)*fs,j*fs:(j+1)*fs], charcol,rankingAlphabetgray[int(position)][2] , cv2.FONT_HERSHEY_SIMPLEX, 0.4,(int(res_img[i,j,0]),int(res_img[i,j,1]),int(res_img[i,j,2])), 1)
                          
    return new_img_gray,new_img_color
