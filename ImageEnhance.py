import cv2
import numpy as np

kernel_gabor0 = np.array([[-0.7022, -0.7359, -0.7475, -0.7359, -0.7022],
                          [-0.8287, -0.8685, -0.8821, -0.8685, -0.8287],
                          [0.00000, 0.0000, 0.0000, 0.0000, 0.0000],
                          [0.8287, 0.8685, 0.8821, 0.8685, 0.8287],
                          [0.7022, 0.7359, 0.7475, 0.7359, 0.7022]], np.float32)
kernel_gabor45 = np.array([[-0.0000, -0.6540, -0.9394, -0.6540, -0.0000],
                           [-0.6540, -0.9692, -0.6961, -0.0000, 0.6540],
                           [-0.9394, -0.6961, 0.0000, 0.6961, 0.9394],
                           [-0.6540, 0.0000, 0.6961, 0.9692, 0.6540],
                           [0.0000, 0.6540, 0.9394, 0.6540, 0.0000]], np.float32)
kernel_gabor90 = np.array([[-0.7022, -0.8287, -0.0000, 0.8287, 0.7022],
                           [-0.7359, -0.8685, -0.0000, 0.8685, 0.7359],
                           [-0.7475, -0.8821, 0.0000, 0.8821, 0.7475],
                           [-0.7359, -0.8685, 0.0000, 0.8685, 0.7359],
                           [-0.7022, -0.8287, 0.0000, 0.8287, 0.7022]], np.float32)
kernel_gabor135 = np.array([[-0.0000, 0.6540, 0.9394, 0.6540, 0.0000],
                            [-0.6540, -0.0000, 0.6961, 0.9692, 0.6540],
                            [-0.9394, -0.6961, 0.0000, 0.6961, 0.9394],
                            [-0.6540, -0.9692, -0.6961, 0.0000, 0.6540],
                            [-0.0000, -0.6540, -0.9394, -0.6540, 0.0000]], np.float32)

def zmMinFilterGray(src, r=7):

    if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    rows_up = I[[0] + [x for x in range(h-1)], :]
    rows_down = I[[x for x in range(1,h)] + [h-1], :]
    res = np.minimum(I, rows_up)
    res = np.minimum(res, rows_down)
    I = res
    cols_left = I[:, [0] + [x for x in range(w-1)]]
    cols_right = I[:, [x for x in range(1, w)] + [w-1]]
    res = np.minimum(I, cols_left)
    res = np.minimum(res, cols_right)
    return zmMinFilterGray(res, r - 1)


def guidedfilter(I, p, r, eps):

    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):
    V1 = np.min(m, 2)
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)
    bins = 200
    ht = np.histogram(V1, bins)
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)

    return V1, A


def deHaze(m, r=6, eps=0.001, w=0.6, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)
    #cv2.imwrite('d:/Endehaze.jpg', V1*255)
    #v_cmp = cv2.imread('d:/TransMatSave.jpg',0);
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.78) / np.log(Y.mean()))
    return Y


def image_convert(img):
    img_temp = img
    img_temp[:, :, :] = 255 - img_temp[:, :, :]
    return img_temp

def image_sharpness(img):
    kernel_log = np.array([[0.1376, -0.0194, 0.1376], [-0.0194, -0.4728, -0.0194], [0.1376, -0.0194, 0.1376]],
                           np.float32)
    result = cv2.filter2D(img, -1, kernel=kernel_log)
    result_sharpen = img-1.10*result
    return result_sharpen

def LuEnhance(img):
    convertimg = image_convert(img)
    dehazeresult = deHaze(convertimg/255.0)*255
    img_result = image_convert(dehazeresult)
    return img_result

def PreProcessing(img):
    bresult = LuEnhance(img)
    fresult = image_sharpness(bresult)
    return  fresult

def TestSingle(img):
    enhanced = LuEnhance(img)
    return enhanced