import torch.nn as nn
import cv2

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        #self.requires_grad = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):

        return self.dwt_init(x)

    def dwt_init(self,x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        ll = self.avg_pool(x_LL)
        hl = self.avg_pool(x_HL)
        lh = self.avg_pool(x_LH)
        hh = self.avg_pool(x_HH)

        out=ll+hl+lh+hh

        return out




if __name__ == "__main__":
    img="C:\\Users\\Administrator\\Desktop\\jinyou11.jpg"
    x = cv2.imread(img)
    # haar = pywt.dwt2(x, 'haar')
    # LL, (LH, HL, HH) = haar
    # bior = pywt.dwt2(x, 'bior2.2')
    # coif = pywt.dwt2(x, 'coif3')
    # sym = pywt.dwt2(x, 'sym3')
    #print(haar)
    # AH = np.concatenate([LL, LH], axis=1)
    # VD = np.concatenate([HL, HH], axis=1)
    # img = np.concatenate([AH, VD], axis=0)
    # plt.imshow(img)
    # plt.title('img')
    # plt.show()
    #print(dwt(x))
    #print(coif)
    #print(sym)

    #re_raw_img = pywt.idwt2(cA, 'haar')
    # a = pywt.wavedec2(x, 'haar', level=11)
    # w = pywt.wavedec2(x, 'haar', level=10)
    # c = pywt.wavedec2(x, 'haar', level=9)
    # d = pywt.wavedec2(x, 'haar', level=8)
    # f = pywt.wavedec2(x, 'haar', level=7)
    # g = pywt.wavedec2(x, 'haar', level=6)
    # h = pywt.wavedec2(x, 'haar', level=5)
    # j = pywt.wavedec2(x, 'haar', level=4)
    # k = pywt.wavedec2(x, 'haar', level=3)
    # l = pywt.wavedec2(x, 'haar', level=2)
    # p = pywt.wavedec2(x, 'haar', level=1)
    # #print(x)
    # print(x.shape)
    # #print(c[0])
    # print(a[0].shape)
    # print(w[0].shape)
    # print(c[0].shape)
    # print(d[0].shape)
    # print(f[0].shape)
    # print(g[0].shape)
    # print(h[0].shape)
    # print(j[0].shape)
    # print(k[0].shape)
    # print(l[0].shape)
    # print(p[0].shape)




    # def cosine(image1, image2):
    #     X = np.vstack([image1, image2])
    #     return pdist(X, 'cosine')[0]
    #
    #
    # image1 = Image.open('image/1.jpg')
    # image2 = Image.open('image/2.jpg')
    # image2 = image2.resize(image1.size)
    # image1 = np.asarray(image1).flatten()
    # image2 = np.asarray(image2).flatten()
    #
    # print(cosine(image1, image2))

    #print(c)
    # plt.imshow(cA, 'gray')
    # plt.title('img')
    # plt.show()
    # cv2.imshow("re_raw_img", cA.astype(np.uint8))
    # cv2.waitKey(0)