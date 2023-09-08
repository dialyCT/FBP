import numpy as np
from phantominator import shepp_logan
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def get_H(dNum,dSize):
    H=np.zeros(dNum)
    for i in range(0,dNum):
        if i==0:
            H[i]=1/(4*dSize**2)
        elif (i)%2==0:
            H[i]=0
        else:
            H[i]=-1/((i)*np.pi*dSize)**2
    return H

# 设置参数--------------------------------------------------------------
dNum = 512 #探测器个数
dSize = 7.4*1e-3 #探测器大小mm
pNum = 256 #重建图像像素个数
views = 360 #采样次数
sod = 600 #光源到旋转中心距离
sdd = 1000 #光源到探测器距离
odd = sdd - sod#旋装中心到探测器距离
da =  2*np.pi / views #采样间隔
l = dNum * dSize / 2 #探测器长度一半
R = sod * l / np.sqrt(l**2 + (sdd)**2) #视野圆半径
pSize = R * 2 / pNum #像素尺寸
dx = 0.5*pSize #一个像素取两个采样点，采样间隔
#detX和detY是大小为（n,1）的数组，代表探测器的横坐标和纵坐标_____________________________
detX = np.linspace(-l+dSize/2,  l-dSize/2,dNum,endpoint=True) #生成探测器X坐标，以旋转中心为原点
detX = detX.reshape((-1,1))
detY = np.zeros_like(detX)+odd
#生成图像坐标————————————————————————————————————————————————————————————————————————————————
temp = np.linspace(-R+pSize/2,R-pSize/2,pNum) #生成图像的X坐标和Y坐标，以旋转中心为原点
[imgX, imgY] = np.meshgrid(temp, temp)
#计算采样点的坐标（以光源为中心推导出的公式，所以y坐标需要做一下变换）
int = np.linspace(-R, R,int(2*R/dx))
sinPerDet = detX / np.sqrt(detX**2 + (sdd)**2)
cosPerDet = sdd / np.sqrt(detX**2 + (sdd)**2)
intX =sinPerDet*(int+sod)
intY =cosPerDet*(int+sod)
intY =intY-sod
#建立模体--------------------------------------------------------------
img = shepp_logan(pNum)
#Projection --------------------------------------------------------------
proj = np.zeros((dNum, views))
#为了实现函数的要求，要修改一下模式
imgX1=imgX.reshape((-1,1))
imgY1=imgY.reshape((-1,1))
img1=img.reshape((pNum*pNum,))

points=[]
for i in range(imgX1.shape[0]):
     points.append((imgX1[i,0],imgY1[i,0]))
points=np.array(points)
#射线驱动作正投
# for k in range(views):
#     print(k)#看看运行到哪，运行有点慢，耐心等待
#     phi = da * k
#     rotx = np.cos(phi)*intX - np.sin(phi)*intY#旋转一次计算一次坐标
#     roty = np.sin(phi)*intX + np.cos(phi)*intY
#     intV= griddata(points, img1, (rotx, roty), method='linear',fill_value=0)#插值得到积分点的像素值，超过范围补零
#     #intV= griddata_d(img,rotx, roty,imgY[0,0],imgY[0,0]-imgY[1,0])
#     #print(np.all(intV==intV1))
#     proj[:, k] = np.sum(intV,1)*dx#每条线求积分得到投影值
# proj.tofile("Fan_FBP_proj_data.raw")
# plt.imshow(proj,cmap='gray')
# plt.show()
proj=np.fromfile("Fan_FBP_proj_data.raw",np.double)
proj=proj.reshape((dNum,views))
proj=proj.T

#对扇束投影操作______________________________________
xproj=np.zeros((views,dNum))
#设置滤波器
H1 = get_H(dNum, dSize)
H2=H1[1:]
H2=H2[::-1]
f_RL=np.hstack((H1,H2))
f_RL=np.fft.fft(f_RL)
f_RL=np.real(f_RL)

for view in range(views):
    #对投影函数修正
    tmp=proj[view,:]*(sdd/np.sqrt(sdd**2+detX.T**2))
    tmp=np.hstack((tmp.reshape((-1,)),np.zeros(dNum-1,)))
    #与滤波函数卷积
    fft_tmp_proj = np.fft.fft(tmp)
    # 乘上滤波器
    filter_result = fft_tmp_proj * f_RL
    ifft_filter_result = np.fft.ifft(filter_result) * dSize  # dSize/2是重建公式离散化的呈现
    xproj[view, :] = np.real(ifft_filter_result[0:dNum])

#反投影————————————————————————————————————————————————————————————
rec=np.zeros((pNum,pNum))
for view in range(views):
    theta = da * view
    delta=xproj[view,:]
    rotx = np.cos(theta) * imgX + np.sin(theta) * imgY
    roty = -np.sin(theta) * imgX + np.cos(theta) * imgY
    uu = sdd * rotx / (roty + sod)  # 获得像素对应探测器的坐标

    xz = detX.shape[0]
    deltaImg = np.interp(uu, detX.reshape((xz,)), delta.reshape((xz,)), 0, 0)  # 获得所要修正的大小
    deltaImg=0.5*(sod*sdd/(sod-np.cos(theta) * imgX + np.sin(theta) * imgY)**2)*deltaImg*da
    rec = rec + deltaImg
rec[rec<0]=0
plt.imshow(rec,cmap='gray')
plt.show()