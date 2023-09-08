import numpy as np
from scipy.interpolate import griddata
from phantominator import shepp_logan
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

#填写数据信息————————————————————————————————————————————————————————————————————————————————
proj_path=r"D:\study\相衬CT\兔头数据结果\after_proj_data.raw"
proj_type=np.double
offset=-17
dNum=2048#探测器个数
dSize=7.4*1e-3#探测器尺寸/mm
pNum=1024#像素个数
pSize=dNum*dSize/pNum#像素尺寸
L=dNum*dSize #探测器长度
views=1800#采样次数，应该足够多
da=np.pi/views #采样间隔
#生成图片像素点的x和y坐标
img_temp=np.linspace(-pNum*pSize/2+pSize/2,pNum*pSize/2-pSize/2,pNum,endpoint=True)
[imgX,imgY]=np.meshgrid(img_temp,img_temp)
#生成采样点的坐标
det_temp=np.linspace(-L/2+dSize/2,L/2-dSize/2,dNum,endpoint=True)
[pointX,pointY]=np.meshgrid(det_temp,det_temp)
#正投影，只运行一次存储数据就行，第二次直接load
proj=np.zeros((views,dNum))
imgX1=imgX.reshape((-1,1))
imgY1=imgY.reshape((-1,1))
points=[]
for i in range(imgX1.shape[0]):
     points.append((imgX1[i,0],imgY1[i,0]))
points=np.array(points)
#导入投影数据——————————————————————————————————————————————————————————————————————
proj=np.fromfile(proj_path,proj_type)
proj=proj.reshape((views,dNum))
rec=np.zeros((pNum,pNum))
iproj=np.zeros((views,dNum))
for view in range(views):
    # 给投影数据补零，最简单补N-1个
    tmp_proj=np.hstack((proj[view,:],np.zeros(dNum-1,)))
    #构建滤波器，注意位置
    H1 = get_H(dNum, dSize)
    H2=H1[1:]
    H2=H2[::-1]
    f_RL=np.hstack((H1,H2))
    f_RL=np.fft.fft(f_RL)
    f_RL=np.real(f_RL)
    # 投影数据作傅里叶变换
    fft_tmp_proj = np.fft.fft(tmp_proj)
    # 乘上滤波器
    filter_result = fft_tmp_proj * f_RL
    ifft_filter_result = np.fft.ifft(filter_result)*dSize/2#dSize/2是重建公式离散化的呈现
    iproj[view, :] = np.real(ifft_filter_result[0:dNum])
#反投影，像素驱动
for view in range(views):
    print(view)
    phi=da*view
    rotx = np.cos(phi) * imgX + np.sin(phi) * imgY+offset*dSize
    rec += np.interp(rotx, det_temp,iproj[view,:], 0, 0)*da#da/2是重建公式离散化的呈现
rec[rec<0]=0
# rec.tofile("try.raw")
# plt.imshow(rec,cmap='gray')
# plt.show()

