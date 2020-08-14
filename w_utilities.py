'''
HR_Encoder extends Alberto Rosas Garcias unofficial StyleGAN2-Tensorflow-2.x implementation (https://github.com/rosasalberto/StyleGAN2-TensorFlow-2.x) to encode your own images into StyleGAN2 *W* space.

Legal information

HR_Encoder.ipynb, HR_Encoder.py and w_utilities.py are Copyright © 2020 HANS ROETTGER (mailto:oss.roettger@posteo.org)
and distributed under the terms of GNU AGPLv3 (https://www.gnu.org/licenses/agpl-3.0.html)

Severability clause: If a provision of the [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html) is or becomes illegal, invalid or unenforceable in any jurisdiction, that shall not affect: 
1. the validity or enforceability in that jurisdiction of any other provision of the GNU AGPLv3; or 
2. the validity or enforceability in other jurisdictions of that or any other provision of the GNU AGPLv3.

!!! WARNING !!!

HR_Encoder makes heavy use of your hardware via the tensorflow libraries. Pay attention to adequate ventilation/ cooling and take all necessary precautions to prevent overload, overheating or consequential damage according to the recommendations of your hardware provider.
'''
import math
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
import cv2
    
def toRGB(generator,w):
    if w is None:
        return np.ones((2,2,3),dtype=np.uint8)*255
    img=generator.synthesis_network(w.reshape((1,-1,512)))
    imgRGB=np.float32(np.asarray(img[0]).transpose(1,2,0))
    imgRGB=imgRGB.clip(-1.0,1.0)
    return np.uint8((imgRGB+1.0)*127.99)

def showDifference(generator,w,img,img_size=(5,5)):
    w_img=toRGB(generator,w)
    d_img=np.absolute(img*1.0-w_img*1.0)
    p_dif=np.sum(d_img)/(d_img.shape[0]*d_img.shape[1]*d_img.shape[2]*2.56)
    d_img=np.sqrt(d_img)
    scale=256.0/d_img.max()
    d_img=np.uint8(d_img*scale)
    w,h,c=d_img.shape
    w,h=float(w),float(h)
    figsize=(3*img_size[0]+2,img_size[1])
    fig, ax = plt.subplots(1,3,figsize=figsize)
    ax[0].imshow(d_img)
    ax[0].text(w/90.0, h/9.5, "\u0394≈ {0:5.2f} %".format(p_dif),size=24,color='black')
    ax[0].text(w/100.0, h/10.5, "\u0394= {0:5.2f} %".format(p_dif),size=24,color='white')
    ax[1].imshow(w_img)
    ax[1].text(w/90.0, h/9.5, "Encoded",size=24,color='black')
    ax[1].text(w/100.0, h/10.5, "Encoded",size=24,color='white')
    ax[2].imshow(img)
    ax[2].text(w/90.0, h/9.5, "Original",size=24,color='black')
    ax[2].text(w/100.0, h/10.5, "Original",size=24,color='white')
    ax[0].axis('off') 
    ax[1].axis('off') 
    ax[2].axis('off') 
    plt.show()
    
    
def showImages(generator,w_tape,img_size=(1,1)):
    try:
        w_tape[0][0][0][0]
        nrow,ncol=len(w_tape),len(w_tape[0])
    except:
        try:
            w_tape[0][0][0]
            nrow,ncol=1,len(w_tape)
            w_tape=[w_tape]
        except:
            nrow,ncol=1,1
            w_tape=[[w_tape]]
    
    figsize=(ncol*img_size[0],nrow*img_size[1])
    fig, ax = plt.subplots(nrow,ncol,figsize=figsize)
    if nrow>1 and ncol>1:
        for r,row in enumerate(w_tape):
            for c,w in enumerate(row):
                print("\rRendering images {0}".format(["/","-","\\","|"][(r+c)%4]),end="")
                ax[r,c].imshow(toRGB(generator,w))
                ax[r,c].axis('off')  
    elif nrow>1:
        for r,row in enumerate(w_tape):
            print("\rRendering images {0}".format(["/","-","\\","|"][r%4]),end="")
            ax[r].imshow(toRGB(generator,row[0]))
            ax[r].axis('off') 
    elif ncol>1:
        for c,w in enumerate(w_tape[0]):
            print("\rRendering images {0}".format(["/","-","\\","|"][c%4]),end="")
            ax[c].imshow(toRGB(generator,w))
            ax[c].axis('off')    
    else:
        ax.imshow(toRGB(generator,w_tape[0][0]))
        ax.axis('off') 
    print("\r"+" "*30,end='\r')
    plt.show()
    
def showVector(w):
    fig,ax=plt.subplots(1, 1, figsize=(16, 0.1*w.shape[0]), dpi=80)
    ax.imshow(w, cmap="seismic",aspect='auto')
    ax.axis('off')
    plt.show()
    
def random(generator,seed):
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, 512).astype('float32')
    w = generator.mapping_network(z)[0].numpy()  
    return w
    
def average(w_tape):
    w_sum=w_tape[0]
    for w in w_tape[1::]:
        w_sum+=w
    return w_sum/len(w_tape)
 
def mask(w,rows=[3,4,5,6,7,8,9,10,11,12,13,14]):
    r=w*0
    r[rows]=w[rows]
    return r    

def previewTape(w_tape,nframes):
    rat=np.array(np.linspace((1/(nframes+1)),1-(1/(nframes+1)),nframes))
    w_tape_p=[]
    for r in rat:
        w_tape_p.append(w_tape[int(r*len(w_tape))])
    return w_tape_p

def morphTapeSin(w1,w2,nframes,loop=False):
    rat=map(lambda x:(math.sin(x*math.pi+math.pi/2)+1)/2,np.array(np.linspace(0.0,1.0,nframes)))
    w_tape=[]
    for r in rat:
        w_tape.append(w1+(w2-w1)*r) 
    if loop:
        w_tape.extend(reversed(w_tape))
    return w_tape

def morphTape(w_tape,nframes=50,speed=4,epsilon=1.0):
    l=len(w_tape*2)-1
    if l<2:
        return [None]
    w_movie=[]
    v=np.zeros(len((w_tape[0].reshape(-1))),dtype=np.float32)
    
    p_f=np.copy(w_tape[0].reshape(-1))
    # for smooth closed loop: go through w_tape twice and record second round
    for i,pn in enumerate((w_tape*2)[1::]):
        pn_f=pn.reshape(-1)
        next_p=False
        while not next_p:
            n_direction=pn_f-p_f
            dist=np.linalg.norm(n_direction)
            next_p=(dist<epsilon)
            n_direction=n_direction/(dist*epsilon)
            v=(speed*v+n_direction)/(speed+1)
            p_f+=v
            if i>=int(l/2):
                w_movie.append(np.copy(p_f.reshape(-1,512)))
    return previewTape(w_movie,nframes)    
    
def speedupTape(ws,speed=1.0,acc=1.05):
    w_tape=[]  
    l=len(ws)
    f=0.0
    while int(f)<l:
        w_tape.append(ws[int(f)]) 
        f+=speed
        speed*=acc
    w_tape.append(ws[-1])
    return w_tape   

def tapeTable(w_tape,no_col=5):
    table,row,col=[],[],0
    fillup=[None]*int(no_col-len(w_tape)%no_col)
    if len(fillup)==no_col:
        fillup=[]
    for w in (w_tape+fillup):
        col+=1
        row.append(w)
        if col>=no_col:
            table.append(row)
            row,col=[],0       
    return table

def renderVideo(generator,w_tape,video_path,video_fps=25):
    l=len(w_tape)
    if l>0:
        h,w,c=toRGB(generator,w_tape[0]).shape        
        video_out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MP4V'), video_fps, (h,w))
        for f,w in enumerate(w_tape):
            print("\rRendering video frame {0}/{1}".format(f,l),end='    ')
            video_out.write(cv2.cvtColor(toRGB(generator,w),cv2.COLOR_BGR2RGB))
        video_out.release()
        print("\r"+" "*40,end='\r')
        
def showVideo(video_path,height=512,loop=1):
    clip = VideoFileClip(video_path)
    return clip.ipython_display(height=height,autoplay=1, loop=loop,rd_kwargs=dict(logger=None,verbose=True))

def videoGIF(video_path,gif_path,video_fps=25):
    clip = VideoFileClip(video_path)
    clip.write_gif(gif_path,program='ffmpeg',fps=video_fps,logger=None)