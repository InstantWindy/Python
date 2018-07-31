import cv2
import glob

fps = 20   #保存视频的FPS，可以适当调整

#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#saveVideo.avi是要生成的视频名称，（384,288）是图片尺寸
videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(384,288))
#imge存放图片
imgs=glob.glob('imge/*.jpg')
for imgname in imgs:
    frame = cv2.imread(imgname)
    videoWriter.write(frame)
videoWriter.release()
