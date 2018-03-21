# Complete the missing frames by copying the information
import glob
from PIL import Image

image_list = glob.glob("*.jpg")
print len(image_list)

# first map the frames to the range 0-74586

'''def map_to_range(frame_num):
    return int(frame_num*74586.0/12239.0)

for image in image_list:
    img = Image.open(image)
    frame_num = int(image.split("_")[2])
    print frame_num
    a,b,d = image.split("_")[0], image.split("_")[1],image.split("_")[3]
    #frame_num = map_to_range(frame_num)
    
    img.save("../full_frames_2014091408/"+a+"_"+b+"_"+str(frame_num)+"_"+d)
    print "../full_frames_2014091408/"+a+"_"+b+"_"+str(frame_num)+"_"+d
'''

max_frame_num = 74586

def alphabetical(n):
    return int(n.split("_")[2])

image_list.sort(key=alphabetical)

for i,image in enumerate(image_list):
    img = Image.open(image)
    frame_num = int(image.split("_")[2])
    print i
    if i+1 < len(image_list):
        frame_num_next = int(image_list[i+1].split("_")[2])
    else:
        frame_num_next = max_frame_num
    print frame_num, frame_num_next
    a,b,d = image.split("_")[0],image.split("_")[1],image.split("_")[3]
    for j in range(frame_num+1,frame_num_next):
        print "./"+a+"_"+b+"_"+str(j)+"_"+d
        img.save("./"+a+"_"+b+"_"+str(j)+"_"+d)
        pass
