import argparse

parser = argparse.ArgumentParser(description='Video annotation file to auto-complete')

parser.add_argument('--video', metavar='video', help='video annotation .txt to auto-complete')
args = parser.parse_args()

with open(args.video,"rb") as f:
    lines = f.readlines()

video_name = args.video.split(".")
video_name[0] = video_name[0]+"_c"
video_name = video_name[0] +"."+ video_name[1]
print video_name

view_counter = 1
view_counts = [0,0,0]
prev_line = 1
with open(video_name,"wb") as f:
    
    for line in lines:
        
        line = line.strip("\n")
        line_info = line.split(" ")
        
        #print int(line_info[0])-prev_line
        view_counts[view_counter] += int(line_info[0]) - prev_line

        view_counter += 1
        if view_counter > 2:
            view_counter = 0
        
        print line, line_info, view_counter

        if len(line_info) == 2:
            #print line_info
            #print view_counter
            if view_counter != int(line_info[1]):
                print "ERROR"
                exit()
        else:
            pass
            #print line_info, len(line_info)
            #view_counter += 1
        #else:
        #    view_counter += 1

        #if view_counter > 2:
        #    view_counter = 0
        #print line
        #print line_info
        #print line_info[0]
        #print view_counter

        f.write(line_info[0]+" "+str(view_counter)+"\n")
        prev_line = int(line_info[0])

#    print "DONE"
print "Amount of plays", len(lines)/3
print "Counts of viewpoints", view_counts
