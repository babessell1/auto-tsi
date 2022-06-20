from file_read_backwards import FileReadBackwards
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from treeTools.rpiTools import *
import sys, getopt
import keyboard
import time
import itertools
import getopt

### For RaspPI with ROS installed ###


def get_mark_size(dbh_list):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in, height_in = bbox.width, bbox.height # inches
    width_pix, height_pix = width_in*fig.dpi, height_in*fig.dpi
    ax_size = plt.axis()
    width_mm = (ax_size[1]-ax_size[0])
    # height_mm = (ax_size[3]-ax_size[2])
    mark_sizes = [(4*dbh*width_pix/width_mm) for dbh in dbh_list]
    
    return mark_sizes

def data_gen():
    def generate_data_from_scan(strings, handle):
        global veri_time_list, save_time
        flag_pass = 0
        # break_flag = 1
        for line in handle:
            line = line.strip()
            if line.startswith(strings[0]):
                scan_str = line.split("[")[1]
                scan_str = scan_str.split("]")[0]
                scan_lst = scan_str.split(", ")
                add_scan = [i for i in scan_lst]
                dist = add_scan[step_start:step_end]
                flag_pass = 1
            if line.startswith(strings[1]) and flag_pass:
                nsecs = line.split()[1]
                while len(nsecs) < 9:
                    nsecs = '0' + nsecs
                
            if line.startswith(strings[2]) and flag_pass:
                secs = line.split()[1]
                save_time = int(secs + nsecs)
                
                yield [dist]
            
    with FileReadBackwards(scanFile) as handle:
        for gen in generate_data_from_scan(["ranges:", "nsecs:", "secs:"], handle):
            #break_flag = 1
            #yield gen[0]
            break
        
    yield gen[0]


def keypress(event):
        global over_color, save_x_list, save_y_list, save_flag, veri_flag, veri_dbh_list, veri_time_list, \
               save_time_list, save_dbh_list, veri_x_list, veri_y_list
       
        if event.key == 'm':
            over_color = 'indigo'
            save_flag = 1
            veri_flag = 0
            
        elif event.key == 'v' and not veri_flag:
            save_flag = 0
            veri_flag = 1
            over_color = 'maroon'
    
        elif event.key == 'p':
            #print(veri_dbh_list)
            print(len(veri_dbh_list))
            print(len(veri_x_list))
            print(len(veri_y_list))
            
            with open(writeFile, 'a') as file:
                file.write('----------------------\n')
                file.write(str(veri_time_list))
                file.write('\n')
                file.write(str(veri_dbh_list))
                file.write('\n')
                file.write(str(veri_x_list))
                file.write('\n')
                file.write(str(veri_y_list))
                file.write('\n')
                veri_time_list = []         
                veri_dbh_list = []
                veri_x_list = []
                veri_y_list = []

def verify(event):
    global x, y, veri_flag, save_time_list, save_x_list, save_y_list, save_dbh_list, cnt, resolved_mark_size, \
           resolved_x, resolved_y, resolved_diameters, veri_x_list, veri_y_list, veri_dbh_list, veri_time_list, distance_maximum

    plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)
    ax.set(xscale = 'linear', yscale = 'linear',
           xlim = (-distance_maximum, distance_maximum),
           ylim = (-distance_maximum, distance_maximum),
           autoscale_on = False)
    plt.gca().set_aspect('equal')
    
    if not veri_flag:
        return

    if cnt == 0:
        resolved_x, resolved_y, resolved_diameters = resolveCircles(save_x_list, save_y_list, save_dbh_list)

    if cnt < len(resolved_x):
        plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)
        ax.set(xscale = 'linear', yscale = 'linear',
               xlim = (-distance_maximum, distance_maximum),
               ylim = (-distance_maximum, distance_maximum),
               autoscale_on = False)
        plt.gca().set_aspect('equal')
        resolved_mark_size = get_mark_size(resolved_diameters)
        plt.scatter(resolved_x[cnt], resolved_y[cnt], c='violet', s=(resolved_mark_size[cnt]))
        
        if event.key == 'k':
            veri_time_list.append(save_time_list[5])
            veri_x_list.append(resolved_x[cnt])
            veri_y_list.append(resolved_y[cnt])
            veri_dbh_list.append(resolved_diameters[cnt])
            
            plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)
            ax.set(xscale = 'linear', yscale = 'linear',
                   xlim = (-distance_maximum, distance_maximum),
                   ylim = (-distance_maximum, distance_maximum),
                   autoscale_on = False)
            plt.gca().set_aspect('equal')
            resolved_mark_size = get_mark_size(resolved_diameters)
            plt.scatter(resolved_x[cnt], resolved_y[cnt], c='lime', s=(resolved_mark_size[cnt]), alpha=0.5)
            print(resolved_diameters[cnt])
            cnt += 1
    
        elif event.key == 't':
            plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)
            ax.set(xscale = 'linear', yscale = 'linear',
                   xlim = (-distance_maximum, distance_maximum),
                   ylim = (-distance_maximum, distance_maximum),
                   autoscale_on = False)
            plt.gca().set_aspect('equal')
            resolved_mark_size = get_mark_size(resolved_diameters)
            plt.scatter(resolved_x[cnt], resolved_y[cnt], marker = 'x', c='red', s=(resolved_mark_size[cnt]))
            
            cnt += 1
            
        elif event.key == 'r':
            
            return
            
            
        fig.clear()
        ax.set(xscale = 'linear', yscale = 'linear',
               xlim = (-distance_maximum, distance_maximum),
               ylim = (-distance_maximum, distance_maximum),
               autoscale_on = False)
        plt.gca().set_aspect('equal')
        plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)
        plt.scatter(x, y, c='cadetblue', s=1)
        resolved_mark_size = get_mark_size(resolved_diameters)
        try:
            plt.scatter(resolved_x[cnt], resolved_y[cnt], c='violet', s=(resolved_mark_size[cnt]))
            plt.scatter(resolved_x, resolved_y, facecolors ='none', edgecolors='darkorange', s=(resolved_mark_size))
        except BaseException:
            plt.scatter(resolved_x, resolved_y, facecolors ='none', edgecolors='darkorange', s=(resolved_mark_size))
        
        for i in range(cnt):
            if resolved_diameters[i] not in veri_dbh_list:
                plt.scatter(resolved_x[i], resolved_y[i], marker = 'x', c='red', s=(resolved_mark_size[i]))
            else:
                plt.scatter(resolved_x[i], resolved_y[i], c='lime', s=(resolved_mark_size[i]), alpha=0.5)

            #plt.axis([-distance_maximum, distance_maximum, -distance_maximum, distance_maximum])
            #plt.gca().set_aspect('equal')
    else:
        print('escape')
        cnt = 0
        veri_flag = 0
        save_flag = 0
        save_time_list = []
        save_dbh_list = []
        save_x_list = []
        save_y_list = []
        
        #overlay = plt.scatter(save_x_list, save_y_list, marker = 'x', c='white', s=0.0001)
        #plt.gca().set_aspect('equal')
        return

def animate(dist, angles):
    global x, y, x_over, y_over, mark_size, save_flag, veri_flag, save_time_list, save_dbh_list, save_x_list, \
        save_y_list, save_time, distance_maximum

    plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)
    ax.set(xscale = 'linear', yscale = 'linear',
           xlim = (-distance_maximum, distance_maximum),
           ylim = (-distance_maximum, distance_maximum),
           autoscale_on = False)
    plt.gca().set_aspect('equal')

    if veri_flag:
        fig.canvas.mpl_connect('key_press_event', verify)
        return
    
    fig.canvas.mpl_connect('key_press_event', keypress)
    angles = np.array(angles)
    
    try:
        distances = np.array(dist, dtype=np.float32)
        
    except:
        pass
    
    try:
        if len(distances) == step_end-step_start:
            distances = 1000.*distances
            x = distances*np.sin(angles)
            y = distances*np.cos(angles)
            stepsList, distList, dbhList, centerDistList, centerStepList = calculateTrees(distances, step_start, step_end, distance_maximum, ratio_minimum, ratio_maximum, limit_noise, diameter_minimum, diameter_maximum)
            fig.clear()
            plt.gca().set_aspect('equal')
            plt.plot(0, 0, 'k+')
            
            col = []
            for st in range(step_start, step_end):
                if st-step_start in [j for i in stepsList for j in i]:
                    col.append('blue')
                else:
                    col.append('cadetblue')
                    
            plt.scatter(x, y, c=col, s=1)
            plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)
            ax.set(xscale = 'linear', yscale = 'linear',
                   xlim = (-distance_maximum, distance_maximum),
                   ylim = (-distance_maximum, distance_maximum),
                   autoscale_on = False)
            plt.gca().set_aspect('equal')
            
            ang_over = np.zeros(len(centerStepList))
            
            for i in range(len(ang_over)):
                step = centerStepList[i]
                ang_over[i] = math.radians((540 - step - step_start)*360/1440) if step <= 540 else math.radians((1980 - step - step_start)*360/1440)
            
            x_over = centerDistList*np.sin(ang_over) 
            y_over = centerDistList*np.cos(ang_over)
            
            mark_size = get_mark_size(dbhList)
            
            plt.scatter(x_over, y_over, c=over_color, s=(mark_size), alpha=0.5)
                                
            plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)
            ax.set(xscale = 'linear', yscale = 'linear',
                   xlim = (-distance_maximum, distance_maximum),
                   ylim = (-distance_maximum, distance_maximum),
                   autoscale_on = False)
            plt.gca().set_aspect('equal')
            print(dbhList)
            
            if save_flag:
                save_x_list.append(x_over)
                save_y_list.append(y_over)
                save_dbh_list.append(dbhList)
                save_time_list.append(save_time)
            
    except:
        pass

def main(argv):

    global scanFile, writeFile, step_start, step_end, distance_maximum, ratio_minimum, ratio_maximum, limit_noise, \
           diameter_minimum, diameter_maximum, time_list, distances, angles, x, y, save_time, save_time_list, \
           save_dbh_list, save_x_list, save_y_list, veri_time_list, veri_dbh_list, veri_x_list, veri_y_list, cnt, \
           veri_flag, save_flag, over_color, resolved_x, resolved_y, animation, fig, ax

    # DEFAULT PARAMETERS, ONLY input_file and output_file required arguments

    scanFile = '' # file to read (output of scan from urg_node in ROS
    writeFile = '' # file to write to for post processing
    step_start = 0 # smallest angle (step number) on LiDAR scanner to start interpreting
    step_end = 1080 # largest angle (step number) on LiDAR scanner to stop interpreting
    distance_maximum = 3000 # maximum distance from LiDAR to interpret from (cm)
    ratio_minimum = 5*1440 # minimum ratio of steps/radians to consider a 'tree' (i.e. how flat a measured object can be
    ratio_maximum = 150*1500 # maximum ratio of steps/radians to consider a 'tree' (i.e. how steep an object can be)
    limit_noise = 35 # max distance the center of an object can deviate from each time step
    diameter_minimum =  2*25.4 # minimum allowable diameter of a tree to consider
    diameter_maximum = 60*25.4 # maximum allowable diamter of a tree to consider

    helpstr = """
    read_scan.py -i <input_file> -o <output_file>, --s <step_start>, -e <step_end>, -x <maximum_distance>,
    -r <minimum ratio>, -a <maximum_ratio>, -l <noise_limit>, -d <minimum_diameter>, -b <maximum_diameter>,
    """

    try:
        opts, args = getopt.getopt(argv, "hi:o:s:e:x:r:a:l:d:b:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('scanVis.py -i <inputfile> -o <outputfile>')

        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('scanVis.py -i <input_file> -o <output_file>')
            sys.exit()
        elif opt in ("-i", "--input_file"):
            scanFile = arg
        elif opt in ("-o", "--output_file"):
            writeFile = arg
        elif opt in ("-s", "--step_start"):
            step_start = arg
        elif opt in ("e", "--step_end"):
            step_end = arg
        elif opt in ("x", "--maximum_distance"):
            distance_maximum = arg
        elif opt in ("-r", "--minimum_ratio"):
            ratio_minimum = arg
        elif opt in ("-a", "--maximum_ratio"):
            ratio_maximum = arg
        elif opt in ("-l", "--noise_limit"):
            limit_noise = arg
        elif opt in ("-d", "--minimum_diameter"):
            diameter_minimum = arg
        elif opt in ("-b", "--maximum_diameter"):
            diameter_maximum = arg
        else:
            print('"{}" is not a valid argument'.format(opt))
            print(helpstr)

    writer = ani.writers['ffmpeg']
    writer = writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    time_list = []
    distances = []
    angles = []
    x = []
    y = []

    save_time = 0
    save_time_list = []
    save_dbh_list = []
    save_x_list = []
    save_y_list = []

    veri_time_list = []
    veri_dbh_list = []
    veri_x_list = []
    veri_y_list = []

    for i in range(step_start, step_end):
        if i <= 540:
            ang = math.radians((540 - i) * 360 / 1440)
        else:
            ang = math.radians((1980 - i) * 360 / 1440)

        angles.append(ang)

    plt.ioff()
    # fig = plt.figure()
    fig, ax = plt.subplots()
    # plt.axis([-distance_maximum, distance_maximum, -distance_maximum, distance_maximum])
    # ax = fig.add_subplot(1,1,1)
    ax.set(xscale='linear', yscale='linear',
           xlim=(-distance_maximum, distance_maximum),
           ylim=(-distance_maximum, distance_maximum),
           autoscale_on=False)
    plt.gca().set_aspect('equal')
    plt.scatter([-distance_maximum, distance_maximum], [-distance_maximum, distance_maximum], c='white', s=0.0001)

    over_color = 'maroon'
    save_flag = 0
    veri_flag = 0

    cnt = 0

    resolved_x, resolved_y, resolved_diameters, resolved_mark_size = [], [], [], []

    animation = ani.FuncAnimation(fig, animate, frames=data_gen, fargs=([angles]), interval=1, repeat=True)
    # animation.save('im.mp4', writer=writer)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])