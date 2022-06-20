import numpy as np
from datetime import datetime
import time
import pytz
import xlwings as xw
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
import math
from shapely.geometry import Point, Polygon
import geopandas as gpd

################################################################################
# Subfunctions


def findNearest(lst, val):
    """
    Finds and returns the closest value
    :param lst: list of values
    :param val: value to search for closest value of
    :return: closest value
    """

    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argmin()

    return idx, lst[idx]


def find_multiple_nearest(lst, val, k):
    """
    Find multiple closest values
    :param lst: list of values
    :param val: value to search for closest value of
    :param k: number of values to return
    :return: list of length k populated with closest values to val from lst
    """
    lst = np.asarray(lst)
    idx = (np.abs(lst - val)).argpartition(k)[k]

    return idx


def determine_circle_intersections(x0, y0, x1, y1, r0, r1):
    """
    Determine whether or not to circles intersect
    :param x0: x-coordinate of circle 0
    :param y0: y-coordinate of circle 0
    :param x1: x-coordinate of circle 1
    :param y1: y-coordinate of circle 1
    :param r0: radius of circle 0
    :param r1: radius of circle 1
    :return: 1 if intersect 0 if not
    """
    buffer = 0
    # buffer = (r1 + r0)
    distSq = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)
    radSumSq = (r0 + r1) * (r0 + r1)
    if distSq < (radSumSq + buffer * buffer):
        # Circles intersect
        return 1
    else:
        # circles do not intersect
        return 0


def reject_outliers(data, m=2.):
    """
    remove outliers from a list
    :param data: list of values
    :param m: max difference from the median, default 2x difference
    :return: input list with outliers removed
    """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.

    return data[s < m]


def interpolate(scan_time, timestamp_list, lower_index, upper_index, list):
    """
    interpolate values in list base from times in telemetry reads and LiDAR scan times
    :param scan_time: list of LiDAR scan time stamps
    :param timestamp_list: list of telemetry time stamps
    :param lower_index: lower bound of telemetry scan time
    :param upper_index: upper bound of telemetry scan time
    :param list: list of values to interpret synced to scan times.
    :return: value in list at telemetry scan time
    """
    val = list[lower_index] + (scan_time - timestamp_list[lower_index]) * (
            list[upper_index] - list[lower_index]) / (timestamp_list[upper_index] - timestamp_list[lower_index])

    return val


def covariance(stdev_x_pos, stdev_y_pos, stdev_yaw, stdev_x_vel, stdev_y_vel, stdev_yaw_vel):
    """
    determine covarience matrix
    :param stdev_x_pos: standard deviation of x-coordinate
    :param stdev_y_pos: standard deviation of y-coordinate
    :param stdev_yaw: standard deviation of yaw
    :param stdev_x_vel: standard deviation of x velocity
    :param stdev_y_vel: standard deviation of y velocity
    :param stdev_yaw_vel: standard deviation of rotation velocity
    :return: covarience matrix
    """
    cov_x_pos_y_pos = stdev_x_pos * stdev_y_pos
    cov_x_pos_yaw = stdev_x_pos * stdev_yaw
    cov_x_pos_x_vel = stdev_x_pos * stdev_x_vel
    cov_x_pos_y_vel = stdev_x_pos * stdev_y_vel
    cov_x_pos_yaw_vel = stdev_x_pos * stdev_yaw_vel

    cov_y_pos_x_pos = stdev_y_pos * stdev_x_pos
    cov_y_pos_yaw = stdev_y_pos * stdev_yaw
    cov_y_pos_x_vel = stdev_y_pos * stdev_x_vel
    cov_y_pos_y_vel = stdev_y_pos * stdev_y_vel
    cov_y_pos_yaw_vel = stdev_y_pos * stdev_yaw_vel

    cov_yaw_x_pos = stdev_yaw * stdev_x_pos
    cov_yaw_y_pos = stdev_yaw * stdev_y_pos
    cov_yaw_x_vel = stdev_yaw * stdev_x_vel
    cov_yaw_y_vel = stdev_yaw * stdev_y_vel
    cov_yaw_yaw_vel = stdev_yaw * stdev_yaw_vel

    cov_x_vel_x_pos = stdev_x_vel * stdev_x_pos
    cov_x_vel_y_pos = stdev_x_vel * stdev_y_pos
    cov_x_vel_yaw = stdev_x_vel * stdev_yaw
    cov_x_vel_y_vel = stdev_x_vel * stdev_y_vel
    cov_x_vel_yaw_vel = stdev_x_vel * stdev_yaw_vel

    cov_y_vel_x_pos = stdev_y_vel * stdev_x_pos
    cov_y_vel_y_pos = stdev_y_vel * stdev_y_pos
    cov_y_vel_yaw = stdev_y_vel * stdev_yaw
    cov_y_vel_x_vel = stdev_y_vel * stdev_x_vel
    cov_y_vel_yaw_vel = stdev_y_vel * stdev_yaw_vel

    cov_yaw_vel_x_pos = stdev_yaw_vel * stdev_x_pos
    cov_yaw_vel_y_pos = stdev_yaw_vel * stdev_y_pos
    cov_yaw_vel_yaw = stdev_yaw_vel * stdev_yaw
    cov_yaw_vel_x_vel = stdev_yaw_vel * stdev_x_vel
    cov_yaw_vel_y_vel = stdev_yaw_vel * stdev_y_vel

    cov_matrix = np.array([
        [stdev_x_pos ** 2, cov_x_pos_y_pos, cov_x_pos_yaw, cov_x_pos_x_vel, cov_x_pos_y_vel, cov_x_pos_yaw_vel],
        [cov_y_pos_x_pos, stdev_y_pos ** 2, cov_y_pos_yaw, cov_y_pos_x_vel, cov_y_pos_y_vel, cov_y_pos_yaw_vel],
        [cov_yaw_x_pos, cov_yaw_y_pos, stdev_yaw ** 2, cov_yaw_x_vel, cov_yaw_y_vel, cov_yaw_yaw_vel],
        [cov_x_vel_x_pos, cov_x_vel_y_pos, cov_x_vel_yaw, stdev_x_vel ** 2, cov_x_vel_y_vel, cov_x_vel_yaw_vel],
        [cov_y_vel_x_pos, cov_y_vel_y_pos, cov_y_vel_yaw, cov_y_vel_x_vel, stdev_y_vel ** 2, cov_y_vel_yaw_vel],
        [cov_yaw_vel_x_pos, cov_yaw_vel_y_pos, cov_yaw_vel_yaw, cov_yaw_vel_x_vel, cov_yaw_vel_y_vel,
         stdev_yaw_vel ** 2]])

    return np.diag(np.diag(cov_matrix))


################################################################################
# Plotting and Visualization

def plot_histogram(df, col):
    """
    Generates a histogram from a pandas dataframe or array
    :param df: dataframe
    :param col: column to plot
    :return: histogram plot
    """
    plt.rcParams["axes.labelsize"] = 50
    fig = sns.distplot(df.DBH.apply(pd.to_numeric, errors='coerce'), bins=100, color=col)
    fig.set(xlim=(0))
    plt.rcParams.update({'font.size': 50})
    # fig.set_title(title);
    fig.xaxis.set_minor_locator(AutoMinorLocator(4))
    fig.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
    return fig


def make_filter_histogram(level_tracking, tree_DF):
    """
    Plots dbh histogram(s) for specified level of filtration from tracking score
    :param level_tracking: tracking level to plot, 0, 1, 2 or 3
    :param tree_DF: dataframe of measured trees
    :return: histogram plot at different at specified filter level
    """
    if level_tracking > 0:
        sns.set(font_scale=20)
        sns.set_style("darkgrid")
        sns.set(rc={'figure.figsize': (40, 28)})
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        if level_tracking == 1 or level_tracking == 5:
            title = "Before Post-Filtering"
            df1 = tree_DF
            fig1 = plot_histogram(df1, 'seagreen')
            textstr = "Total Values: " + str(len(df1.index))
            fig1.text(0.05, 0.95, textstr, transform=fig1.transAxes, fontsize=70,
                      verticalalignment='top', bbox=props)
            fig1.get_figure().savefig('figure1.png')
            plt.show()

        if level_tracking == 2 or level_tracking == 5:
            title = "DBH Distribution - Low Tracking Score"
            df2 = df1[df1.Track != 0]
            fig2 = plot_histogram(df2, 'royalblue')
            textstr = "Total Values: " + str(len(df2.index))
            fig2.text(0.05, 0.95, textstr, transform=fig2.transAxes, fontsize=70,
                      verticalalignment='top', bbox=props)
            fig2.get_figure().savefig('figure2.png')
            plt.show()

        if level_tracking == 3 or level_tracking == 5:
            title = "DBH Distribution - Mid Tracking Score"
            df3 = df2[df2.Track != 1]
            fig3 = plot_histogram(df3, 'mediumorchid')
            textstr = "Total Values: " + str(len(df3.index))
            fig3.text(0.05, 0.95, textstr, transform=fig3.transAxes, fontsize=70,
                      verticalalignment='top', bbox=props)
            fig3.get_figure().savefig('figure3.png')
            plt.show()

        if level_tracking == 4 or level_tracking == 5:
            title = "DBH Distribution - High Tracking Score"
            df4 = df3[df3.Track != 2]
            fig4 = plot_histogram(df4, 'crimson')
            textstr = "Total Values: " + str(len(df4.index))
            fig4.text(0.05, 0.95, textstr, transform=fig4.transAxes, fontsize=70,
                      verticalalignment='top', bbox=props)
            fig4.get_figure().savefig('figure4.png')
            plt.show()


def make_geograph(sample_areas_GDF, tree_DF_resolved, shp_reference, flag_plot):
    """
    Generate a map of measured trees
    :param sample_areas_GDF: geodataframe with measured treas
    :param tree_DF_resolved: dataframe of resolved measured trees
    :param shp_reference: shpfile to plot over
    :param flag_plot: boolean , show plot or not
    :return: resolved geodataframe
    """

    print(tree_DF_resolved.head())

    dbh = tree_DF_resolved.DBH.to_numpy()
    lat = tree_DF_resolved.Latitude.to_numpy()
    lon = tree_DF_resolved.Longitude.to_numpy()

    geometry_resolved = [Point(x, y).buffer(d / 2000) for x, y, d in
                         zip(tree_DF_resolved.Longitude, tree_DF_resolved.Latitude, tree_DF_resolved.DBH)]
    tree_GDF_resolved = gpd.GeoDataFrame(tree_DF_resolved, geometry=geometry_resolved)

    if flag_plot == 1:
        reference_GDF = gpd.read_file(shp_reference)
        reference_GDF.crs = {'init': 'epsg:4326'}
        reference_GDF = reference_GDF.to_crs(epsg=32617)

        xmin = np.min(lon)
        xmax = np.max(lon)
        ymin = np.min(lat)
        ymax = np.max(lat)
        dbh_max = np.max(dbh)

        f, ax = plt.subplots()

        sample_basal_area_per_hectare, sample_basal_area_ft_per_acre, trees_per_acre_list, avg_dbh_in_list = calculate_basal_area_stationary(
            tree_GDF_resolved, sample_areas_GDF)
        print(sample_basal_area_ft_per_acre)
        print(trees_per_acre_list)
        print(avg_dbh_in_list)
        sample_basal_area_per_hectare = np.array(sample_basal_area_per_hectare)
        lower = sample_basal_area_per_hectare.min()
        upper = sample_basal_area_per_hectare.max()
        colors = plt.cm.jet((sample_basal_area_per_hectare - lower) / (upper - lower))

        sample_areas_GDF.plot(ax=ax, alpha=0.25, color=colors)
        ax = tree_GDF_resolved.plot(ax=ax, color='b', alpha=0.9)
        tree_GDF_resolved.apply(
            lambda x: ax.annotate(s=float("{:.2f}".format(x.DBH / 25.4)), xy=x.geometry.centroid.coords[0],
                                  ha='center'), axis=1);

        reference_GDF.plot(ax=ax, alpha=0.4, color="grey")

        ax.set_xlim(xmin - 50, xmax + 50)  # added/substracted value is to give some margin around total bounds
        ax.set_ylim(ymin - 50, ymax + 50)
        plt.show()

    return tree_GDF_resolved


def plot_sample_area(scan_center_x_list, scan_center_y_list, scan_center_yaw_list, maximum_distance, step_start,
                     step_end):
    """
    Highlight area where LiDAR measured
    :param scan_center_x_list: center x-coordinate list
    :param scan_center_y_list: center y-coordinate list
    :param scan_center_yaw_list:  center yaw list
    :param maximum_distance: max distance trees were measured from LiDAR
    :param step_start: start angle on LiDAR
    :param step_end: end-angle on LiDAR
    :return: GDF with scan areas
    """
    sample_areas = []
    for i in range(len(scan_center_yaw_list)):
        yaw = scan_center_yaw_list[i]
        scan_boundary = [(scan_center_x_list[i], scan_center_y_list[i])]
        for step in range(step_start, step_end):
            if step <= 540:
                ang = math.radians((540 - step) * 360 / 1440)
            else:
                ang = math.radians((1980 - step) * 360 / 1440)

            scan_boundary.append(((scan_center_x_list[i] + maximum_distance * np.sin(yaw + ang) / 1000),
                                  (scan_center_y_list[i] + maximum_distance * np.cos(yaw + ang) / 1000)))

        print(scan_boundary)
        polygon = Polygon(scan_boundary)
        sample_areas.append(polygon)
        print(polygon)

    print([scan_center_x_list, scan_center_y_list])
    sample_areas_DF = pd.DataFrame(
        np.transpose(np.vstack((np.transpose(scan_center_x_list), np.transpose(scan_center_y_list)))),
        columns=['Center X', 'Center Y'])
    print(sample_areas_DF.head())
    sample_areas_GDF = gpd.GeoDataFrame(sample_areas_DF, geometry=sample_areas)

    return sample_areas_GDF


################################################################################

def extract_laser_ros(txt_file):
    """
    txt_file is raw information from the rover (.txt)
    timestart is initial time to start considering data
    ('FIRST' to start from begining, otherwise '%Y-%m-%d %H:%M')
    timestop is the time to stop considering data.
    ('LAST' to use the rest, otherwise '%Y-%m-%d %H:%M'
    """

    time = np.empty((0, 1), int)
    distance = np.empty((0, 1081), float)

    # read raw rover data >> time in epoch and nanoepoch.For each stamp, also
    # output distance and step# for each step
    print('open file')
    cnt = 0
    with open(txt_file, 'r') as handle:
        for line in handle:
            print(cnt)
            thisLine = line.strip()

            if thisLine.startswith("secs:"):
                thisEpoch = int(thisLine.split()[1])

            elif thisLine.startswith("nsecs:"):
                strNtime = thisLine.split()[1]

                # correct any instances where trailing '0's were remove from nanoepoch values
                while len(strNtime) < 9:
                    strNtime = '0' + strNtime

                thisTime = int(str(thisEpoch) + strNtime)
                time = np.append(time, thisTime)

            elif thisLine.startswith("ranges:"):
                scanStr = thisLine.split("[")[1]
                scanStr = scanStr.split("]")[0]
                # print(scanStr)
                scanLst = scanStr.split(", ")
                addScan = [i for i in scanLst]
                # addScan = np.array(addScan)
                if len(addScan) == 1081:
                    distance = np.append(distance, [addScan], axis=0)
                    print(cnt)

            cnt += 1

    # convert m to mm and store in an array
    print(distance.shape)
    print(distance)
    distance = 1000 * np.asarray(distance).astype(np.float32)
    print(time.shape)
    print(time)

    return time, distance


def extract_laser_stationary(stat_file):
    """
    Read output from stationary_sampling.py
    :param stat_file:
    :return: lists of scan times, coordinates, and diameters
    """
    scan_time_list, dbh_list, x_list, y_list = [], [], [], []
    with open(stat_file, 'r') as handle:
        for line in handle:
            line = line.replace('array', '')
            line = line.replace('(', '')
            line = line.replace(')', '')
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace('array', '')
            line = line.replace(' ', '')
            line = line.replace('\n', '')
            if line.startswith('----'):
                cnt = 0
            else:
                this_list = line.split(',')
                if this_list != ['']:
                    if cnt == 1:
                        this_list = np.asarray(this_list, dtype='uint64')
                        scan_time_list.append(this_list)

                    elif cnt == 2:
                        this_list = np.asarray(this_list, dtype='float32')
                        dbh_list.append(this_list)
                    elif cnt == 3:
                        this_list = np.asarray(this_list, dtype='float32')
                        x_list.append(this_list)
                    elif cnt == 4:
                        this_list = np.asarray(this_list, dtype='float32')
                        y_list.append(this_list)

            cnt += 1

        flat_list = []
        for sublist in scan_time_list:
            for item in sublist:
                flat_list.append(item)
        scan_time_list = flat_list

        scan_time_list = np.asarray(scan_time_list, dtype='uint64')

        flat_list = []
        for sublist in dbh_list:
            for item in sublist:
                flat_list.append(item)
        dbh_list = flat_list

        dbh_list = np.asarray(dbh_list, dtype='float32').flatten()

        flat_list = []
        for sublist in x_list:
            for item in sublist:
                flat_list.append(item)
        x_list = flat_list

        x_list = np.asarray(x_list, dtype='float32').flatten()

        flat_list = []
        for sublist in y_list:
            for item in sublist:
                flat_list.append(item)
        y_list = flat_list

        y_list = np.asarray(y_list, dtype='float32').flatten()

        print(scan_time_list)
        print(dbh_list)
        print(x_list)
        print(y_list)

    return scan_time_list, dbh_list, x_list, y_list


def extract_gps(txt_file):
    """
    Read telemetry values from Mission Planner
    :param txt_file: mission planner telemetry file
    :return: geodataframe with GPS reads
    """
    timestamp_list = []
    yawList = []
    latitude_list = []
    longitude_list = []
    yaw_vel_list = []
    lat_vel_list = []
    lon_vel_list = []
    yaw_accel_list = []
    lat_accel_list = []
    lon_accel_list = []

    handle = open(txt_file, 'r', encoding="utf8")
    time_ms_str = "000000"
    yaw_err = 100
    lat = 0
    lon = 0
    lat_vel = 0
    lon_vel = 0
    lat_accel = 0
    lon_accel = 0
    yaw_accel = 0
    newPos = 0
    newYaw = 0
    newAccel = 0
    time_epoch = 0
    boot_time0 = 0

    for line in handle:
        if " time_unix_usec " in line and " 2 " in line:  # clock
            newPos = 0
            newYaw = 0
            spl = line.split()
            time_epoch_idx = spl.index("time_unix_usec") + 1
            time_epoch = int(spl[time_epoch_idx])  # unix time in microsec (epoch)
            boot_time0_idx = spl.index("time_boot_ms") + 1
            boot_time0 = int(spl[boot_time0_idx])

        if "time_boot_ms" in line and " 1E " in line and " yaw " in line:  # compass
            newYaw = 1
            spl = line.split()
            boot_time1_idx = spl.index("time_boot_ms") + 1
            boot_time1 = int(spl[boot_time1_idx])
            time_epoch = int(((time_epoch / 1000) + boot_time1 - boot_time0) * 1000)
            yaw_idx = spl.index("yaw") + 1
            yaw = float(spl[yaw_idx])  #
            yaw_vel_idx = spl.index("yawspeed") + 1
            yaw_vel = float(spl[yaw_vel_idx])

            time_epoch = str(time_epoch)
            while len(time_epoch) < 19:
                time_epoch = time_epoch + '0'

            time_epoch = int(time_epoch)

        if " 21 " in line and " lat " in line and " lon " in line and " vx " in line:  # gps
            newPos = 1
            spl = line.split()
            lat_idx = spl.index("lat") + 1
            lat = float(spl[lat_idx]) * (10 ** (-7))
            lon_idx = spl.index("lon") + 1
            lon = float(spl[lon_idx]) * (10 ** (-7))

        if newPos and newYaw and lat != 0 and lon != 0:
            newPos = 0
            newYaw = 0

            timestamp_list.append(time_epoch)
            yawList.append(yaw)
            latitude_list.append(lat)
            longitude_list.append(lon)

    handle.close()
    timestamp_list = np.asarray(timestamp_list[1:]).astype(np.uint64)
    yawList = np.asarray(yawList[1:])
    latitude_list = np.asarray(latitude_list[1:])
    longitude_list = np.asarray(longitude_list[1:])

    gps_data = np.column_stack((timestamp_list, longitude_list, latitude_list, yawList))
    gps_DF = pd.DataFrame(gps_data, columns=['Time', 'Longitude', 'Latitude', 'Yaw'])
    gps_GDF = gpd.GeoDataFrame(gps_DF, geometry=gpd.points_from_xy(gps_DF.Longitude, gps_DF.Latitude))
    gps_GDF.crs = {'init': 'epsg:4326'}
    gps_GDF = gps_GDF.to_crs(epsg=32617)

    return gps_GDF


def filter_kalman(timestamp_list, measurement_array):
    """
    Filter measurements using a Kalman filter
    :param timestamp_list: list of time stamps for measurments
    :param measurement_array: measurements in order (longiture, latitude, yaw, longitudinal velocity,
           latitudinal velocity, yaw velocity)
    :return: filter measurements in same order
    """

    longitude_list_filt = np.zeros(measurement_array[0][:].shape)
    latitude_list_filt = np.zeros(measurement_array[1][:].shape)
    yawList_filt = np.zeros(measurement_array[2][:].shape)
    lon_vel_list_filt = np.zeros(measurement_array[3][:].shape)
    lat_vel_list_filt = np.zeros(measurement_array[4][:].shape)
    yaw_vel_list_filt = np.zeros(measurement_array[5][:].shape)

    # Population
    N = len(timestamp_list)

    # Initial State Matrix
    X = np.array([measurement_array[0][0],
                  measurement_array[1][0],
                  measurement_array[2][0],
                  measurement_array[3][0],
                  measurement_array[4][0],
                  measurement_array[5][0]])

    print("\nInitial State:")
    print(X)

    # Initial Covariance Matrix

    P = np.eye(6) * 10
    print("\nInitial Covariance Matrix:")
    print(P)

    # Process Noise Covariance Matrix
    stdev_x_pos = np.std(measurement_array[0][:])
    stdev_y_pos = np.std(measurement_array[1][:])
    stdev_yaw = np.std(measurement_array[2][:])
    stdev_x_vel = np.std(measurement_array[3][:])
    stdev_y_vel = np.std(measurement_array[4][:])
    stdev_yaw_vel = np.std(measurement_array[5][:])

    Q = covariance(stdev_x_pos, stdev_y_pos, stdev_yaw, stdev_x_vel, stdev_y_vel, stdev_yaw_vel)
    print("\nProcess Noise Covariance Matrix:")
    print(Q)

    # Measurement Noise Covariance Matrix
    # compass_certainty = 0.1
    # gps_certainty = 0.00001
    # imu_certainty = 10.
    com_certainty = 10.
    gps_certainty = 100.
    imu_certainty = 1000.
    # R = np.eye(6)*sensor_certainty
    R = np.array([[gps_certainty, 0, 0, 0, 0, 0],
                  [0, gps_certainty, 0, 0, 0, 0],
                  [0, 0, com_certainty, 0, 0, 0],
                  [0, 0, 0, imu_certainty, 0, 0],
                  [0, 0, 0, 0, imu_certainty, 0],
                  [0, 0, 0, 0, 0, imu_certainty]])

    # Unit matrix
    I = np.eye(6)

    for i in range(0, N - 1):
        print('\n\n------------------------------------')
        print("Index: " + str(i))
        # time step --#.0001
        dt = (timestamp_list[i + 1] - timestamp_list[i])
        # print(dt)
        # Measurement Matrix
        H = np.eye(6)
        print("\nMeasurement Matrix")
        print(H)

        # State Transition Matrix
        A = np.array([[1., 0., 0., dt, 0., 0.],
                      [0., 1., 0., 0., dt, 0.],
                      [0., 0., 1., 0., 0., dt],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.]])

        print("\nState Transition Matrix")
        print(A)

        # Control Vector
        U = np.array([measurement_array[6][i],
                      measurement_array[7][i],
                      0])

        print("\nControl Vector")
        print(U)

        # Control Matrix
        B = np.array([[0.5 * dt * dt, 0., 0.],
                      [0., 0.5 * dt * dt, 0.],
                      [0., 0., 0.5 * dt * dt],
                      [dt, 0., 0., ],
                      [0., dt, 0., ],
                      [0., 0., dt]])

        print("\nControl Matrix")
        print(B)
        # Prediction
        X = A @ X + B @ U
        P = A @ P @ (A.T) + Q

        # Correction
        S = H @ P @ (H.T) + R
        K = (P @ (H.T)) @ np.linalg.pinv(S)

        # Update estimate from measurements
        Z = measurement_array[:6, i].reshape(6, 1)
        y = Z - (H @ X)

        X = X + (K @ y)
        print("New State Matrix:")
        print(X)
        # update error covariance
        P = (I - (K @ H)) @ P
        print("\nNew Covariance Matrix:")
        print(P)

        longitude_list_filt[i] = X[0][0]
        latitude_list_filt[i] = X[1][0]
        yawList_filt[i] = X[2][0]
        lon_vel_list_filt[i] = X[3][0]
        lat_vel_list_filt[i] = X[4][0]
        yaw_vel_list_filt[i] = X[5][0]

    plt.title("longitude")
    plt.plot(timestamp_list[:-2], longitude_list_filt[:-2], 'b')
    plt.plot(timestamp_list[:-2], measurement_array[0][:-2], 'r.')
    plt.show()

    plt.title("latitude")
    plt.plot(timestamp_list, latitude_list_filt, 'b')
    plt.plot(timestamp_list, measurement_array[1], 'r.')
    plt.show()

    plt.title("yaw")
    plt.plot(timestamp_list, yawList_filt, 'b')
    plt.plot(timestamp_list, measurement_array[2], 'r.')
    plt.show()

    plt.title("longitude-velocity")
    plt.plot(timestamp_list, lon_vel_list_filt, 'b')
    plt.plot(timestamp_list, measurement_array[3], 'r.')
    plt.show()

    plt.title("latitude-velocity")
    plt.plot(timestamp_list, lat_vel_list_filt, 'b')
    plt.plot(timestamp_list, measurement_array[4], 'r.')
    plt.show()

    plt.title("yawspeed")
    plt.plot(timestamp_list, yaw_vel_list_filt, 'b')
    plt.plot(timestamp_list, measurement_array[5], 'r.')
    plt.show()

    return longitude_list_filt[:], latitude_list_filt[:], yawList_filt[:], \
           lon_vel_list_filt[:], lat_vel_list_filt[:], yaw_vel_list_filt[:]


def calculate_trees(distance, step_start, step_end, distance_maximum, ratio_minimum, ratio_maximum, limit_noise,
                    diameter_minimum, diameter_maximum):
    """
    Determines which objects are round enough to be trees, also uses other parameters like step range to consider to
    determine what objects are trees.
    :param distance: distance of object
    :param step_start: angle where measurments start
    :param step_end:  angle where measurements end
    :param distance_maximum: maximum distance for objects to be measured
    :param ratio_minimum: allowable flatness to consider a tree (not used right now, can be reactivated)
    :param ratio_maximum: allowable steepness to consider a tree (not used right now, can be reactivated)
    :param limit_noise: max distance the center of an object can deviate from each time step
    :param diameter_minimum: minimum diameter a tree can be measured at
    :param diameter_maximum: max diameter a tree can be measured at
    :return: list of step number (angle) lists that define trees, list of distances that define a measured tree,
             list of center distances for each tree, and a list of center step number (angle) for each tree
     steps_list, dist_list, dbh_list, center_dist_list, center_step_list_
    """

    steps_list = []
    dist_list = []
    dbh_list = []
    center_dist_list = []
    center_step_list_ = []
    new_steps = []
    new_dist = []  # step, distance
    step = distance[step_start:step_end]

    for j in range(len(step)):
        dist = step[j]

        # noise filter
        if new_steps and np.absolute((dist - new_dist[-1])) <= limit_noise:
            new_steps.append(j + step_start)
            new_dist.append(dist)

        elif new_steps and np.absolute((dist - new_dist[-1])) > limit_noise:
            # if len(newSteps)*new_dist[int(len(new_dist)/2)] >= ratio_minimum \
            # and len(newSteps)*new_dist[int(len(new_dist)/2)] <= ratio_maximum \
            if new_dist[int(len(new_dist) / 2)] <= distance_maximum:
                steps_list.append(new_steps)
                dist_list.append(new_dist)
                new_steps = []
                new_dist = []

            else:
                new_steps = []
                new_dist = []

        else:
            new_steps = [(j + step_start)]
            new_dist = [dist]

    if steps_list:
        for j in range(len(steps_list)):
            steps = steps_list[j]
            distan = dist_list[j]

            if 2 * abs(distan[0] - distan[-1]) / (distan[-1] + distan[0]) < 0.1:  # 0.3
                # length to middle of arc
                midStep = int((steps[-1] + steps[0]) / 2)
                # chordlength
                C = (distan[-1] + distan[0]) * np.sin(np.pi / 720 * (midStep - steps[0]))
                #
                L = (((distan[-1] + distan[0]) / 2) ** 2 - (C / 2) ** 2) ** 0.5
                midDist = distance[midStep]
                # chord height
                h = L - midDist
                # intersecting chords theorem -> diameter
                dbh = (4 * h ** 2 + C ** 2) / (4 * h)

                # ensure dimensions are sensible for an arc
                if C / dbh > 0.20 and h / C < 0.5 and h > 0 \
                        and abs(midDist - min(distan)) / midDist < 0.05:
                    dbh_list.append(dbh)
                    center_dist_list.append(L)
                    center_step_list_.append(midStep)

                else:
                    dbh_list.append(0)
                    center_dist_list.append(0)
                    center_step_list_.append(0)

            else:
                dbh_list.append(0)
                center_dist_list.append(0)
                center_step_list_.append(0)

    deleteIdx = []

    for j in range(len(dbh_list)):
        if dbh_list[j] < 0 or dbh_list[j] < diameter_minimum or dbh_list[j] > diameter_maximum:
            deleteIdx.append(j)

    if deleteIdx:
        steps_list = [i for j, i in enumerate(steps_list)
                      if j not in deleteIdx]

        dist_list = [i for j, i in enumerate(dist_list)
                     if j not in deleteIdx]

        dbh_list = [i for j, i in enumerate(dbh_list)
                    if j not in deleteIdx]

        center_dist_list = [i for j, i in enumerate(center_dist_list)
                            if j not in deleteIdx]

        center_step_list_ = [i for j, i in enumerate(center_step_list_)
                             if j not in deleteIdx]

    return steps_list, dist_list, dbh_list, center_dist_list, center_step_list_


def tree_track_filtered(dbh_list, center_dist_list, center_step_list_, diameter_list_1, center_distance_list_1,
                        center_step_list_1, diameter_list_2, center_distance_list_2, center_step_list_2, diameter_list_3,
                        center_distance_list_3, center_step_list_3, limit_tracking_error):
    """
    Looks at up to 4 continuous scans and filters based of consensus between each scan. Accepts a set of information
    from adjacent timesteps to apply tracking scores on values that consistantly confirm themselves to be extremely
    close to their neighboring measurements and tags them to demonstrate how filtering can potentially reduce error if
    a fast sampling speed is used. Currently, this filtering is only for current plotting and investigation purposes.
    :param dbh_list: list of diameters
    :param center_dist_list: list of distances to center
    :param center_step_list_: list of step number (angle) of center of trees
    :param diameter_list_1: second list of diameters measured after first
    :param center_distance_list_1: second list of center distances
    :param center_step_list_1: second list of enter step number
    :param diameter_list_2: third list of diameters after after first
    :param center_distance_list_2: etc
    :param center_step_list_2: etc
    :param diameter_list_3: etc
    :param center_distance_list_3: etc
    :param center_step_list_3: etc
    :param limit_tracking_error: percent difference required for two trees to be considered the same
    :return:
    """

    lst0, lst1, lst2, lst3, isNew, inView = [], [], [], [], [], []
    passCheck1 = False
    passCheck2 = False
    passCheck3 = False
    failCheck1 = False
    failCheck2 = False
    failCheck3 = False

    for idx0 in range(len(center_step_list_)):
        # flags the measurement as being in dead center of few from the camera.
        lst0.append(idx0)
        stepVal = center_step_list_[idx0]
        viewFlag = True if stepVal <= 600 and stepVal >= 480 else False
        inView.append(viewFlag)

        k = 0
        while passCheck1 == False and failCheck1 == False:
            if not center_step_list_1 or k >= len(center_step_list_1):
                lst1.append(np.nan)
                break

            idx1 = find_multiple_nearest(center_step_list_1, stepVal, k)

            if np.abs(diameter_list_1[idx1] - dbh_list[idx0]) / dbh_list[idx0] <= limit_tracking_error \
                    and np.abs(center_distance_list_1[idx1] - center_dist_list[idx0]) / \
                    center_dist_list[idx0] <= limit_tracking_error:

                passCheck1 = True
                lst1.append(idx1)

            elif k <= 3:
                k += 1

            else:
                lst1.append(np.nan)
                failCheck1 = True
                break

        k = 0
        while passCheck2 == False and failCheck2 == False:
            if not center_step_list_2 or k >= len(center_step_list_2):
                lst2.append(np.nan)
                break

            if passCheck1 == False:
                idx1 = idx0
                stepVal = center_step_list_[idx0]
                compDist = center_dist_list[idx0]
                compDbh = dbh_list[idx0]

            else:
                stepVal = center_step_list_1[idx1]
                compDist = center_distance_list_1[idx1]
                compDbh = diameter_list_1[idx1]

            idx2 = find_multiple_nearest(center_step_list_2, stepVal, k)

            if np.abs(diameter_list_2[idx2] - compDbh) / compDbh <= limit_tracking_error \
                    and np.abs(center_distance_list_2[idx2] - compDist) / compDist <= limit_tracking_error:
                passCheck2 = True
                lst2.append(idx2)

            elif k <= 3:
                k += 1

            else:
                lst2.append(np.nan)
                failCheck2 = True
                break

        k = 0
        while passCheck3 == False and failCheck3 == False:
            if not center_step_list_3 or k >= len(center_step_list_3):
                lst3.append(np.nan)
                break

            if passCheck2 == False and passCheck1 == False:
                idx2 = idx0
                stepVal = center_step_list_[idx0]
                compDist = center_dist_list[idx0]
                compDbh = dbh_list[idx0]

            elif passCheck2 == False and passCheck1 == True:
                idx2 = idx1
                stepVal = center_step_list_1[idx1]
                compDist = center_distance_list_1[idx1]
                compDbh = diameter_list_1[idx1]

            else:
                stepVal = center_step_list_2[idx2]
                compDist = center_distance_list_2[idx2]
                compDbh = diameter_list_2[idx2]

            idx3 = find_multiple_nearest(center_step_list_3, stepVal, k)

            if np.abs(diameter_list_3[idx3] - compDbh) / compDbh <= limit_tracking_error \
                    and np.abs(center_distance_list_3[idx3] - compDist) / compDist <= limit_tracking_error:
                passCheck3 = True
                lst3.append(idx3)

            elif k <= 3:
                k += 1

            else:
                lst3.append(np.nan)
                failCheck3 = True
                break

        if sum([passCheck1, passCheck2, passCheck3]) == 3:
            isNew.append(3)

        elif sum([passCheck1, passCheck2, passCheck3]) == 2:
            isNew.append(2)

        elif passCheck1 == True:
            isNew.append(1)

        else:
            isNew.append(0)

    return lst0, lst1, lst2, lst3, isNew, inView


def tree_track_unfiltered(center_step_list_):
    """
    Accepts a set of information from adjacent timesteps to apply tracking scores on values that consistantly confirm
    themselves to be extremely close to their neighboring measurements and tags them to demonstrate how filtering can
    potentially reduce error if a fast sampling speed is used. Currently, this filtering is only for current plotting
    and investigation purposes.
    :param dbh_list:
    :param center_dist_list:
    :param center_step_list_:
    :return:
    """

    lst0 = []

    in_view = []

    for idx0 in range(len(center_step_list_)):
        # flags the measurement as being in dead center of few from the camera.
        lst0.append(idx0)
        step_val = center_step_list_[idx0]
        view_flag = True if step_val <= 600 and step_val >= 480 else False
        in_view.append(view_flag)

    return lst0, in_view


def alignLaserGPS(time_unix, timestamp_list, yawList, latitude_list, longitude_list):
    """
    Align LiDAR measurements with GPS measurments
    :param time_unix: unix time at LidAR measurement
    :param timestamp_list: list of time steps from mission planner
    :param yawList: list of yaw vals to align from mission planner
    :param latitude_list: list of lat vals to align from mission planner
    :param longitude_list: list of lon vals to align
    :return: values for lon, lat and yaw closest to LiDAR measurement, return nan if not suffiently close
    """
    strTime = str(time_unix)
    while len(strTime) < 19:
        strTime = strTime + '0'
        print(strTime)
    time = int(strTime)

    try:
        upperidx = np.where(timestamp_list == np.max(timestamp_list[timestamp_list <= time]))[0][0]
        loweridx = np.where(timestamp_list == np.min(timestamp_list[timestamp_list >= time]))[0][0]

        if timestamp_list[upperidx] <= time + 1000000 and timestamp_list[loweridx] >= time - 1000000:
            sensor_yaw = interpolate(time, timestamp_list, loweridx, upperidx, yawList)
            sensor_latitude = interpolate(time, timestamp_list, loweridx, upperidx, latitude_list)
            sensor_longitude = interpolate(time, timestamp_list, loweridx, upperidx, longitude_list)
            # print(sensor_yaw)
            # print(time)

            # if np.abs(sensor_yaw-yawList[idx-1]) > 3

        else:
            sensor_yaw = np.nan
            sensor_latitude = np.nan
            sensor_longitude = np.nan

    except:
        sensor_yaw = np.nan
        sensor_latitude = np.nan
        sensor_longitude = np.nan

    return sensor_latitude, sensor_longitude, sensor_yaw


def determine_coord(sensor_latitude, sensor_longitude, sensor_yaw, dist, step):
    """
    Return absolute corrdinates from relative distance from the LiDAR and GPS coordinates
    :param sensor_latitude: latitude reported from the GPS
    :param sensor_longitude: longitude report from the GPS
    :param sensor_yaw: yaw reported from the IMU
    :param dist: measured for this object at this time step
    :param step: centerline step number to this object
    :return: absolute coordinates
    """

    dist = dist / 1000  # convert mm to m
    if sensor_yaw < 0:
        sensor_yaw += 2 * np.pi

    if step <= 540:
        tree_ang = math.radians((540 - step) * 360 / 1440)
    else:
        tree_ang = math.radians((1980 - step) * 360 / 1440)

    ang = tree_ang + sensor_yaw

    dx, dy = dist * np.sin(ang), dist * np.cos(ang)
    coord = sensor_longitude + dx, sensor_latitude + dy

    return coord


def determine_coord_stationary(sensor_latitude, sensor_longitude, sensor_yaw, x, y):
    """
    Return absolute corrdinates from relative distance from the LiDAR and GPS coordinates, for stationary test
    :param sensor_latitude: latitude reported from the GPS
    :param sensor_longitude: longitude report from the GPS
    :param sensor_yaw:  yaw reported from the IMU
    :param x: x-coordinate of object (relative)
    :param y: y-coordinate of object (relative)
    :return: absolute coordinates
    """
    x = float(x) / 1000
    y = float(y) / 1000
    dist = (x * x + y * y) ** (0.5)
    tree_ang = np.arctan(x / y)
    ang = tree_ang + sensor_yaw
    dx, dy = dist * np.sin(ang), dist * np.cos(ang)
    coord = sensor_longitude + dx, sensor_latitude + dy

    return coord


def calculate_basal_area(tree_GDF_resolved):
    """
    Calculate Basal area from measured trees and sample area approximated as max and min coordinates traveled
    NOTE THIS CALC IS ONLY RELIABLE IF SAMPLE AREA IS A RECTANGLE
    :param tree_GDF_resolved:
    :return: gdf with basal area, basal area / hectare
    """
    geometry = tree_GDF_resolved['geometry']
    minx = 9999999
    miny = 9999999
    maxx = -9999999
    maxy = -9999999

    for geo in geometry:
        bounds = geo.bounds
        minx = bounds[0] if bounds[0] < minx else minx
        miny = bounds[1] if bounds[1] < miny else miny
        maxx = bounds[2] if bounds[2] > maxx else maxx
        maxy = bounds[3] if bounds[3] > maxy else maxy

    polygon_sampled = Polygon([(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)])
    polygon_area = (maxx - minx) * (maxy - miny)
    basal_area = np.zeros(tree_GDF_resolved['DBH'].shape)
    i = 0
    for dbh in tree_GDF_resolved['DBH']:
        basal_area[i] = np.pi * (dbh / 2000) ** 2
        i += 1

    tree_GDF_resolved['Basal Area'] = basal_area
    basal_area_per_hectare = np.sum(basal_area) / (polygon_area / 10000)

    return tree_GDF_resolved, basal_area_per_hectare


def calculate_basal_area_stationary(tree_GDF_stationary, sample_areas_GDF):
    """
    Calculates bases are of a set of stationary sample areas where stationary_sampling.py was used
    :param tree_GDF_stationary: geodataframe of stationary sampling
    :param sample_areas_GDF: sample area geodataframe
    :return: list of basal areas / hectare, list of sample basal areas ft / acre, list of trees per acre,
             list of average dbh
    """

    sample_basal_area_per_hectare = []
    sample_basal_area_ft_per_acre = []
    trees_per_acre_list = []
    avg_dbh_in_list = []
    for polygon in sample_areas_GDF['geometry']:
        dbh_inside = []
        basal_list = []
        for i in range(len(tree_GDF_stationary['Latitude'])):
            y = tree_GDF_stationary['Latitude'][i]
            x = tree_GDF_stationary['Longitude'][i]
            p = Point((x, y))
            if p.within(polygon):
                d = tree_GDF_stationary['DBH'][i]
                dbh_inside.append(d)

        i = 0
        for dbh in dbh_inside:
            basal_list.append(np.pi * ((dbh / 2) ** 2) / 1000000)
            i += 1

        avg_dbh_in = np.mean(dbh_inside) / 25.4
        avg_dbh_in_list.append(avg_dbh_in)
        basal_per_hectare = np.sum(basal_list) / (polygon.area / 10000)
        trees_per_acre = len(basal_list) / (polygon.area / 4046.86)
        sample_basal_area_per_hectare.append(basal_per_hectare)
        trees_per_acre_list.append(trees_per_acre)
        basal_ft_per_acre = basal_per_hectare * 10.764 / 2.471052
        sample_basal_area_ft_per_acre.append(basal_ft_per_acre)

    return sample_basal_area_per_hectare, sample_basal_area_ft_per_acre, trees_per_acre_list, avg_dbh_in_list


################################################################################
def dbh_pipeline(scan_file, telem_log, xls_raw, level_tracking, step_start, step_end,
                 distance_maximum, ratio_minimum, ratio_maximum, limit_noise, diameter_minimum, diameter_maximum,
                 limit_tracking_error):
    """
    Main pipeline to process scan and telemetry reads into a dataframe of tree measurments
    :param scan_file: path to raw LiDAR collected from ROS urg_node function (.txt)
    :param telem_log: path of desired output to be written in .xlsx format
    :param xls_raw:
    :param level_tracking: number of tracking levels to use 0-4 for filtering
    :param step_start: the first lidar step interested in
    :param step_end: the last lidar step interested in
    :param distance_maximum: max allowable to distance to measure a tree
    :param ratio_minimum: minimum ratio of steps/radians to consider a 'tree' (allowable flatness)
    :param ratio_maximum: maximum ratio of steps/radians to consider a 'tree' (allowable steepness)
    :param limit_noise: max distance the center of an object can deviate from each time step
    :param diameter_minimum: minumum allowable diameter to measure a tree at
    :param diameter_maximum: maximum allowable diameter to measure a tree at
    :param limit_tracking_error: maximum percent change in a scan elligible to consider a the same object between scans.
    :return: dataframe of measured trees
    """
    tree_DF = pd.DataFrame(columns=['timestep', 'Latitude', 'Longitude', 'DBH',
                                    'Distance', 'Track', 'Timestamp', 'inView'])
    print('start extract laser')
    time, distance = extract_laser_ros(scan_file)
    print('end extract laser')
    gps_GDF = extract_gps(telem_log)

    timestamp_list = gps_GDF['Time'].to_numpy()
    latitude_list = np.zeros(timestamp_list.shape)
    longitude_list = np.zeros(timestamp_list.shape)

    coordList = gps_GDF['geometry'].to_numpy()

    idx = 0
    for point in coordList:
        longitude_list[idx] = point.x
        latitude_list[idx] = point.y
        idx += 1

    yawList = gps_GDF['Yaw'].to_numpy()

    ###################################################################################################################
    # EXPERIMENTAL
    kalmanFlag = 0  # hardcoded because it is still experimental, change to 1 to enable kalman filtering, edit telemetry
    # to pull required accelaration and velocity values from IMU

    # Convert to consistant units for dynamic/kinematics calculation

    if kalmanFlag == 1:
        timestamp_list_sec = timestamp_list / 1000000000.  # nanoepoch to sec
        # print(timestamp_list)

        meterPerDeg_lat = 111111
        # meterPerDeg_lon = 111111*np.cos(np.mean(latitude_list))
        for i in range(len(timestamp_list)):
            meterPerDeg_lon = 111111 * np.cos(math.radians(latitude_list[i]))
            latitude_list[i] = latitude_list[i] * meterPerDeg_lat  # latitide to meters
            longitude_list[i] = longitude_list[i] * meterPerDeg_lon  # longitude to meters

        lat_vel_list = lat_vel_list / 100  # cm/s to m/sec
        lon_vel_list = lon_vel_list / 100  # cm/s to m/sec
        lat_accel_list = lat_accel_list / 100  # cm/s/s to m/s/s
        lon_accel_list = lon_accel_list / 100  # cm/s/s to m/s/s

        measurement_array = np.vstack((longitude_list, latitude_list, yawList, lon_vel_list, lat_vel_list, yaw_vel_list,
                                       lon_accel_list, lat_accel_list))

        longitude_list_filt, latitude_list_filt, yawList_filt, lon_vel_list_filt, lat_vel_list_filt, yaw_vel_list_filt = filter_kalman(
            timestamp_list_sec, measurement_array)

        # Convert kalman filtered data back to original units
        yawList = yawList_filt
        for i in range(len(timestamp_list_sec)):
            latitude_list[i] = latitude_list_filt[i] / meterPerDeg_lat  # meters to latitude
            meterPerDeg_lon = 111111 * np.cos(math.radians(latitude_list[i]))
            longitude_list[i] = longitude_list_filt[i] / meterPerDeg_lon  # meters to longitude

    # END OF EXPERIMENTAL BLOCK
    ####################################################################################################################

    dbh_list, diameter_list_1, diameter_list_2, diameter_list_3 = [], [], [], []
    testcnt = 0
    tot_time = len(time)
    objcnt = 0

    # Tree Track Filtering is used (for testing and calibration)
    if level_tracking > 0:
        print("Process Filtered DBH Progress:")
        for i in range(tot_time):
            testcnt += 1
            if testcnt % 50 == 0: print(str("{0:.1f}".format(100 * testcnt / tot_time)) + "%")
            if diameter_list_2:
                diameter_list_3, center_distance_list_3, center_step_list_3 = diameter_list_1, \
                                                                              center_distance_list_1, center_step_list_1
            else:
                diameter_list_3, center_distance_list_3, center_step_list_3 = [], [], []

            if diameter_list_1:
                diameter_list_2, center_distance_list_2, center_step_list_2 = diameter_list_1, \
                                                                              center_distance_list_1, center_step_list_1
            else:
                diameter_list_2, center_distance_list_2, center_step_list_2 = [], [], []

            if dbh_list:
                diameter_list_1, center_distance_list_1, center_step_list_1 = dbh_list, \
                                                                              center_dist_list, center_step_list_
            else:
                diameter_list_1, center_distance_list_1, center_step_list_1 = [], [], []

            stepsList, distList, dbh_list, center_dist_list, center_step_list_ = \
                calculate_trees(distance[i], step_start, step_end, distance_maximum, ratio_minimum,
                                ratio_maximum, limit_noise, diameter_minimum, diameter_maximum)

            objcnt += len(dbh_list)
            lst0, lst1, lst2, lst3, isNew, inView = tree_track_filtered(dbh_list,
                                                                        center_dist_list, center_step_list_,
                                                                        diameter_list_1, center_distance_list_1,
                                                                        center_step_list_1, diameter_list_2,
                                                                        center_distance_list_2, center_step_list_2,
                                                                        diameter_list_3, center_distance_list_3,
                                                                        center_step_list_3, limit_tracking_error)

            for j in range(len(lst0)):
                sensor_latitude, sensor_longitude, sensor_yaw = alignLaserGPS(time[i],
                                                                              timestamp_list, yawList, latitude_list,
                                                                              longitude_list)

                if np.isnan(sensor_yaw) == False and isNew[j] >= level_tracking:
                    coord = determine_coord(sensor_latitude, sensor_longitude, sensor_yaw,
                                            center_dist_list[j], center_step_list_[j])

                    new_entry = pd.DataFrame(np.array([[i, coord[1], coord[0],
                                                        dbh_list[j], center_dist_list[j], isNew[j], time[i],
                                                        inView[j]]]),
                                             columns=['timestep', 'Latitude', 'Longitude', 'DBH',
                                                      'Distance', 'Track', 'Timestamp', 'inView'])

                    tree_DF = pd.concat([tree_DF, new_entry], ignore_index=True)

    # Filtering is not used (faster method with much more data)
    else:
        print("Process DBH Progress:")
        for i in range(tot_time):
            # output progress
            testcnt += 1
            if testcnt % 50 == 0: print(str("{0:.1f}".format(100 * testcnt / tot_time)) + "%")

            stepsList, distList, dbh_list, center_dist_list, center_step_list_ = \
                calculate_trees(distance[i], step_start, step_end, distance_maximum, ratio_minimum,
                                ratio_maximum, limit_noise, diameter_minimum, diameter_maximum)

            lst0, inView = tree_track_unfiltered(dbh_list, center_dist_list, center_step_list_)
            isNew = np.zeros(shape=np.shape(lst0))

            for j in range(len(lst0)):
                sensor_latitude, sensor_longitude, sensor_yaw = alignLaserGPS(time[i],
                                                                              timestamp_list, yawList, latitude_list,
                                                                              longitude_list)

                if np.isnan(sensor_yaw) == False:
                    coord = determine_coord(sensor_latitude, sensor_longitude, sensor_yaw,
                                            center_dist_list[j], center_step_list_[j])

                    new_entry = pd.DataFrame(np.array([[i, coord[1], coord[0],
                                                        dbh_list[j], center_dist_list[j], isNew[j], time[i],
                                                        inView[j]]]),
                                             columns=['timestep', 'Latitude', 'Longitude', 'DBH',
                                                      'Distance', 'Track', 'Timestamp', 'inView'])

                    # Create Pandas dataframe for raw data
                    tree_DF = pd.concat([tree_DF, new_entry], ignore_index=True)

    # save to excel spreadsheet
    wb = xw.Book(xls_raw)
    sht = wb.sheets['Sheet1']
    sht.range('A1').value = tree_DF

    return tree_DF


def resolve_tree_data(tree_DF_raw, xls_resolved, threshold_intersection_filter):
    """
    Resolve tree measurement dataframe into one where overlapping measurements are averaged into a single measurment
    :param tree_DF_raw: dataframe to resolve
    :param xls_resolved: xlsx filename to write resolved dataframe to
    :param threshold_intersection_filter: minimum number of overlapping measurements to consider a tree at a location
    :return:
    """
    pd.set_option('display.max_colwidth', -1)
    tree_DF = tree_DF_raw.sort_values(by=['DBH'], ascending=True)
    print(tree_DF_raw.head())
    tree_array = tree_DF_raw.to_numpy()
    # sort_idx = np.argsort(tree_array[:,4])
    iterCount = 0
    iterPass = False
    # meterPerDeg_lat = 111111

    resolved_diameters = np.zeros(shape=(0, 1))
    resolved_x = np.zeros(shape=(0, 1))
    resolved_y = np.zeros(shape=(0, 1))
    resolved_dist = np.zeros(shape=(0, 1))
    fullPass = False

    while fullPass == False:
        array_len0 = len(tree_array)
        inter_idx = [0]
        x0 = tree_array[0][2]
        y0 = tree_array[0][1]
        # meterPerDeg_lon = 111111*np.cos(math.radians(y0))
        # meterPerDeg_avg = (meterPerDeg_lon+meterPerDeg_lat)/2
        # r0 = tree_array[0][3]/2000/meterPerDeg_avg #convert mm to m to lat/lon
        r0 = tree_array[0][3] / 2000

        for i in range(len(tree_array)):

            if i != 0:
                x1 = tree_array[i][2]
                y1 = tree_array[i][1]
                # r1 = tree_array[i][3]/2000/meterPerDeg_avg
                r1 = tree_array[i][3] / 2000

                intersectFlag = determine_circle_intersections(x0, y0, x1, y1, r0, r1)

                if (intersectFlag == 1):
                    inter_idx.append(i)

        if inter_idx:
            inter_diameters = [tree_array[i][3] for i in inter_idx]
            inter_y = [tree_array[i][1] for i in inter_idx]
            inter_x = [tree_array[i][2] for i in inter_idx]
            # inter_dist = [tree_array[i][4]/1000/meterPerDeg_avg for i in inter_idx]
            inter_dist = [tree_array[i][4] / 1000 for i in inter_idx]

            inter_diameters = np.array(inter_diameters)
            inter_y = np.array(inter_y)
            inter_x = np.array(inter_x)
            inter_dist = np.array(inter_dist)

            if len(inter_diameters) >= threshold_intersection_filter:
                inter_diameters = reject_outliers(inter_diameters)
                inter_y = reject_outliers(inter_y)
                inter_x = reject_outliers(inter_x)
                inter_dist = reject_outliers(inter_dist)

                avg_inter_diameter = np.mean(inter_diameters)
                avg_inter_y = np.mean(inter_y)
                avg_inter_x = np.mean(inter_x)
                avg_inter_dist = np.mean(inter_dist)

                resolved_diameters = np.vstack((resolved_diameters, avg_inter_diameter))
                resolved_x = np.vstack((resolved_x, avg_inter_x))
                resolved_y = np.vstack((resolved_y, avg_inter_y))
                resolved_dist = np.vstack((resolved_dist, avg_inter_dist))

            else:
                inter_idx = [0]

                # delete used indexes from the original list
            tree_array = [i for j, i in enumerate(tree_array) if j not in inter_idx]
            array_len1 = len(tree_array)

            if array_len0 - array_len1 == 0:
                iterCount += 1

            if array_len1 <= 0:
                fullPass = True

    print(resolved_diameters)
    print(len(resolved_diameters))

    resolved_array = np.hstack((resolved_diameters, resolved_y, resolved_x, resolved_dist))

    # unused but storing for reference
    if 1 == 2:
        for i in range(len(resolved_array)):
            dbh = resolved_array[i][0]
            lat = resolved_array[i][1]
            lng = resolved_array[i][2]
            comp_list = [i]
            for j in range(len(resolved_array)):
                dbh_comp = resolved_array[j][0]
                lat_comp = resolved_array[j][1]
                lng_comp = resolved_array[j][2]
                if j != i and dbh / dbh_comp < 1.1 and dbh / dbh_comp > 0.9 \
                        and lat / lat_comp < 1.0000001 and lat / lat_comp > 0.9999999 \
                        and lng / lng_comp < 1.0000001 and lng / lng_comp > 0.9999999:
                    comp_list.append(j)
            if len(comp_list) > 0:
                # print(resolved_array)
                comp_list = np.array(comp_list)
                new_dbh = np.mean(resolved_array[comp_list][0])
                new_lat = np.mean(resolved_array[comp_list][1])
                new_lng = np.mean(resolved_array[comp_list][2])
                resolved_array = np.delete(resolved_array, (comp_list[1:]), axis=0)
                resolved_array[i][0] = new_dbh
                resolved_array[i][1] = new_lat
                resolved_array[i][2] = new_lng
                resolved_array[i][3] = 0

    tree_DF_resolved = pd.DataFrame(resolved_array, columns=['DBH', 'Latitude',
                                                             'Longitude', 'SensorDistance'])

    tree_DF_resolved = tree_DF_resolved.dropna(axis=0, how='any')

    wb = xw.Book(xls_resolved)
    sht = wb.sheets['Sheet1']
    sht.range('A1').value = tree_DF_resolved

    return tree_DF_resolved


def dbh_pipeline_stationary(stat_file, telem_log, distance_maximum, step_start, step_end):
    """
    pipeline to interpret reads from stationary_sampling.py and convert to dataframe of measured trees
    :param stat_file: stationary read file (output from stationary_sampling.py)
    :param telem_log: temelemetry log output from Mission Planner
    :param distance_maximum: max distance to measure a tree at
    :param step_start: lowest step (angle) of lidar to start measuring at
    :param step_end: highest step (angle) of lidar to start measuring at
    :return: data frame of mesure trees, geodataframe of sample areas
    """
    tree_DF = pd.DataFrame(columns=['timestep', 'Latitude', 'Longitude', 'DBH', 'Timestamp'])
    scan_time_list, dbh_list, x_list, y_list = extract_laser_stationary(stat_file)
    gps_GDF = extract_gps(telem_log)

    gps_timestamp_list = gps_GDF['Time'].to_numpy()
    latitude_list = np.zeros(gps_timestamp_list.shape)
    longitude_list = np.zeros(gps_timestamp_list.shape)
    scan_center_x_list = np.zeros(x_list.shape)
    scan_center_y_list = np.zeros(x_list.shape)
    scan_center_yaw_list = np.zeros(x_list.shape)
    coordList = gps_GDF['geometry'].to_numpy()

    idx = 0
    for point in coordList:
        longitude_list[idx] = point.x
        latitude_list[idx] = point.y
        idx += 1

    yawList = gps_GDF['Yaw'].to_numpy()
    for j in range(len(dbh_list)):
        sensor_latitude, sensor_longitude, sensor_yaw = alignLaserGPS(scan_time_list[j],
                                                                      gps_timestamp_list, yawList, latitude_list,
                                                                      longitude_list)

        scan_center_x_list[j] = sensor_longitude
        scan_center_y_list[j] = sensor_latitude
        scan_center_yaw_list[j] = sensor_yaw

        if np.isnan(sensor_yaw) == False:
            coord = determine_coord_stationary(sensor_latitude, sensor_longitude, sensor_yaw,
                                               x_list[j], y_list[j])

            newEntry = pd.DataFrame(np.array([[j, coord[1], coord[0], dbh_list[j], scan_time_list[j]]]),
                                    columns=['timestep', 'Latitude', 'Longitude', 'DBH', 'Timestamp'])

            # Create Pandas dataframe for raw data
            tree_DF = pd.concat([tree_DF, newEntry], ignore_index=True)

    scan_center_x_list = np.unique(scan_center_x_list)
    scan_center_y_list = np.unique(scan_center_y_list)
    scan_center_yaw_list = np.unique(scan_center_yaw_list)

    sample_areas_GDF = plot_sample_area(scan_center_x_list, scan_center_y_list, scan_center_yaw_list, distance_maximum,
                                        step_start, step_end)

    print(scan_center_x_list)
    print(scan_center_y_list)
    print(scan_center_yaw_list)

    return tree_DF, sample_areas_GDF
