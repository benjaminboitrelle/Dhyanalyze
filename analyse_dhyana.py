# -*- coding: utf-8 -*-

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def get_metadata(metadata):
    ''' Open metada file and store exposure time to corresponding images
    '''

    with open(metadata, 'r') as meta_file:
        file_content = meta_file.read().splitlines()

        # data looks like this: <exposurer_time>  <file_prefix>
        file_content = [s.split("\t") for s in file_content]
        for i, string in enumerate(file_content):
            try:
                string[0] = float(string[0])
            except ValueError:
                if string == ['']:
                    # remove empty lines
                    del file_content[i]
                else:
                    raise
    return sorted(file_content)


def get_meta(meta_name):
    ''' Return a matrix of the metada containing the images associated
        to an exposure time
    '''

    metadata = get_metadata(meta_name)
    meta = [[metadata[time][0],
            (metadata[time-1][1].strip(),
             metadata[time][1].strip())]
            for time, file in enumerate(metadata)
            if metadata[time-1][0] == metadata[time][0]]
    return meta


def get_images(meta, exposure_time, roi):
    ''' For a given exposure time return a stack of arrays corresponding
        to the 2 images taken
    '''
    img1 = np.array(plt.imread(meta[exposure_time][1][0] + ".tif"),
                    dtype=float)
    img2 = np.array(plt.imread(meta[exposure_time][1][1] + ".tif"),
                    dtype=float)
    img = np.dstack((img1[roi:-roi, roi:-roi],
                     img2[roi:-roi, roi:-roi]))
#    img = img1 - img2

#    return img1[roi:-roi, roi:-roi], img2[roi:-roi, roi:-roi]
    return img


def get_images_diff(meta, exposure_time, roi):
    '''Blabla
    '''

    img1 = np.array(plt.imread(meta[exposure_time][1][0] + ".tif"),
                    dtype=float)
    img2 = np.array(plt.imread(meta[exposure_time][1][1] + ".tif"),
                    dtype=float)
    img = (img1 - img2) / np.sqrt(2)

    return img[roi:-roi, roi:-roi]


def get_images_sum(meta, exposure_time, roi):
    ''' Blabla
    '''

    img1 = np.array(plt.imread(meta[exposure_time][1][0] + ".tif"),
                    dtype=float)
    img2 = np.array(plt.imread(meta[exposure_time][1][1] + ".tif"),
                    dtype=float)
    img = (img1 + img2) / 2

    return img[roi:-roi, roi:-roi]


def get_images_PTC_sum(input_dir_path, exposure_time, roi):
    ''' Blabla
    '''

    # Get dark images
    os.chdir(input_dir_path + '/Dark')
    meta = get_meta('metadata.txt')
    img_dark1 = np.array(plt.imread(meta[exposure_time][1][0] + '.tif'),
                         dtype=float)
    img_dark2 = np.array(plt.imread(meta[exposure_time][1][1] + '.tif'),
                         dtype=float)

    # Get light images
    os.chdir(input_dir_path + '/Light')
    meta = get_meta('metadata.txt')
    img_light1 = np.array(plt.imread(meta[exposure_time][1][0] + '.tif'),
                          dtype=float)
    img_light2 = np.array(plt.imread(meta[exposure_time][1][1] + '.tif'),
                          dtype=float)

    # Remove Dark from Light
    img1 = img_light1 - img_dark1
    img2 = img_light2 - img_dark2

    img_sum = (img1 + img2) / 2

    return img_sum[roi:-roi, roi:-roi]


def get_images_PTC_diff(input_dir_path, exposure_time, roi):
    ''' Blabla
    '''

    # Get dark images
    os.chdir(input_dir_path + '/Dark')
    meta = get_meta('metadata.txt')
    img_dark1 = np.array(plt.imread(meta[exposure_time][1][0] + '.tif'),
                         dtype=float)
    img_dark2 = np.array(plt.imread(meta[exposure_time][1][1] + '.tif'),
                         dtype=float)

    # Get light images
    os.chdir(input_dir_path + '/Light')
    meta = get_meta('metadata.txt')
    img_light1 = np.array(plt.imread(meta[exposure_time][1][0] + '.tif'),
                          dtype=float)
    img_light2 = np.array(plt.imread(meta[exposure_time][1][1] + '.tif'),
                          dtype=float)

    # Remove Dark from Light
    img1 = img_light1 - img_dark1
    img2 = img_light2 - img_dark2

    img_diff = (img1 - img2) / np.sqrt(2)

    return img_diff[roi:-roi, roi:-roi]


def plot_result(x, y, m, c, out_fname, title, label_x, label_y):
    ''' Show plot of data and data fitted
    '''
    plt.plot(x, y, '.', x, m*x+c, '-')
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    ax = plt.axes()
    if measurement_type == 'PTC':
        plt.text(0.8, 0.07,
                 'slope :  %.2f ADU/e- \n offset :  %.2f ADU' % (m, c),
                 fontsize=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes)

    elif measurement_type == 'Dark':
        plt.text(0.8, 0.07,
                 'slope :  %.2f ADU/s \n offset :  %.2f ADU' % (m*1000, c),
                 fontsize=12,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes)
    plt.savefig(out_fname)
    plt.show()
    plt.close()


def get_fit_parameters(x, y):
    ''' Linear fitting of data

        Output:
            slope, offset, residuals
    '''

    m, c = np.polyfit(x, y, 1)

    return m, c


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        dest='input_dir_path',
                        type=str,
                        help='Path to the input directory')

    parser.add_argument('--output',
                        dest='output_dir_path',
                        type=str,
                        help='Path to the output directory')

    parser.add_argument('--type',
                        dest='measurement_type',
                        type=str,
                        help='Type of measurements: \n Dark \n PTC')

    parser.add_argument('--roi',
                        dest='roi',
                        type=int,
                        nargs='?',
                        default=5,
                        help='Select a region of interest. This ROI is \
                              squared and centered. \n By default: \
                              10 pixels are excluded in vertically and \
                              horizontally.')

    args = parser.parse_args()

    output_dir_path = args.output_dir_path
    measurement_type = args.measurement_type
    roi = args.roi

    if measurement_type == 'Dark':
        input_dir_path = args.input_dir_path
        os.chdir(input_dir_path)
        meta = get_meta('metadata.txt')
        time_range = len(meta)

# Create a stack of images depending on the exposure time:
#        img.shape() = (snap, nb_pixel_x, nb_pixel_y)

        img = np.asarray([get_images(meta, i, roi)
                          for i in range(time_range)])

        img_sum = np.sum(img, axis=3) / 2
        img_diff = (np.diff(img, axis=3) / np.sqrt(2)).reshape(time_range,
                                                               500,
                                                               500)
        avr_mat = np.mean(img_sum, axis=(1, 2))
        std_mat = np.std(img_diff, axis=(1, 2))
        exp_time = np.asarray([meta[time][0] for time in range(time_range)])
        slope_avr, offset_avr = get_fit_parameters(exp_time, avr_mat)
        slope_std, offset_std = get_fit_parameters(exp_time, std_mat)

        dark_current = np.mean(avr_mat)
        min_std = np.min(std_mat)
        max_std = np.max(std_mat)
        dark_temp_noise = np.mean(std_mat)

        print("Dark current {} ADU".format(dark_current))
        print("Std min {} ADU".format(min_std))
        print("Std max {} ADU".format(max_std))
        print("Dark temporal noise {} ADU".format(dark_temp_noise))

        os.chdir(output_dir_path)

        plot_result(exp_time,
                    avr_mat,
                    slope_avr,
                    offset_avr,
                    'meanVsExposure_roi'+str(roi)+'.png',
                    "Mean of the dark VS exposure time",
                    "Exposure time [ms]",
                    "Mean [ADU]")

        plot_result(exp_time,
                    std_mat,
                    slope_std,
                    offset_std,
                    'standardDeviationVsExposure_roi'+str(roi)+'.png',
                    "Standard deviation of the dark VS exposure time",
                    "Exposure time [ms]",
                    "Standard deviation [ADU]")

    if measurement_type == 'PTC':
        input_dir_path = args.input_dir_path

        input_dir_path_light = input_dir_path + "/Light"
        input_dir_path_dark = input_dir_path + "/Dark"
        os.chdir(input_dir_path_light)
        meta_light = get_meta("metadata.txt")
        os.chdir(input_dir_path_dark)
        meta_dark = get_meta("metadata.txt")
        time_range = len(meta_light)

        img_sum = np.array([get_images_PTC_sum(input_dir_path,
                                               snap,
                                               roi)
                            for snap in range(time_range)])
        img_diff = np.array([get_images_PTC_diff(input_dir_path,
                                                 snap,
                                                 roi)
                             for snap in range(time_range)])

        img_std = np.var(img_diff, axis=(1, 2), ddof=1) - 6*6
        img_avr = np.mean(img_sum, axis=(1, 2))

        fit_bounds = np.where(np.logical_and(img_avr > 0,
                                             img_avr < 50000))
        slope, offset = get_fit_parameters(img_avr[fit_bounds],
                                           img_std[fit_bounds])
        print("Max of variance {} ADU^2".format(np.max(img_std)))
        saturation_adu = img_avr[img_std == np.max(img_std)]
        saturation_electron = saturation_adu / slope
        max_snr = np.log2(np.sqrt(saturation_electron))
        print("Saturation {} ADU".format(saturation_adu))
        print("Saturation {} electrons".format(saturation_electron))
        print("Max SNR {}".format(max_snr))
        print(slope, offset)
        os.chdir(output_dir_path)

        plot_result(img_avr,
                    img_std,
                    slope,
                    offset,
                    'stdVsMeanLight_roi'+str(roi)+'.png',
                    'Transfer curve',
                    'Mean value [ADU]',
                    'Variance [ADU^2]')
