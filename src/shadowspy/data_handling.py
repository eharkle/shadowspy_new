import logging
import datetime
import pandas as pd

from src.shadowspy.image_util import read_img_properties


def fetch_and_process_data(opt):

    use_azi_ele = False
    use_image_times = False
    if opt.azi_ele_path not in [None, 'None']:
        use_azi_ele = True

        if not opt.point_source:
            logging.error("* Can only provide azimuth&elevation when using a point source.")
            exit()

        data_list = pd.read_csv(opt.azi_ele_path).values.tolist()
        print(data_list)

    elif opt.images_index not in [None, 'None']:
        use_image_times = True
        images_index = opt.images_index
        cumindex = pd.read_csv(images_index, index_col=None)
        # get list of images from cumindex
        data_list = read_img_properties(cumindex, columns=['PRODUCT_ID', 'START_TIME', 'ortho_path'])
        # data_list['meas_path'] = [f"{opt.indir}{img}_map.tif"
        #                                   for img in data_list.PRODUCT_ID.values]
        data_list = [(row[0], row[1], row[2]) for idx, row in data_list.iterrows()]

    elif len(opt.epos_utc) > 0:
        data_list = opt.epos_utc

    else:
        start_time = datetime.datetime.strptime(opt.start_time, '%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.datetime.strptime(opt.end_time, '%Y-%m-%d %H:%M:%S.%f')
        s = pd.Series(pd.date_range(start_time, end_time, freq=f'{opt.time_step_hours}H')
                      .strftime('%Y-%m-%d %H:%M:%S.%f'))
        data_list = s.values.tolist()

    return data_list, use_azi_ele, use_image_times
