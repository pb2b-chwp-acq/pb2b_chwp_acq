# DetectorSampleTimes: (Array)
# RawTimestreams_I: PB20.13.39_Comb26Ch09: (Data)
# RawTimestreams_Q: PB20.13.39_Comb26Ch09: (Data)

import os
import numpy as np
from spt3g import core
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CHWPDataDemod:
    chwp_data_dir: str

    @classmethod
    def __call__(frame):
        if frame.type == core.G3FrameType.Scan:
            self.load_chwp_data(frame)

    @classmethod
    def load_chwp_data(frame):
        times = np.array(frame['DetectorSampleTimes'])

        start_day, start_sec, _ = self.destring_date(times[0])
        end_day, end_sec, _ = self.destring_date(times[-1])

        chwp_g3_files = find_files(start_day, start_sec, end_day, end_sec)

        def find_files(start_d, start_s, end_d, end_s):
            dates = []
            data_files = []
            for root, dirs, files in os.walk(self.chwp_data_dir):
                if root == self.chwp_data_dir:
                    _dirs = np.array([int(_dir) for _dir in dirs])
                    
                    # Add the day just before the start date
                    start_dif = _dirs[np.where((_dirs-int(start_d)) < 0)]
                    if len(start_dif) != 0:
                        dates.append(max(start_dif))
                    # Add the days between the start and end dates
                    for _dir in _dirs:
                        if _dir >= int(start_d) and _dir <= int(end_D):
                            dates.append(_dir)
                    # Add the day after the end date
                    end_dif = _dirs[np.where((_dirs-int(end_d)) > 0)]
                    if len(end_dif) != 0:
                        dates.append(min(end_dif))

                elif int(root.split('/')[-1]) in dates:
                    for _file in files:
                        data_files.append(_file.split('.')[0].split('_'))

            _times = [int(_file[0] + _file[1]) for _file in data_files]
            
            start_time = int(str(start_d) + str(start_s))
            end_time = int(str(end_d) + str(end_s))
            
            start_file_time = max((_times - start_time) < 0)
            end_file_time = min((_times - end_time) > 0)

            mask = np.logical_or(np.logical_and(np.where(_times >= start_time), 
                                                np.where(_time <= end_time)),
                                 np.where(_times == start_file_time),
                                 np.where(_times == end_file_time))

            data_files = np.array(data_files)[mask]
            for i, _file in enumerate(data_files):
                data_files[i] = os.path.join(self.chwp_data_dir, _file[0],
                                             _file[0] + '_' + _file[1] + '.g3')

            return data_files

    @staticmethod
    def destring_date(date_str, raw_utc = False):
        _date_str = date_str.split('.')
        dt = datetime.strptime(_date_str[0],'%d-%b-%Y:%H:%M:%S')
        day = dt.strftime('%Y%m%d')
        
        if not raw_utc:
            sec = dt.strftime('%H%M%S')
        else:
            sec = datetime.timestamp(dt) + int(_date_str[1])*10**-9
        
        return day, sec
