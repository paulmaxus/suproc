import xml.etree.ElementTree as ET
import pandas as pd
import datetime
import zipfile

from pathlib import Path


class DataLoader:
    def __init__(self, path_to_trace_data="./data/trace_data", path_to_esm_file="./data/esm.csv", esm_all_columns=False):
        self.data_folder = Path(path_to_trace_data)
        self.folders = [x for x in self.data_folder.glob("**/*") if x.is_dir()]
        self.esm = self._load_esm(Path(path_to_esm_file), all_columns=esm_all_columns)
        self.n = len(self.folders)
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.n:
            # For each participant, create esm subset
            try:
                esm_subset = self.esm.copy()[self.esm.id == self.folders[self.current].stem].reset_index(drop=True)
            except KeyError:
                esm_subset = None
                print(f"Participant {self.folders[self.current].stem} not found in esm")
            result = Data(self.folders[self.current], esm_subset) 
            self.current += 1
            return result
        else:
            raise StopIteration
        
    def _esm_convert_datetime(self, date):
        try:
            return datetime.datetime.strptime(date, "%d.%m.%y%H:%M:%S")
        except ValueError:
            return datetime.datetime.strptime(date, "%Y-%m-%d%H:%M:%S")
        
    def _load_esm(self, esm_file, all_columns):
        esm_dtypes = {
            'id': 'O',
            'formc': 'float64',
            'form_finish_date': 'O',
            'form_finish_time': 'O',
        }
        df = pd.read_csv(esm_file, delimiter=",", usecols=esm_dtypes.keys() if not all_columns else None, dtype=esm_dtypes)
        # preprocessing
        df = df[~pd.isna(df.formc)]  # remove missing forms
        #df["form_finish_datetime"] = df.form_finish_date.str.zfill(6) + df.form_finish_time
        df["form_finish_datetime"] = df.form_finish_date + df.form_finish_time
        df["form_finish_datetime"] = df.form_finish_datetime.apply(self._esm_convert_datetime)
        return df


class Data:

    def __init__(self, folder, esm=None):
        self.folder = folder
        self.id = folder.stem
        self.esm = esm
        self.config = None
        self.app_usage = None
        self.display_on = None
        self._load()

    def _load(self, logs=False):
        if logs:
            self._load_logs(self._read_files)
        else:
            with open(self.folder / "unisens.xml") as file1, open(self.folder / "AppUsage.csv") as file2, open(self.folder / "DisplayOn.csv") as file3:
                self._read_files(file1, file2, file3)

    def _read_files(self, file1, file2, file3):
        self.config = ET.parse(file1).getroot().attrib
        self.app_usage = pd.read_csv(file2, header=None, names=["timestamp","app","event"])
        self.display_on = pd.read_csv(file3, header=None, names=["timestamp", "is_on"])

    def _load_logs(self, file_reader):
        # In some cases, there are zipped log files
        log_folders = list(self.folder.glob("*_log.zip"))
        if len(log_folders) > 1:
            print("More than 1 log file folders, picking first")
        with zipfile.ZipFile(log_folders[0], 'r') as zip_ref:
            with zip_ref.open("unisens.xml") as file1, zip_ref.open("AppUsage.csv") as file2, zip_ref.open("DisplayOn.csv") as file3:
                file_reader(file1, file2, file3)
    

class Processor:
    def __init__(self, data: Data):
        self.esm = data.esm.copy()[["formc","form_finish_datetime"]] if not data.esm is None else None
        self.ts_start = datetime.datetime.strptime(data.config["timestampStart"], '%Y-%m-%dT%H:%M:%S.%f')

    def _filter_by_hours_prior(self, df_in, event_time, hours=2):
        df = df_in.copy()
        df = df[(df.datetime_start >= event_time - datetime.timedelta(hours=hours)) & (df.datetime_start < event_time)]
        return df
    
    def by_esm(self, df_in, hours_prior=2, group=None):
        if self.esm is None:
            return
        by_esm = []
        for formc, ffdt in zip(self.esm.formc, self.esm.form_finish_datetime):
            df = self._filter_by_hours_prior(df_in, event_time=ffdt, hours=hours_prior)
            df["duration"] = df.datetime_end.apply(lambda x: min(x, ffdt)) - df.datetime_start  # maximum duration is until end of esm
            # Unit should be minutes
            df["duration"] = df.duration.apply(lambda x: x.total_seconds() / 60)
            if group is not None:
                df = df.groupby(group).duration.sum().reset_index()
                by_esm.append((formc, df))
            else:
                by_esm.append((formc, df.duration.sum()))
        return by_esm
    
    def by_period(self, df_in, period="day", group=None):

        period_map = {
            "day": "D",
            "weekday": "D",
            "hour": "h",
        }
        if period not in period_map:
            raise ValueError("Period must be one of day, weekday or hour")
        freq = period_map[period]

        df = df_in.copy()
        s = []
        for x,y in zip(df.datetime_start, df.datetime_end):
            pr = pd.period_range(x, y, freq=freq)
            s.append(pr)
        df["period"] = s

        df = df.explode("period")

        df["datetime_start_period"] = df.period.apply(lambda x: x.to_timestamp(freq=freq))
        df["datetime_end_period"] = df.period + 1
        df["datetime_end_period"] = df.datetime_end_period.apply(lambda x: x.to_timestamp(freq=freq))

        df["new_datetime_start"] = df[['datetime_start','datetime_start_period']].max(axis=1)
        df["new_datetime_end"] = df[['datetime_end','datetime_end_period']].min(axis=1)
        df["duration"] = df.new_datetime_end - df.new_datetime_start

        match period: 
            case "day":
                df["by_period"] = df.period
            case "weekday":
                df["by_period"] = df.period.apply(lambda x: x.dayofweek)
            case "hour":
                df["by_period"] = df.period.apply(lambda x: x.hour)

        extra_group = [group] if group is not None else []
        df =  df.groupby(["by_period"] + extra_group).duration.sum().reset_index()
        # Unit in minutes is required
        df["duration"] = df.duration.apply(lambda x: x.total_seconds() / 60)
        return df


class ScreenTime(Processor):

    def __init__(self, data: Data):
        super().__init__(data)
        self.screen_time = self._preprocess_display_on(data.display_on)

    def _preprocess_display_on(self, display_on):
        df = display_on.copy()
        df["datetime_start"] = df.timestamp.apply(lambda x: self.ts_start + datetime.timedelta(seconds=x))
        df["datetime_end"] = df.datetime_start.shift(-1)
        # Remove last event as we don't know when it ended
        df = df.iloc[:-1,:]   
        # Remove display off events
        df = df[df.is_on==1]
        
        return df
    
    def by_esm(self, hours_prior=2):
        screen_time = super().by_esm(self.screen_time, hours_prior=hours_prior)
        return pd.DataFrame(screen_time, columns=["formc","screen_time"])
    
    def by_period(self, period="day"):
        screen_time = super().by_period(self.screen_time, period=period)
        # For plotting, set index
        return screen_time.set_index("by_period")
    
class AppUsage(Processor):

    def __init__(self, data: Data):
        super().__init__(data)
        self.platforms = [
            "facebook", 
            "messenger", 
            "instagram", 
            "twitter", 
            "snapchat", 
            "tiktok", 
            "youtube", 
            "whatsapp", 
            "telegram", 
            "viber", 
            "pinterest", 
            "linkedin"
        ]
        self.app_usage = self._preprocess_app_usage(data.app_usage)


    def _preprocess_app_usage(self, app_usage):
        df = app_usage.copy()
        df["datetime_start"] = df.timestamp.apply(lambda x: self.ts_start + datetime.timedelta(seconds=x))
        df["datetime_end"] = df.datetime_start.shift(-1)
        # Remove last event as we don't know when it ended
        df = df.iloc[:-1,:]
        # Add platform
        df['platform'] = df.event.str.extract(f"({'|'.join(self.platforms)})")
        # Remove events with no platform
        df = df[~pd.isna(df.platform)]
        
        return df
    
    def by_esm(self, hours_prior=2):
        app_times = super().by_esm(self.app_usage, hours_prior=hours_prior, group="platform")
        app_time = pd.DataFrame()
        for formc, df in app_times:
            df["formc"] = formc
            app_time = pd.concat([app_time, df])
        return app_time.pivot(index="formc", columns="platform", values="duration")
    
    def by_period(self, period="day"):
        app_time = super().by_period(self.app_usage, period=period, group="platform")
        # For plotting, pivot and set index
        return app_time.reset_index().pivot(index="by_period", columns="platform", values="duration")
    
