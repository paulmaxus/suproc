import xml.etree.ElementTree as ET
import pandas as pd
import datetime
import zipfile

from pathlib import Path
from datetime import timedelta
from typing import Dict


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
            if df.empty:
                continue
            df["time"] = df.datetime_end.apply(lambda x: min(x, ffdt)) - df.datetime_start  # maximum is until end of esm
            # Unit should be minutes
            try:
                df["time"] = df.time.dt.total_seconds() / 60
            except AttributeError:
                print(f"No total_seconds() method for {df['time']}")
                raise AttributeError
            #df["time"] = df.time.dt.total_seconds() / 60
            # Convert type to float, otherwise sum() might result in wrong type for time=0
            #df["time"] = df.time.astype(float)
            if group:
                df = df.groupby(group).time.sum().reset_index()
                by_esm.append((formc, df))
            else:
                by_esm.append((formc, df.time.sum()))
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
        df["time"] = df.new_datetime_end - df.new_datetime_start

        match period: 
            case "day":
                df["by_period"] = df.period
            case "weekday":
                df["by_period"] = df.period.apply(lambda x: x.dayofweek)
            case "hour":
                df["by_period"] = df.period.apply(lambda x: x.hour)

        extra_group = [group] if group is not None else []
        df =  df.groupby(["by_period"] + extra_group).time.sum().reset_index()
        # Unit in minutes is required
        df["time"] = df.time.apply(lambda x: x.total_seconds() / 60)
        return df


class ScreenTime(Processor):

    def __init__(self, data: Data):
        super().__init__(data)
        self.screen_time = self._preprocess_display_on(data.display_on)

    def _preprocess_display_on(self, display_on):
        df = display_on.copy()
        # sometimes timestamps are not unique!
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        df = df.sort_values(by="timestamp", ascending=True)
        df["datetime_start"] = df.timestamp.apply(lambda x: self.ts_start + datetime.timedelta(seconds=x))
        df["datetime_end"] = df.datetime_start.shift(-1)
        # Remove last event as we don't know when it ended
        df = df.iloc[:-1,:]   
        # Remove display off events
        df = df[df.is_on==1]
        df = df.reset_index(drop=True)
        
        return df
    
    def by_esm(self, hours_prior=2):
        screen_time = super().by_esm(self.screen_time, hours_prior=hours_prior)
        if not screen_time:
            return pd.DataFrame()
        return pd.DataFrame(screen_time, columns=["formc","screentime"])
    
    def by_period(self, period="day"):
        screen_time = super().by_period(self.screen_time, period=period)
        screen_time.rename(columns={"time": "screentime"}, inplace=True)
        # For plotting, set index
        return screen_time.set_index("by_period")
    
class AppUsage(Processor):

    def __init__(self, data: Data, all_apps=False):
        super().__init__(data)
        self.platforms = [
            "facebook.katana", 
            "facebook.orca",  # messenger 
            "instagram",  # includes umatech.instagram, get.instagram.follower
            "twitter", 
            "snapchat", 
            "tiktok", 
            "youtube",  # includes vanced.android.youtube
            "whatsapp",  # includes whatsapp.w4b
            "telegram", 
            "viber", 
            "pinterest", 
            "linkedin"
        ]
        self.app_usage = self._preprocess_app_usage(data.app_usage, all_apps=all_apps)


    def _preprocess_app_usage(self, app_usage, all_apps, check_platforms=False):
        df = app_usage.copy()
        # sometimes timestamps are not unique, but that's okay for app usage
        df = df.sort_values(by="timestamp", ascending=True)  # just make sure they are sorted
        df["datetime_start"] = df.timestamp.apply(lambda x: self.ts_start + datetime.timedelta(seconds=x))
        df["datetime_end"] = df.datetime_start.shift(-1) 
        # Remove last event as we don't know when it ended
        df = df.iloc[:-1,:]
        # Remove screen_on events: if app usage is missing then screen_on would count torwards app usage
        df = df[df.event != "com.android/android.intent.action.SCREEN_ON"]
        # Add platform
        # App name is in first part
        df["event"] = df.event.str.split("/").str[0]
        df['platform'] = df.event.str.extract(f"({'|'.join(self.platforms)})")
        if check_platforms:
            # Check whether platform events are unique
            events = df.groupby("platform").event
            if events.nunique().max() > 1: 
                print(f"platform events are not unique: {events.unique()}")
        # Remove events with no platform
        if not all_apps:
            df = df[~pd.isna(df.platform)]
        
        return df
    
    def by_esm(self, hours_prior=2):
        app_times = super().by_esm(self.app_usage, hours_prior=hours_prior, group="platform")
        app_time = pd.DataFrame()
        for formc, df in app_times:
            df["formc"] = formc
            app_time = pd.concat([app_time, df])
        if app_time.empty:
            return app_time
        else:
            return app_time.pivot(index="formc", columns="platform", values="time")
    
    def by_period(self, period="day"):
        app_time = super().by_period(self.app_usage, period=period, group="platform")
        # For plotting, pivot and set index
        return app_time.reset_index().pivot(index="by_period", columns="platform", values="time")
    
class MissingAppUsage(Processor):
    
    def __init__(self, data: Data):
        super().__init__(data)
        self.missing_app_usage = self._get_missing_app_usage(ScreenTime(data).screen_time, AppUsage(data, all_apps=True).app_usage)

    def _get_missing_app_usage(self, st, au):
        missing_periods = []
        for _, row in st.iterrows():
            missing_periods.extend(self._get_missing_app_usage_by_period(row["datetime_start"], row["datetime_end"], au))
        return pd.DataFrame(missing_periods, columns=["datetime_start", "datetime_end"])

    def _get_missing_app_usage_by_period(self, st_start, st_end, au):
        missing_periods = []
        au_periods = au[(au["datetime_start"] < st_end) & (au["datetime_end"] > st_start)]
        if au_periods.empty:
            # No app usage, add the entire period
            missing_periods.append((st_start, st_end))
        else:
             # Process overlapping periods
            current_start = st_start
            for _, row in au_periods.iterrows():
                # If there's a gap before this period, add it
                if current_start < row["datetime_start"]:
                    missing_periods.append((current_start, row["datetime_start"]))
                current_start = max(current_start, row["datetime_end"])
            # Add any remaining period after the last overlap
            if current_start < st_end:
                missing_periods.append((current_start, st_end)) 
        return missing_periods
    
    def by_esm(self, hours_prior=2):
        missing_app_usage = super().by_esm(self.missing_app_usage, hours_prior=hours_prior)
        if not missing_app_usage:
            return pd.DataFrame()
        return pd.DataFrame(missing_app_usage, columns=["formc","missing_app_usage"])
    
    def by_period(self, period="day"):
        missing_app_usage = super().by_period(self.missing_app_usage, period=period)
        missing_app_usage.rename(columns={"time": "missing_app_usage"}, inplace=True)
        # For plotting, set index
        return missing_app_usage.set_index("by_period")

class Diagnostics:

    def __init__(self, screen_time, app_usage):
        self.screen_time = screen_time.screen_time.copy()
        self.app_usage = app_usage.app_usage.copy()
        self.screen_time["timedelta_screentime"] = self.screen_time.apply(lambda row: (row["datetime_end"] - row["datetime_start"]), axis=1)
        self.screen_time["timedelta_app_usage"] = self.screen_time.apply(lambda row: self._calculate_app_usage(row["datetime_start"], row["datetime_end"], self.app_usage), axis=1)

    def _calculate_app_usage(self, st_start, st_end, au):
        au_sub = au[(au["datetime_start"] < st_end) & (au["datetime_end"] > st_start)]
        td = au_sub.apply(lambda row: min(row["datetime_end"], st_end) - max(row["datetime_start"], st_start), axis=1)
        return td.sum() if not td.empty else timedelta(0)
    
    def calculate_missing_app_usage(self):
        return (self.screen_time.timedelta_screentime - self.screen_time.timedelta_app_usage).sum()
    
    def get_missing_app_usage_timestamps(self):
        return self.screen_time[(self.screen_time.timedelta_screentime - self.screen_time.timedelta_app_usage) != timedelta(0)].timestamp.to_list()
    
    def get_overlapping_screen_time(self):
        # screen_on periods can't overlap
        return self.screen_time[self.screen_time['datetime_end'] > self.screen_time['datetime_start'].shift(-1)].timestamp.to_list()
        
    def diagnostics(self):
        return {
            "missing_app_usage_total": self.calculate_missing_app_usage(),
            "missing_app_usage_timestamps": self.get_missing_app_usage_timestamps(),
            "overlapping_screen_time": self.get_overlapping_screen_time()
        }
    
class DynamicOutput:
    def __init__(self):
        self._dataframes: Dict[str, pd.DataFrame] = {}
    
    def __getattr__(self, name: str):
        if name in self._dataframes:
            return self._dataframes[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def add_dataframe(self, name: str, df: pd.DataFrame):
        self._dataframes[name] = df

class Pipeline:

    def __init__(self, data_loader: DataLoader, processors: list[Processor], period="day", hours_prior=2):
        self.data_loader = data_loader
        self.processors = processors
        self.period = period
        self.hours_prior = hours_prior

    def run(self):
        output = DynamicOutput()
        dfp_all = pd.DataFrame()
        dfe_all = pd.DataFrame()
        for data in self.data_loader:
            dfp = pd.DataFrame()
            dfe = pd.DataFrame()
            for processor in self.processors:
                p = processor(data)
                # by period
                bp = p.by_period(period=self.period).reset_index()
                if dfp.empty:
                    dfp = bp
                elif not bp.empty:
                    dfp = pd.merge(dfp, bp, how="outer", on="by_period")
                # by esm
                be = p.by_esm(hours_prior=self.hours_prior)
                if dfe.empty:
                    dfe = be
                elif not be.empty:
                    dfe = pd.merge(dfe, be, how="outer", on="formc")
            if not dfp.empty:
                dfp.insert(0, "id", data.id)
                dfp_all = pd.concat([dfp_all, dfp])
            if not dfe.empty:
                dfe.insert(0, "id", data.id)
                dfe_all = pd.concat([dfe_all, dfe])
        output.add_dataframe('by_period', dfp_all)
        output.add_dataframe('by_esm', dfe_all)
        return output
