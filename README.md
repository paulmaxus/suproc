# suproc

A Python library for processing smartphone usage data.

## Installation

To install the library, run the following command:
```bash
pip install suproc
```
## Usage

### Loading Data

To load data, create a `DataLoader` object and iterate over it:
```python
from suproc import DataLoader

dl = DataLoader(path_to_trace_data="./data/trace_data", path_to_esm_file="./data/esm.csv")
for data in dl:
    # process data
```

Load a single data file like this
```python
data_first = next(dl)  # first participant
```

### Screen Time

To get the screen time for a participant, create a `ScreenTime` object:
```python
from suproc import ScreenTime

st = ScreenTime(data_first)
```
You can then aggregate the screen time by period:
```python
st_day = st.by_period(period="day")
```
And plot it:
```python
st_day.plot()
```
### App Usage

To get the app usage for a participant, create an `AppUsage` object:
```python
from suproc import AppUsage

au = AppUsage(data_first)
```
You can then aggregate the app usage by period:
```python
au_hour = au.by_period(period="hour")
```
And plot it:
```python
au_hour.plot()
```