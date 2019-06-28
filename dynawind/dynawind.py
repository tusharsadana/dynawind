"""
dynawind.dynawind
===================
Overview
-------------------
The dynawind.dynawind module is the main modules that contains all functions to interact with timeseries and tdms
functions.

Definitions
-------------------
"""
import datetime
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import pytz
from pylookup import pylookup as lookup
from pytz import utc

import dynawind.mpe as mpe

# def parseCLIarguments(argv):
# import getopt

# filetype = 'tbd'  # To be determined
# site = None
# location = None

# opts, args = getopt.getopt(argv, 'f:s:l:t:', ['file=',
# 'site=',
# 'location=',
# 'type='])
# for opt, arg in opts:
# if opt in ('-f', '--file'):
# filePath = arg
# elif opt in ('-s', '--site'):
# site = arg
# elif opt in ('-l', '--location'):
# location = arg
# elif opt in ('-t', '--type'):
# filetype = arg
# # %%
# processFile(filePath, site=site, location=location, filetype=filetype)
def process_period(location, filetype, start_dt, stop_dt, root=r"\\192.168.119.14"):
    """ Process a period of data of at a location of a filetype """

    from progressbar import ProgressBar
    from dynawind.db import make_dt_list

    dt_list = make_dt_list(start_dt, stop_dt)
    pbar = ProgressBar()

    for dt in pbar(dt_list):
        path = getTDMSpath(dt, filetype, location, root=root)
        processFile(path)


# # %%

def load_postprocessing_config(site):
    """ Opens the post-processing configuration of a site """
    import configparser
    import pkg_resources

    resource_package = __name__

    config = configparser.ConfigParser()
    resource_path = "/".join(
        ("config", site.lower(), site.lower() + "_postprocessing.ini")
    )
    ini_file = pkg_resources.resource_filename(resource_package, resource_path)
    config.read(ini_file)

    return config


def processFile(filepath, site=None, location=None, file_format=None):
    """ This is the main function to process a file into the database.
    The function does serves for both tdms files and other types, e.g. SCADA csv files.


    :param filepath: the path to the file to process
    :param site: site of origin, e.g. 'belwind'. This is only relevant for non tdms files
    :param location: If not None (default) the configuration for this location is used, else the site configuration is used
    :param file_format:If the file is not a TDMS use this attribute to specify the type. Based on this information the site-specific code for this filetype will be triggered, e.g. SCADA.

    """
    import importlib
    import os.path
    from math import isnan

    if os.path.isfile(filepath):
        pass
    else:
        print(filepath + " : File does not exist")
        return
    if file_format is None:
        if filepath[-4:] == "tdms":
            file_format = "tdms"
    # %%
    if file_format == "tdms":
        # Default processing
        signals = readTDMS(filepath)
        if signals:
            # Default : Single entry every 10 minutes
            site = [signals[0].site]
            location = [signals[0].location]
        else:
            print(filepath + " : File is empty")
            return
    else:
        signals = None
        """
        # A custom function will be triggered based on site, location and type
        Output is a list of dict
        (by using list a single file can generate multiple records)
        # ! Site, location and type need to be specified
        """
        if location is None:
            customMethod = getattr(
                importlib.import_module(
                    "dynawind.config." + site.lower() + ".custom_" + site.lower()
                ),
                file_format.upper() + "_" + site.lower(),
            )
        else:
            customMethod = getattr(
                importlib.import_module(
                    "dynawind.config." + site.lower() + ".custom_" + site.lower()
                ),
                file_format.upper() + "_" + location.lower(),
            )
        dt, site, location, data = customMethod(filepath)

    # %% load site config file

    config = load_postprocessing_config(site[0])
    timescale = 600
    if "general" in config:
        timescale = int(config["general"].get("timescale", 600))
    if signals:
        if timescale == 600:
            # Default behaviour every ten minutes
            data = [stats2dict(signals)]
            dt = [signals[0].timestamp + datetime.timedelta(minutes=10)]
        else:
            # Generate HF (more than every 10 minutes)
            dt, site, location, data = HF_stats2dict(signals, timescale)
    # Step 2 : write to temp file
    json_list = []
    for t, s, l, d in list(zip(dt, site, location, data)):
        clean_d = {k: d[k] for k in d if isinstance(d[k], str) or not isnan(d[k])}
        json_list.extend([write2json(
            t,
            s,
            l,
            clean_d,
            root=config["json"]["jsonFolder"],
            fileext=config["json"]["tmpExtension"],
        )])
    return json_list


def write2json(dt, site, location, data, root="", fileext=".json"):
    """ Writes data to a json file

    :param dt: timestamp
    :type dt: :class:`datetime.datetime`
    :param site: site associated with the data, e.g Belwind
    :param location: location associated with the data, e.g BBC01
    :param data: the actual data to store
    :type data: dict.
    :param root: root of the json files
    :param fileext: file extension to store the data is, defaults .json. Note that this has no effect on the formatting of the data which will always be json style.
    :returns: the path to the json file
    """
    import json
    from dynawind.db import returnJSONfilePath, clean_dict

    data = clean_dict(data)  # Clean up dict for correct processing in JSON

    jsonPath = returnJSONfilePath(dt, site, location, root=root, fileext=fileext)
    jsonFile = open(jsonPath, "r")
    record = json.load(jsonFile)
    jsonFile.close()
    for key in data.keys():
        record[0][key] = data[key]
    jsonFile = open(jsonPath, "w")
    json.dump(record, jsonFile, indent=2)
    jsonFile.close()

    return jsonPath


class Series(object):
    def __init__(self, paths):
        # Outdated, was based on first concepts
        self.sourcePaths = paths
        self.fileList = {}
        for path in paths:
            openh5_temp = pd.HDFStore(path)
            tempList = list(openh5_temp["data"])
            openh5_temp.close()
            for item in tempList:
                self.fileList[item] = path

    def plot(self, y=None, x=None, start=None, stop=None):
        if x is None:
            df = self.get_df(y, start=start, stop=stop)
            ax = df.plot()
            plt.xlabel("Time")
            plt.xlim(0, 600)
        else:
            df = self.get_df([x, y], start=start, stop=stop)
            ax = df.plot(kind="scatter", x=x, y=y)
            ax.set_axisbelow(True)
            plt.minorticks_on()
        plt.grid(b=True, which="major", linestyle="-")
        plt.grid(b=True, which="minor", linestyle="dotted")

    def plotCalendar(self, tuples):
        """ Early version of the calendar plot.
         Only shows day to day availability."""

        df = self.get_df(tuples=tuples)
        NumberCount = df.groupby(df.index.date).count() / 1.44
        NumberCount.index = pd.to_datetime(NumberCount.index)
        NumberCount.plot(style="o:")
        plt.xlabel("Date")
        plt.grid(which="both")
        plt.ylabel("Availability (%)")

    def get_df(self, tuples="all", start=None, stop=None):
        # Maybe better to have some persistence here
        df_lst = []
        where = None
        if start is not None:
            where = "index>" + start
        if stop is not None:
            if where is not None:
                where = where + "&index<" + stop
            else:
                where = "index<" + stop
        if tuples == "all":
            tuples = list(self.fileList.keys())
        if type(tuples) is not list:
            tuples = [tuples]
        for xtuple in tuples:
            store = pd.HDFStore(self.fileList[xtuple])
            df = store.select("data", columns=[xtuple], where=where)
            # Still loads each columns individual, can be optimized
            store.close()
            df_lst.append(df)
        df = pd.concat(df_lst, axis=1)
        df = df.sortlevel(0, axis=1)
        return df

    def delete(self, indices, tuples="all", df=None):
        from numpy import nan

        if tuples == "all":
            tuples = list(self.fileList.keys())
        if df is None:
            df = self.get_df()
            # Loads all dataframes for all timestamps, can be optimized
        df.set_value(indices, tuples, nan)
        return df

    def export(self, path=None, tuples="all", start=None):
        # Exports a series object to CSV

        if tuples == "all":
            tuples = list(self.fileList.keys())
        df = self.get_df(tuples=tuples)
        if start is not None:
            df = df[start:]
        if path is None:
            path = "DW_" + df.columns.levels[0][0]
        TsStart = min(df.index).strftime("%Y%m%d")
        TsEnd = max(df.index).strftime("%Y%m%d")
        df.to_csv(path_or_buf=path + "_" + TsStart + "_" + TsEnd + ".csv")

    def __repr__(self):
        return "DYNAwind series object"


def get_campaign_info():
    import pkg_resources
    return pkg_resources.resource_string("dynawind.dynawind", "jsonlookups/campaign_info.json")


def getSite(location):
    return get_site(location)


def get_site(location):
    """ Returns the site (e.g. belwind) for a given location (e.g. BBC01)

    :param location: location of which the site is requested.
    """
    json_data = get_campaign_info()

    try:
        site = lookup.lookup_location(json_data, location)
    except NameError:
        site = "unknown"

    return site


def get_locations(site):
    """ Returns all locations from a measurement site: e.g. Nobelwind

    :param site: site (case-insensitive)
    """
    json_data = get_campaign_info()

    return lookup.get_locations(json_data, site)


class SignalList(list):
    """ List of dynawind Signal objects """

    def __getitem__(self, item):
        result = list.__getitem__(self, item)
        try:
            return SignalList(result)
        except TypeError:
            return result

    def plot(self, absoluteTime=False, color=None, ax=None, legend=True):
        """ Plot all signals in a SignalList

        :param absoluteTime:
        :type absoluteTime: bool
        :param color:
        :param ax:
        :param legend:
        """
        if ax is None:
            ax = plt.gca()

        for signal in self:
            signal.plot(absoluteTime=absoluteTime, color=color, ax=ax, legend=legend)
        if legend:
            ax.legend()

    def plotPSD(self, rpm=None, xlim=None, ax=None, window='hann'):
        """ Plot PSD of all signals in a SignalList """
        if ax is None:
            ax = plt.gca()
        for signal in self:
            signal.plotPSD(rpm=rpm, xlim=xlim, ax=ax, window=window)
        ax.legend()

    def edit(self, round_limits=True):
        """ Launches a GUI in which you can edit the signals. Currently limited to clicking and selecting a part the data.

        """
        from numpy import float64
        plt.figure()
        self.plot()
        plt.title('Select two points to set new edges')
        selected_points = plt.ginput(n=2, show_clicks=True)
        t_stamps = [x[0] for x in selected_points]
        t_stamps.sort()
        # Everything within 5 seconds of beginning and end is reverted to the 0 or the end mark
        if round_limits:
            if t_stamps[1] > self[0].time()[-1] - 5:
                t_stamps[1] = self[0].time()[-1]
            if t_stamps[0] < 5:
                t_stamps[0] = float64(0)

        for signal in self:
            signal.select(t0=t_stamps[0], te=t_stamps[1])
        plt.gca().clear()
        self.plot()

    def drop(self, name_string):
        """ Drop all signals for which the name has the name_string in its signalname"""
        for signal in self:
            if name_string in signal.name:
                self.remove(signal)

    def detrend(self, approach='mean'):
        """ Detrend all signals, by default this is subtracting the mean value"""

        if approach == 'mean':
            for signal in self:
                signal.data = signal.data - signal.mean()


class Signal(object):
    """The signal class is the main class for handling timeseries inside dynawind.

    Note that typically a Signal instance is not called directly, but is the output of e.g. :func:`dynawind.dynawind.readTDMS`.

    :param location: location of origin, e.g. BBC01
    :param group: type of data, e.g. strain
    :param Fs: Sample frequency (Hz)
    :param data: data considered in the signal
    :param name: name of the signal
    :param unit: unit of the signal
    :type unit: str.
    :param timestamp: timestamp of the start of the signal
    :param db_con: Dynawind database open connection from which the e.g. SCADA will be drawn
    """

    # This object will contain a single (!) signal
    def __init__(self, location, group, Fs, data, name, unit, timestamp, db_con=None):
        """This is the __init__ function"""
        self.source = location  # Legacy! Turbine or location,
        self.location = location  # Location
        self.site = getSite(self.location)
        self.name = name  # Sensor name
        self.group = group  # E.g. acceleration
        self.data = data  # Signal
        self.Fs = Fs  # Sampling frequency
        self.unit_string = unit  # Engineering unit
        self.timestamp = timestamp  # Start of measurements (UTC)
        self.temperature_compensation = None  # Temperature compensation
        self.db_con = db_con  # set database_connection
        self = processSignal(self)  # Process the sensor based on config file

    def time(self, absoluteTime=False):
        """ Returns a vector with the timesteps of the signal.

        :param absoluteTime: Return time in absolute terms, i.e. time of day, or (False) as time since start of signal expressed in seconds.
        :type absoluteTime: bool
        """
        if absoluteTime:
            time_vector = [
                self.timestamp + datetime.timedelta(0, x / self.Fs)
                for x in range(0, len(self.data))
            ]
        else:
            time_vector = range(0, len(self.data))
            time_vector = [x / self.Fs for x in time_vector]
        return time_vector

    def calcPSD(self, window="hann"):
        """ Calculates the PSD for the signal, uses :func:`scipy.signal.pwelch`

        :param window: String specifying the window to use.
        :type window: str
        :return: f, PSD
        """
        from scipy import signal
        from numpy import eye

        if type(window) == type(eye(3)):
            nperseg = len(window)
        else:
            nperseg = self.Fs * 60
        f, Pxx_den = signal.welch(
            self.data, fs=self.Fs, nperseg=int(nperseg), window=window
        )
        self.f = f
        psd = Pxx_den
        self.PSD = psd
        return f, psd

    def plotPSD(self, window="hann", rpm=None, xlim=None, template=None, color=None, ax=None):
        """ Plot the Power spectral density as by calculated :func:`dynawind.dynawind.Signal.calcPSD`.

        :param template:
        :param color:
        :param ax:
        :param window:
        :param rpm: When a value is provided lines are plotted to indicate all harmonics up to 12p.
        :type rpm: int
        :param xlim: Lower and upper limit of the frequency range to plot, e.g.[0,4]
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if ax is None:
            ax = plt.subplot()

        self.calcPSD(window=window)
        if color:
            ax.semilogy(self.f, self.PSD, label=self.name, color=color)
        else:
            ax.semilogy(self.f, self.PSD, label=self.name)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (" + self.unit_string + "²/Hz)")
        if rpm is not None:
            for i in [1, 3, 6, 9, 12]:
                ax.axvline(
                    x=rpm * i / 60,
                    color="r",
                    linestyle="dotted",
                    linewidth=1,
                    label="_nolegend_",
                )
        if xlim is not None:
            ax.set_xlim(xlim)
            idx0 = (np.abs(self.f - xlim[0])).argmin()
            idxe = (np.abs(self.f - xlim[1])).argmin()
            ymax = np.max(self.PSD[idx0:idxe])
            ymin = np.min(self.PSD[idx0:idxe])
            ax.set_ylim(ymin * 0.9, ymax * 1.1)
        ax.grid(True, "both", "both", ls=":", lw=.5, c="k", alpha=.3)

    def rms(self, LFB=None, UFB=None, remove_offset=True):
        """ Calculate the RMS of the signal

        If LFB and UFB are used the rms is calculated using the PSD. This allows for instance to calc the energy in the 1P harmonic band.

        :param LFB: Lower bound of the frequency range to consider
        :param UFB: Upper bound of the frequency range to consider
        :param remove_offset: remove mean before calculating the RMS value
        :type remove_offset: bool.

        :returns: rms-value
        :rtype: float
        """
        import numpy as np

        if LFB is None and UFB is None:
            x = self.data
            if remove_offset:
                x = x - self.mean()
            rms = (sum(x ** 2) / len(x)) ** 0.5
            return rms
        else:
            f, PSD = self.calcPSD()
            if LFB is None:
                LFB = 0
            if UFB is None:
                UFB = f[-1]

            if np.isnan(LFB) or np.isnan(UFB):
                return np.nan
            # Find the closest frequency index to the desired frequency bands

            ind_0 = (np.abs(f - LFB)).argmin()
            ind_e = (np.abs(f - UFB)).argmin()
            df = f[2] - f[1]
            rms = np.sqrt(np.sum(PSD[ind_0: ind_e + 1]) * df)

            if remove_offset:
                offset = np.sqrt((PSD[ind_0] + PSD[ind_e]) / 2 * len(PSD[ind_0: ind_e + 1]) * df)
                rms = rms - offset
            return rms

    def rms1p(self, rpm=None, width=0.02, corrected=False):
        """ Calculates the energy in a band around 1P frequency

        :param rpm: rotor speed at the time of the measurement
        :param width: Frequency band that is considered for the calculation of 1P energy
        :returns: rms1p-value
        :param corrected: Subtract the boundary energy to better quantify the actual peak
        :rtype: float
        """
        if rpm is None:
            rpm = stat4signal(self, 'rpm', db_con=self.db_con)
        f = rpm / 60
        rms1p = self.rms(LFB=f - width, UFB=f + width, remove_offset=corrected)

        return rms1p

    def append(self, Signal):
        """ Append data to the end of the Signal.data"""
        from numpy import append

        # Appends a new signal to an existing one
        self.data = append(self.data, Signal.data)

    def select(self, t0=0, te=None):
        """ Selects (or crops) a time period from the data inplace

        :param t0: starting time (s) to select from
        :type t0: float
        :param te: end time (s) to select untill
        :type te: float
        :returns: None

        """
        if te is None:
            te = len(self.data) / self.Fs
        if t0 < 0:
            # Start from the back, e.g t0=-30 : take the last 30 seconds
            t0 = len(self.data) / self.Fs + t0

        self.data = self.data[int(t0 * self.Fs): int(te * self.Fs)]
        self.timestamp = self.timestamp + datetime.timedelta(0, t0)

    def std(self):
        """ Calculates the standard deviation

        :returns: std
        :rtype: float
        """
        from math import sqrt

        std = sqrt(sum((self.data - self.mean()) ** 2) / len(self.data))
        return std

    def median(self):
        """ Calculates the median of the signal

        :returns: median
        :rtype: float
        """
        from statistics import median

        median = median(self.data)
        return median

    def mean(self):
        """ Calculates the mean of the signal

        :returns: mean
        :rtype: float
        """
        mean = sum(self.data) / len(self.data)
        return mean

    def plot(self, absoluteTime=False, color=None, ax=None, legend=True):
        """ Plots the signal in a preconfigured way

        :param ax:
        :param absoluteTime: When true plots the x-axis in UTC
        :type absoluteTime: bool
        :param color: Additional color specification (as allowed by pyplot)
        :param legend: Boolean to indicate whether a legend should be plot or not

        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if color:
            ax.plot(self.time(absoluteTime=absoluteTime), self.data, label=self.name, color=color)
        else:
            ax.plot(self.time(absoluteTime=absoluteTime), self.data, label=self.name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self.group + " (" + self.unit_string + ")")
        plt.tight_layout()
        if legend:
            ax.legend()
        ax.grid(True, "both", "both", ls=":", lw=.5, c="k", alpha=.3)
        if not absoluteTime:
            ax.set_xlim(0, max(self.time()))
        else:
            ax.set_xlim([self.timestamp, self.time(absoluteTime=True)[-1]])

    def filter(self, LFB=0, UFB=5, order=8):
        """Code verified by WW on 07/02/2017 : http://192.168.119.55/x/WoF0
         this function does not prevent to filter twice, this can be
         especially harmfull when the first filter was narrower
        than the second

        :param LFB: Lower frequency bound
        :param UFB: Upper frequency bound
        :param order: filter order
        """
        from scipy import signal

        # Will use a butterworth filter to the data in
        if LFB == 0:
            frequencyband = UFB / self.Fs * 2
            b, a = signal.butter(order, frequencyband, "low")
        else:
            b, a = signal.butter(
                order, [LFB / self.Fs * 2, UFB / self.Fs * 2], "bandpass"
            )

        y_filtered = signal.filtfilt(b, a, self.data)
        self.data = y_filtered

    def downsample(self, fs):
        """ Downsample the signal

        Currently only works with integer multiples, will raise a ValueError when not an integer multiple

        :param fs: frequency to downsample to
        :raises ValueError: if the sample frequency is not an integer multiple of the downsample one
        """
        if not self.Fs % fs == 0:
            raise ValueError(
                "original sample frequency should be an integer multiple of the downsample"
            )
        step = int(self.Fs / fs)
        self.data = self.data[::step]
        self.Fs = fs

    def __repr__(self):
        # This can be later used to give an overview of the properties in DYNAwind
        descr_str = (
                "DYNAwind signal object\n"
                + "source:\t"
                + self.source
                + "\n"
                + "name:\t"
                + self.name
                + "\n"
        )
        return descr_str


def get_config(signal=None, timestamp=datetime.datetime.now().replace(tzinfo=pytz.utc), source=None, name=None,
               group=None):
    """ Pull the configuration for a specified :class:`dynawind.dynawind.Signal`, a location or a sensor name
    :param signal:
    :type signal: Signal
    :param timestamp:
    :param source:
    :param name:
    :param group:
    :return:
    """

    import json
    import pkg_resources

    resource_package = __name__
    from datetime import datetime

    if signal is not None:
        source = signal.location
        group = signal.group
        timestamp = signal.timestamp
        name = signal.name
        site = getSite(signal.location)
    else:
        site = getSite(source)

    if source != "unknown":
        resource_path = "/".join(("config", site.lower(), source + ".json"))
        if pkg_resources.resource_exists(resource_package, resource_path):
            if hasattr(get_config, "source") and get_config.source == source:
                # rely on persistent config file from a previous call
                pass
            else:
                json_file = pkg_resources.resource_string(
                    resource_package, resource_path
                )
                get_config.source = source
                get_config.data = json.loads(json_file)
            for i in range(0, len(get_config.data)):
                recordTimestamp = datetime.strptime(
                    get_config.data[i]["time"], "%d/%m/%Y %H:%M"
                )
                recordTimestamp = pytz.utc.localize(recordTimestamp)
                if recordTimestamp > timestamp:
                    record = get_config.data[i - 1]
                    break
                record = get_config.data[i]
            if name is not None:
                if group is None:
                    keepkeys = ["time", name]
                else:
                    keepkeys = ["time", name, group]

                if name + "/name" in record.keys():
                    keepkeys.append(record[name + "/name"])
                recordkeys = []
                for keepkey in keepkeys:
                    for key in record.keys():
                        if keepkey in key:
                            recordkeys.append(key)
                record = {recordkey: record[recordkey] for recordkey in recordkeys}
            return record
        else:
            warnings.warn("No config found for : " + str(Signal.location))
            return None
    else:
        return None


def get_sensor_list(location, sensor_type=None, timestamp=datetime.datetime.now(utc), ignore_broken=False):
    """ Function to provide a list of sensors associated with a certain location and optionally a sensor type (e.g. LVDT)

    :param location:
    :param sensor_type:
    :param timestamp:
    :param ignore_broken:
    :return: List of all sensors
    """
    timestamp
    config = get_config(source=location, timestamp=timestamp)
    sensor_list = []
    for key in config.keys():
        if 'status' in key:
            if config[key] == 'ok' or not ignore_broken:
                if sensor_type is None or sensor_type in key:
                    sensor_list.append(key[:key.find('/')])
    return sensor_list


def yawTransformation(Signals, stat_dict):
    raise PendingDeprecationWarning('Replaced by yaw_transformation')
    return yaw_transformation(Signals, stat_dict)


def yaw_transformation(Signals, stat_dict):
    """ Perform a yaw transformation on the signals in Signals

    For an overview of this function : https://24seaa.atlassian.net/wiki/spaces/~wout.weijtjens/pages/536412565/Yaw+transformation

    :param Signals: list of :class:`dynawind.dynawind.Signal`
    :param stat_dict: Dictionary containing the information on the yaw angle.
    :type stat_dict: dict.
    """
    from math import pi, sin, cos
    from numpy import asmatrix, sign

    def yawTransformList(Signals, stat_dict):
        # When more than two Signals are passed this script will use the secondaries property in the config files to pair the signals and perform a FA,SS calculation
        secondaries_list = []
        for sgnl in Signals:
            config = get_config(sgnl)
            secondaries_list.append(config[sgnl.name + "/secondaries"])
        for i in range(len(secondaries_list)):
            for j in range(i + 1, len(secondaries_list)):
                if secondaries_list[i] == secondaries_list[j]:
                    FASS = yaw_transformation([Signals[i], Signals[j]], stat_dict)
                    Signals.append(FASS[2])
                    Signals.append(FASS[3])
        return Signals

    if len(Signals) > 2:
        Signals = yawTransformList(Signals, stat_dict)
        return Signals

    Signal1 = Signals[0]
    Signal2 = Signals[1]
    yaw_angle = stat_dict["yaw/mean"]
    # Code verified on 07/02/2017 by WW : http://192.168.119.55/x/SoF0
    if Signal1.timestamp != Signal2.timestamp:
        raise NameError("Timestamps of signals does not match")
    # Step 1: Apply the signs of the heading
    if sign(Signal1.heading) != 0:
        s1 = Signal1.data * sign(Signal1.heading)
    else:
        s1 = Signal1.data
    if sign(Signal2.heading) != 0:
        s2 = Signal2.data * sign(Signal2.heading)
    else:
        s2 = Signal2.data
    # Step 2: Identfy the setup ('XX' or 'XY')
    if Signal1.orientation + Signal2.orientation == "XY":
        sx = s1
        sy = s2
    elif Signal1.orientation + Signal2.orientation == "YX":
        sx = s2
        sy = s1
    elif Signal1.orientation + Signal2.orientation == "XX":
        raise NameError("XX configuration not implemented yet")
    else:
        raise NameError("Unexpected orientation of sensors")
    # Step 3 : construct the rotation matrix
    Xheading = max([abs(Signal1.heading), abs(Signal2.heading)]) / 180 * pi
    R1 = [[cos(Xheading), sin(Xheading)], [-sin(Xheading), cos(Xheading)]]
    yaw_angle = (yaw_angle + 180) / 180 * pi
    # +180 as the positive FA direction is pointing towards Aft
    R2 = [[cos(yaw_angle), -sin(yaw_angle)], [sin(yaw_angle), cos(yaw_angle)]]

    R_tot = asmatrix(R2) * asmatrix(R1)

    # Step 4 : perform the rotation
    sFA = R_tot[0, 0] * sx + R_tot[0, 1] * sy
    sSS = R_tot[1, 0] * sx + R_tot[1, 1] * sy
    # Step 5 : return a list of DYNAwind signals

    SensorString = Signal1.name.split("_")
    del SensorString[-2:]  # remove heading and orientation
    SensorString = "_".join(SensorString)

    Signal_FA = Signal(
        Signal1.location,
        Signal1.group,
        Signal1.Fs,
        sFA,
        SensorString + "_FA",
        Signal1.unit_string,
        Signal1.timestamp,
    )
    Signal_SS = Signal(
        Signal1.location,
        Signal1.group,
        Signal1.Fs,
        sSS,
        SensorString + "_SS",
        Signal1.unit_string,
        Signal1.timestamp,
    )

    if hasattr(Signal1, "level"):
        setattr(Signal_FA, "level", Signal1.level)
        setattr(Signal_SS, "level", Signal1.level)
    setattr(Signal_FA, "heading", "FA")
    setattr(Signal_SS, "heading", "SS")

    Signals.extend([Signal_FA, Signal_SS])
    return Signals


def updateSecondary(
        Series,
        tuples_signal,
        tuples_in,
        tuples_out,
        functions,
        SourceFolder,
        start=None,
        stop=None,
):
    import datetime
    import os

    """
    tuples_signal :
    sensors you need the timeseries from to calculate
    the secondary parameters

    tuples_in     : statistics required for the calculation of
    the secondary parameters (eg. SCADA yaw)
    tuples_out    :
    """
    # %% Filter for the timestamps where you should be able
    # to calculate the secondary parameters
    tuples = tuples_in.copy()
    tuples.extend(tuples_signal)
    df = Series.get_df(tuples, start=start, stop=stop)
    df.dropna(axis=0, how="any", inplace=True)
    # %%
    df_second = Series.get_df(tuples_out, start=start, stop=stop)
    result = pd.concat([df, df_second], axis=1, join_axes=[df.index])
    sensors = []
    for signaltuple in tuples_signal:
        sensors.append(signaltuple[2])

    for index in result.index:
        datestr = datetime.datetime.strftime(
            index - datetime.timedelta(0, 600), "%Y%m%d_%H%M%S"
        )
        yyyy = datestr[:4]
        mm = datestr[4:6]
        dd = datestr[6:8]
        if os.path.isfile(
                SourceFolder
                + os.sep
                + yyyy
                + os.sep
                + mm
                + os.sep
                + dd
                + os.sep
                + datestr
                + ".tdms"
        ):
            signals = readTDMS(
                SourceFolder
                + os.sep
                + yyyy
                + os.sep
                + mm
                + os.sep
                + dd
                + os.sep
                + datestr
                + ".tdms"
            )
        elif os.path.isfile(SourceFolder + os.sep + datestr + ".tdms"):
            signals = readTDMS(SourceFolder + os.sep + datestr + ".tdms")
        else:
            continue
        for signal in signals:
            if signal.name not in sensors:
                signals.remove(signal)
        stat_dict = dict()
        # Passes the ten minute values as dictionary
        for stats in tuples_in:
            stat_dict[stats[2] + "/" + stats[3]] = result[stats][index]
        for function in functions:
            signals = function(signals, stat_dict)
        for signal in signals:
            updateSeries(Series, signal, overwrite=True)


def processSignal(Signal):
    record = get_config(Signal)
    if record is not None:
        # The current record is applicable to the current signal,
        # Set additional properties
        def setSecondarySensorProperties():
            # name should always be first!
            propertyList = [
                "name",
                "unit_string",
                "sensitivity",
                "heading",
                "level",
                "orientation",
                "group",
                "offset",
                "Ri",
                "Ro",
                "filterPassband",
                "filterOrder",
                "downsample",
                "status"
            ]
            for prop in propertyList:
                if Signal.name + "/" + prop in record:
                    setattr(Signal, prop, record[Signal.name + "/" + prop])
            return Signal

        Signal = setSecondarySensorProperties()
        # Check if the signal has a filter defined
        if Signal.name + "/filterPassband" in record.keys():
            Signal.filter(
                record[Signal.name + "/filterPassband"][0],
                record[Signal.name + "/filterPassband"][1],
                record[Signal.name + "/filterOrder"],
            )
        elif Signal.group + "/filterPassband" in record.keys():
            Signal.filter(
                record[Signal.group + "/filterPassband"][0],
                record[Signal.group + "/filterPassband"][1],
                record[Signal.group + "/filterOrder"],
            )


        # Check if the signal has to be detrended
        if Signal.name + "/detrend" in record.keys():
            if record[Signal.name + "/detrend"]:
                Signal.data = Signal.data - Signal.mean()
        # Check if the signal is to be downsampled
        if Signal.name + "/downsample" in record.keys():
            Signal.downsample(record[Signal.name + "/downsample"])
        elif Signal.group + "/filterPassband" in record.keys():
            Signal.downsample(record[Signal.group + "/downsample"])

        # Verify that the signal is in engineering units and correct if necessary
        if hasattr(Signal, "group"):
            if Signal.group == "acceleration" or Signal.group == "vibrations":
                if Signal.unit_string != "g":
                    try:
                        Signal.data = Signal.data / record[Signal.name + "/sensitivity"]
                        Signal.unit_string = "g"
                        Signal.group = 'acc'
                    except:
                        raise ValueError(
                            "Conversion to engineering units failed : sensitivity from ("
                            + Signal.unit_string
                            + ") to (g) of "
                            + Signal.name
                            + " not defined"
                        )
            elif Signal.group == "strain":
                if Signal.unit_string != "microstrain":
                    Signal = processStrainGauges(Signal, record)
            elif Signal.group == "displacement":
                if Signal.name + "/sensitivity" in record.keys():
                    Signal.data = Signal.data / record[Signal.name + "/sensitivity"]
            else:
                if Signal.name + "/sensitivity" in record.keys():
                    Signal.data = Signal.data / record[Signal.name + "/sensitivity"]
                if Signal.name + "/offset" in record.keys():
                    Signal.data = Signal.data - record[Signal.name + "/offset"]

            # Temperature compensation (Not functional yet!!!!)
            # !!! To Do update temperature_compensation to pull temp from JSON
    #            if Signal.name+'/TCSensor' in record.keys():
    #                temperature_sensor = record[Signal.name+'/TCSensor']
    #                if Signal.name+'/TemperatureCompensation' in record.keys():
    #                    temp_coef = record[Signal.name+'/TemperatureCompensation']
    #                else:
    #                    temp_coef=None # will trigger default settings
    #                Signal = temperature_compensation(Signal,
    #                                                  temperature_sensor=temperature_sensor,
    #                                                  temp_coef=temp_coef,
    #                                                  group=Signal.group)

    return Signal


def processStrainGauges(Signal, record):
    if Signal.unit_string == "strain":
        Signal.data = Signal.data * 1e6
        Signal.unit_string = "microstrain"
    elif Signal.unit_string == "Nm":
        pass  # Bending moment
    elif Signal.unit_string == "N":
        pass  # Normal Load
    else:
        if Signal.name + "/bridgeType" in record:
            if record[Signal.name + "/bridgeType"] == "quarter":
                # Quarter bridge calculation
                # - without lead compensations
                # - No shear stress compensation, assumption of a uni-axial stress condition
                Signal.data = (
                        -4
                        * Signal.data
                        / record[Signal.name + "/gageFactor"]
                        / (1 + 2 * Signal.data)
                        * 1e6
                )  # NI documentation : Strain Gauge Configuration types
                Signal.unit_string = "microstrain"
            else:
                raise NameError("Brigetype specified in config file is not supported")
        else:
            raise NameError("Bridge type not specified in config file")
    return Signal


def remove_faulty_sensors(signals):
    """ Code to remove all signals with status not equal to ok"""
    signals = [x for x in signals if x.status == 'ok']

    return signals


def calc_relative_inclination(signals, only_ok=True):
    """
    This function serves to calculate the relative inclination from at least three LVDT sensors.
    Geometrical details of the sensors are derived from the configuration of the LVDT sensors.

    :param signals: list of :class:`dynawind.dynawind.Signal`
    :param only_ok: Boolean to express whether or not only sensors with status ok are considered

    """
    import numpy as np

    if only_ok:
        signals = remove_faulty_sensors(signals)

    Ri = np.empty([len(signals), 1])
    headings = np.empty([len(signals), 1])
    displacements = np.empty([len(signals), len(signals[0].data)])

    for i, signal in zip(range(len(signals)), signals):
        if signal.unit_string == 'mm':
            Ri[i] = signal.Ri * 1000  # Convert to mm
        elif signal.unit_string == 'm':
            Ri[i] = signal.Ri
        else:
            raise ValueError(signal.name + ' : unit_string should be  m or mm')

        headings[i] = signal.heading
        displacements[i, :] = signal.data

    K = np.empty([len(signals), 3])
    for i in range(len(signals)):
        K[i, 0] = 1
        K[i, 1] = Ri[i, 0] * np.sin((360 - headings[i]) / 180 * np.pi)
        K[i, 2] = Ri[i, 0] * np.cos((360 - headings[i]) / 180 * np.pi)

    K_pinv = np.linalg.pinv(K)
    theta = np.dot(K_pinv, displacements)

    plane_coor = np.ones(theta.shape)
    plane_coor[1:, :] = theta[1:, :]

    inclination = np.arccos(
        np.divide(1, np.sqrt(np.sum(plane_coor ** 2, axis=0)))) / np.pi * 180  # Inclination angle (phi)
    theta_angle = np.arctan2(theta[2, :], -theta[1, :]) / np.pi * 180 + 90  # the direction of the inclination (theta)
    theta_angle = -theta_angle
    theta_angle[theta_angle < 0] = theta_angle[theta_angle < 0] + 360

    inclination_direction = theta_angle
    mean_displacement = theta[0, :]

    sensor_string = signals[0].name.split("_")
    sensor_string = "_".join(sensor_string[:3])  # remove heading and orientation

    signals.append(
        Signal(
            signals[0].source,
            signals[0].group,
            signals[0].Fs,
            inclination,
            sensor_string + "_RelIncl",
            "deg.",
            signals[0].timestamp,
        )
    )

    signals.append(
        Signal(
            signals[0].source,
            signals[0].group,
            signals[0].Fs,
            theta_angle,
            sensor_string + "_InclDir",
            "deg.",
            signals[0].timestamp,
        )
    )

    signals.append(
        Signal(
            signals[0].source,
            signals[0].group,
            signals[0].Fs,
            mean_displacement,
            sensor_string + "_displacement",
            "deg.",
            signals[0].timestamp,
        )
    )

    return signals


def calcBendingMomentSignal(Signals, stat_dict, measurement_location='inner'):
    Signals = calc_bending_moment(Signals, yaw_angle=stat_dict["yaw/mean"], measurement_location='inner')
    return Signals


def calc_bending_moment(signals, yaw_angle=None, measurement_location='inner'):
    """
    :param signals:
    :param yaw_angle: actual yaw angle of the turbine to convert into Mtn-Mtl frame of reference. By default this is None, resulting in Bending moment and bending direction
    :param measurement_location: whether the strain gauges are on the inside or outside of the turbine wall

    """
    import numpy as np

    Ri = np.empty([len(signals), 1])
    Ro = np.empty([len(signals), 1])
    headings = np.empty([len(signals), 1])
    strains = np.empty([len(signals), len(signals[0].data)])
    for i in range(len(signals)):
        Ri[i] = signals[i].Ri
        Ro[i] = signals[i].Ro
        headings[i] = signals[i].heading
        strains[i, :] = signals[i].data
    A = np.pi * (Ro ** 2 - Ri ** 2)  # Surface area
    Ic = np.pi / 4 * (Ro ** 4 - Ri ** 4)  # Area Moment of Inertia
    if signals[0].unit_string == "microstrain":
        # Young modulus of steel + conversion from microstrain to strain
        YoungModulus = 210e9 / 1e6
    else:
        raise NameError("Strains not in microstrain")

    K = np.empty([len(signals), 3])
    for i in range(len(signals)):
        K[i, 0] = 1 / A[i, 0]
        if measurement_location == 'inner':
            K[i, 1] = Ri[i, 0] / Ic[i] * np.sin((360 - headings[i]) / 180 * np.pi)
            K[i, 2] = -Ri[i, 0] / Ic[i] * np.cos((360 - headings[i]) / 180 * np.pi)
        elif measurement_location == 'outer':
            K[i, 1] = Ro[i, 0] / Ic[i] * np.sin((360 - headings[i]) / 180 * np.pi)
            K[i, 2] = -Ro[i, 0] / Ic[i] * np.cos((360 - headings[i]) / 180 * np.pi)
    K = K / YoungModulus
    K_pinv = np.linalg.pinv(K)
    Theta = np.dot(K_pinv, strains)

    if yaw_angle is not None:
        # Transform into the Mtn and Mtl frame work

        yaw_angle = (-yaw_angle + 180) / 180 * np.pi
        cd = np.cos(yaw_angle)
        sd = np.sin(yaw_angle)
        R = np.asmatrix([[cd, sd], [-sd, cd]])
        M = np.dot(R, Theta[1:, :])

    else:
        M = np.sqrt(Theta[1, :] ** 2 + Theta[2, :] ** 2)
        M_dir = 180 + np.arctan2(Theta[1, :], Theta[2, :]) / np.pi * 180
    # Turn results into DYNAwind signals
    SensorString = signals[0].name.split("_")
    SensorString = "_".join(SensorString[:3])  # remove heading and orientation
    signals.append(
        Signal(
            signals[0].source,
            signals[0].group,
            signals[0].Fs,
            np.squeeze(np.asarray(Theta[0, :])),
            SensorString + "_N",
            "N",
            signals[0].timestamp,
        )
    )
    if yaw_angle is not None:

        signals.append(
            Signal(
                signals[0].source,
                signals[0].group,
                signals[0].Fs,
                np.squeeze(np.asarray(M[0, :])),
                SensorString + "_Mtl",
                "Nm",
                signals[0].timestamp,
            )
        )
        signals.append(
            Signal(
                signals[0].source,
                signals[0].group,
                signals[0].Fs,
                np.squeeze(np.asarray(M[1, :])),
                SensorString + "_Mtn",
                "Nm",
                signals[0].timestamp,
            )
        )
    else:
        signals.append(
            Signal(
                signals[0].source,
                signals[0].group,
                signals[0].Fs,
                np.squeeze(np.asarray(M_dir)),
                SensorString + "_Mdir",
                "Nm",
                signals[0].timestamp,
            )
        )
        signals.append(
            Signal(
                signals[0].source,
                signals[0].group,
                signals[0].Fs,
                np.squeeze(np.asarray(M)),
                SensorString + "_Mtn",
                "Nm",
                signals[0].timestamp,
            )
        )
    return signals


def plotLoadEvent(signals, yaw_angle, measurement_location='inner'):
    stat_dict = {'yaw/mean': yaw_angle}
    f, (ax1, ax2) = plt.subplots(2, 1)
    for signal in signals:
        ax2.plot(signal.time(), signal.data)
    plt.xlabel("Time (s)")
    plt.ylabel(signal.group + " (" + signal.unit_string + ")")
    plt.legend()
    plt.grid(True, "both", "both", ls=":", lw=.5, c="k", alpha=.3)
    plt.xlim(0, max(signal.time()))
    #
    BM_signals = calcBendingMomentSignal(signals, stat_dict, measurement_location=measurement_location)
    #    # Two subplots, the axes array is 1-d
    ax1.plot(BM_signals[2].time(), BM_signals[-1].data / 1e3)  # Mtn
    ax1.plot(BM_signals[1].time(), BM_signals[-2].data / 1e3)  # Mtl
    ax1.legend("Mtn", "Mtl")
    ax1.grid(True, "both", "both", ls=":", lw=.5, c="k", alpha=.3)
    ax1.set_xlim(0, max(signal.time()))
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Bending Moments (kNm)")
    ax1.set_title('Load event at ' + str(signal.timestamp) + ': Level ' + str(signal.level))

    return BM_signals


def exportSignals(Signals, filename):
    """ function to export a list of Signals to .txt files that can be easily read by MATLAB

    :param Signals: list of :class:`dynawind.dynawind.Signal`
    :param filename: filename to store the export as
    """
    exportFile = open(filename, "w")
    exportFile.write("Export from DYNAwind" + "\n")
    exportFile.write("Generated : " + str(datetime.datetime.utcnow()) + "\n")
    exportFile.write("\n")
    exportFile.write("Time")
    for i in range(0, len(Signals)):
        exportFile.write(",\t" + Signals[i].name)
    exportFile.write("\n")
    exportFile.write("")
    for i in range(0, len(Signals)):
        exportFile.write(",\t" + Signals[i].group)
    exportFile.write("\n")
    exportFile.write("(s)")
    for i in range(0, len(Signals)):
        exportFile.write(",\t(" + Signals[i].unit_string + ")")
    exportFile.write("\n\n")
    time = Signals[0].time()
    for i in range(0, len(time)):
        exportFile.write(str(time[i]))
        for j in range(0, len(Signals)):
            exportFile.write(",\t" + str(Signals[j].data[i]))
        exportFile.write("\n")


def stats2df(Signals, Operators=[]):
    """ Converts a list of signals to a :class:`pandas.DataFrame` of statistics. The statistics applied are found in the config file (json) of the location.

    :param Operators:
    :param Signals: list of :class:`dynawind.dynawind.Signal`
    :returns: statistics calculated for the input signals
    :rtype: :class:`pandas.DataFrame`
    """
    # input is a list of signals
    import pandas as pd
    import pytz
    from numpy import nan
    import numpy as np

    if not Operators:
        getOperators = True
    else:
        getOperators = False

    multiIndexTuples = []
    for i in range(0, len(Signals)):
        df = pd.DataFrame()
        if getOperators:
            Operators = get_operators(Signals[i])
        timestamp = Signals[i].timestamp + datetime.timedelta(0, 0, 0, 0, 10, 0)
        df.loc[0, "time"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        for j in range(0, len(Operators)):
            multiIndexTuples.append(
                (Signals[i].source, Signals[i].group, Signals[i].name, Operators[j])
            )
            # Not so happy with this section
            if Operators[j] == "min":
                df.loc[0, Operators[j] + "_" + Signals[i].name] = np.min(
                    Signals[i].data
                )
            elif Operators[j] == "max":
                df.loc[0, Operators[j] + "_" + Signals[i].name] = np.max(
                    Signals[i].data
                )
            elif Operators[j] == "offset":
                if hasattr(Signals[i], "offset"):
                    df.loc[0, Operators[j] + "_" + Signals[i].name] = Signals[i].offset
                else:
                    df.loc[0, Operators[j] + "_" + Signals[i].name] = 0
            elif Operators[j] == "tc":
                df.loc[0, Operators[j] + "_" + Signals[i].name] = nan
            else:
                if callable(getattr(Signals[i], Operators[j])):
                    df.loc[0, Operators[j] + "_" + Signals[i].name] = getattr(
                        Signals[i], Operators[j]
                    )()
                else:
                    df.loc[0, Operators[j] + "_" + Signals[i].name] = getattr(
                        Signals[i], Operators[j]
                    )

        if i == 0:
            result = df
        else:
            result = pd.merge(result, df, on="time")
    result.index = result["time"]
    if result.index.tzinfo is not None:
        if not result.index.tzinfo == pytz.utc:
            result.index = result.index.tz_convert(tz=pytz.utc)
    else:
        result.index = result.index.tz_localize(tz=pytz.utc, ambiguous="infer")
    # TDMS are in UTC!

    del result["time"]
    multiIndex = pd.MultiIndex.from_tuples(multiIndexTuples)
    result.columns = multiIndex
    result.sort_index(axis=1, inplace=True)
    return result


def get_operators(signal):
    Operators = []
    record = get_config(signal)
    for key in record.keys():
        if "stats" in key:
            for operator in record[key]:
                Operators.append(operator)
    return Operators


def get_decimals(signal):
    decimals = None
    # decimals is None implies do nothing, default behavior
    record = get_config(signal)
    if record:
        for key in record.keys():
            if "decimals" in key:
                decimals = int(record[key])

    return decimals


def stats2dict(Signals, Operators=[], decimals=None):
    """ Converts a list of signals to a dictionary of statistics. The statistics applied are found in the config file
    (json) of the location.

    :param Operators:
    :param decimals:
    :param Signals: list of :class:`dynawind.dynawind.Signal`
    :returns: The statistics calculated for the parsed signals.
    :rtype: dict.
    """
    from numpy import float64  # JSON parser supports float64, not float32
    import importlib
    def round_result(result, decimals):
        if decimals is not None:
            result = round(result, decimals)
        return result

    if not Operators:
        get_operators_bool = True
    else:
        get_operators_bool = False

    result = dict()
    for i in range(0, len(Signals)):
        if get_operators_bool:
            Operators = get_operators(Signals[i])
        if decimals is None:
            decmls = get_decimals(Signals[i])
        else:
            decmls = decimals

        for j in range(0, len(Operators)):
            if Operators[j] == "min":
                result[Operators[j] + "_" + Signals[i].name] = round_result(float64(
                    min(Signals[i].data)
                ), decmls)
            elif Operators[j] == "max":
                result[Operators[j] + "_" + Signals[i].name] = round_result(float64(
                    max(Signals[i].data)
                ), decmls)
            else:
                if callable(getattr(Signals[i], Operators[j])):
                    result[Operators[j] + "_" + Signals[i].name] = round_result(getattr(
                        Signals[i], Operators[j]
                    )(), decmls)
                else:
                    result[Operators[j] + "_" + Signals[i].name] = round_result(getattr(
                        Signals[i], Operators[j]
                    ), decmls)
            ###

    # %% Check for custom functions, that return a dict to add to the current data
    record = get_config(Signals[0])
    if record:
        if Signals[0].group + "/custom" in record:
            site = Signals[0].site
            for fnct in record[Signals[0].group + "/custom"]:
                customMethod = getattr(
                    importlib.import_module(
                        "dynawind.config." + site.lower() + ".custom_" + site.lower()
                    ),
                    fnct,
                )
                result.update(customMethod(Signals))
        # %% Check for MPE
        if Signals[0].group + "/mpe/directions" in record:
            for direction in record[Signals[0].group + "/mpe/directions"]:
                mpe_signals = pullSignalsFromList(
                    Signals, record[Signals[0].group + "/mpe/" + direction]
                )
                mpe_results = mpe.MPE(mpe_signals, direction=direction)
                result.update(mpe_results.exportdict(trackedOnly=True))

    return result


def HF_stats2dict(Signals, timescale):
    """ Allows to generate statistics for higher frequencies than the original
    length of the Signal, e.g. every 60s """

    import copy

    if timescale > 600:
        raise ValueError("Timescale requested exceeds the 10-minute maximum")
    if not 600 % timescale == 0:
        raise ValueError(
            "The default 10 minute file length is not an integer multiple of the requested timescale"
        )
    NewLength = int(timescale * Signals[0].Fs)
    nrOfFiles = int(600 / timescale)
    #
    data = []
    site = [Signals[0].site] * nrOfFiles
    location = [Signals[0].location] * nrOfFiles
    dt = []
    Signals_short = copy.deepcopy(Signals[:])
    for i in range(nrOfFiles):
        for [signal, signal_short] in zip(Signals, Signals_short):
            signal_short.data = signal.data[i * NewLength: (i + 1) * NewLength]
        data.append(stats2dict(Signals_short))
        dt.append(signal.timestamp + datetime.timedelta(0, (i + 1) * timescale))

    return (dt, site, location, data)


def pullSignalsFromList(Signals, names):
    pulledSignals = []
    for signal in Signals:
        for name in names:
            if name == signal.name:
                pulledSignals.append(signal)
                continue
        continue
    return pulledSignals


def readTDMS(path, Source=None):
    """ Primary function to read tdms files

    :param path: path to a tdms file
    :param Source: or location of origin, e.g. BBC01
    """
    from nptdms import TdmsFile
    from datetime import datetime
    from os import sep
    from os.path import exists
    import pytz

    if not exists(path):
        print("No such file :" + path)
        Signals = []
        return Signals

    try:
        tdms_file = TdmsFile(path)
    except:
        print("Failed to open :" + path)
        Signals = []
        return Signals

    def getSource(tdms_file, path):
        # (Preferred) It is also possible to determine the source from the tdms_file.object().properties
        # Legacy solution, use the path
        source = "unknown"
        strList = path.split(sep)
        for i in range(0, len(strList)):
            if strList[i] == "TDD":
                source = strList[i - 1]
                break
        return source

    if Source == None:
        Source = getSource(tdms_file, path)
    Groups = tdms_file.groups()
    Signals = SignalList()  # Make list into dw.SignalList object
    for i in range(0, len(Groups)):
        Channel_lst = tdms_file.group_channels(Groups[i])
        for j in range(0, len(Channel_lst)):
            Name = str(Channel_lst[j]).split("/")[2][1:-2]
            if "unit_string" in Channel_lst[j].properties:
                Unit = Channel_lst[j].properties["unit_string"]
            else:
                Unit = ""

            Data = tdms_file.object(Groups[i], Name)
            Fs = 1 / Channel_lst[j].properties["wf_increment"]
            timestampstr = path[-20:-5]
            # timestamp is start of measurement
            timestamp = datetime.strptime(timestampstr, "%Y%m%d_%H%M%S")
            timestamp = pytz.utc.localize(timestamp)
            Signals.append(
                Signal(Source, Groups[i], Fs, Data.data, Name, Unit, timestamp)
            )
    # %% Quality check of Signals
    signalLengths = []
    for signal in Signals:
        length = signal.data.shape[0]
        if length == 0:
            Signals.remove(signal)
        else:
            signalLengths.append(length)

    return Signals


def read_RecoVib(path, Fs=1024, local_tz=pytz.timezone('Europe/Amsterdam'), sep=',', decimal='.'):
    """ Function to import data that is exported from the Micromega Recovib units. To use this function make sure you have the recovib data exported as a csv file, with seperator (,) and decimal (.)

    :param path: path to the folder containing the Measure_*.csv
    :type path: str.
    :param Fs: Sample frequency of recovib, current defaults to 1024Hz as stated in Manual p.17
    :param local_tz: time zone of the dataset
    :type local_tz: pytz timezone
    :param sep: seperator in Recovib Measure_*.csv file, typically either , or ;
    :param decimal: seperator for decimals in Recovib Measure_*.csv, typically either '.' or ','
    """
    from glob import glob
    from os.path import join

    unit_str = 'g'  # In fact m/s^2 but will be converted later

    file_path = join(path, 'Measure_*.csv')
    file_path = glob(file_path)

    if len(file_path) == 0:
        raise IOError('No Measure_*.csv found in ' + path)
    else:
        file_path = file_path[0]

    df = pd.read_csv(file_path, sep=sep, decimal=decimal)
    # Pull timestamp
    filename = join(path, 'basic', 'times_log.txt')
    # Using the newer with construct to close the file automatically.
    with open(filename) as f:
        data = f.readlines()
    ts = data[0].split('Start at (yy-mm-dd hh:mm:ss):')[1][:17]
    duration = data[0].split('Time duration:')[1][:8]
    dt = datetime.datetime.strptime(ts, '%y-%m-%d %H:%M:%S')
    # convert to UTC
    dt = local_tz.normalize(local_tz.localize(dt)).astimezone(pytz.utc)
    signals = SignalList()
    for col in df.columns[1:]:
        if 'x' in col:
            name = 'MM_recovib_ACC_X'
        elif 'y' in col:
            name = 'MM_recovib_ACC_Y'
        elif 'z' in col:
            name = 'MM_recovib_ACC_Z'
        signal = Signal('RecoVib',
                        'acc',
                        Fs,
                        df[col].values / 9.81,
                        name, unit_str,
                        dt)
        signals.append(signal)
    return signals


def getTDMSpath(timestamp, filetype, location, site=None, root=r"\\192.168.119.14"):
    """ Returns the tdms path for a given timestamp, filetype and location

    """
    from os.path import join

    if not site:
        site = getSite(location)
    if isinstance(timestamp, str):
        dt = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    else:
        dt = timestamp
    path = join(
        root,
        "data_primary_" + site.lower(),
        location,
        "TDD",
        "TDD_" + filetype,
        str(dt.year),
        str(dt.month).zfill(2),
        str(dt.day).zfill(2),
        dt.strftime("%Y%m%d_%H%M%S") + ".tdms",
    )
    return path


def getTDMS(timestamp, filetype, location, site=None, root=r"\\192.168.119.14", duration=1):
    """ Alternative function to reach TDMS files stored in a controlled folder structure. Rather than to give the full path you can use this function to retrieve a tdms file based on timestamp, type and location.

    :param timestamp: timestamp
    :type timestamp: :class:`datetime.datetime`
    :param filetype: filetype to import using the short code, e.g. 'acc' or 'fiber'
    :param location: location of the measurement, e.g. BBC01
    :param site: site associated with location, not necessary if the location can be found using :func: `dynawind.dynawind.get_site`
    :param root: root of the folder structure, defaults to NAS4
    :param duration: Expressed in number of consecutive files, will thus take into account the actual length of the file. When larger than 1 dynawind will merge consecutive files, uses mergeTDMS.
    """
    if duration == 1:
        path = getTDMSpath(timestamp, filetype, location, site=site, root=root)
        signals = readTDMS(path)
    else:
        signals = mergeTDMS(timestamp, duration, location, filetype=filetype, root=root, site=site)

    return signals


def mergeTDMS(dt, duration, location, filetype=None, root=r"\\192.168.119.14", single_folder=False, site=None):
    """ Function to load consecutive TDMS files and merge them into a single set of signals, i.e. of longer duration.

    Note: this approach only works for files in the default folder-structure.

    :param site:
    :param dt: timestamp to start with
    :type dt: :class:`datetime.datetime`
    :param duration: Expressed in number of consecutive files, will thus take into account the actual length of the file.
    :type duration: int.
    :param location: location to import data from, e.g. BBC01
    :param filetype: filetype to import using the short code, e.g. 'acc' or 'fiber'
    :param root: root of the folder structure, defaults to NAS4
    :param single_folder: Indicates whether the files are to be taken from a single folder or not
    :type single_folder: Boolean
    """
    from os.path import join
    if single_folder:
        path = join(root, dt.strftime('%Y%m%d_%H%M%S') + '.tdms')
        Signals = readTDMS(path)
    else:
        # duration : in multiples of 10 minutes
        Signals = getTDMS(timestamp=dt, location=location, root=root, filetype=filetype, site=site)

    signal_length = len(Signals[0].data) / Signals[0].Fs  # (s)
    for i in range(1, duration):
        dt_local = dt + datetime.timedelta(0, signal_length * i)
        if single_folder:
            path = join(root, dt_local.strftime('%Y%m%d_%H%M%S') + '.tdms')
            Signals_new = readTDMS(path)
        else:
            Signals_new = getTDMS(
                timestamp=dt_local, location=location, root=root, filetype=filetype, site=site
            )
        for s1, s2 in zip(Signals, Signals_new):
            if s1.name == s2.name:
                s1.append(s2)
            else:
                raise NameError("Signallists in consecutive files does not match")
    return Signals


def readTDMSfolder(path):
    """ Function to import and merge all tdms files in a folder. Typically used for DV and 1P assessment measurements

    :param path: path to folder for which all files are to be imported
    :type path: str.
    """
    from os import listdir
    from pytz import utc
    files = listdir(path)
    dt_start = datetime.datetime.strptime(files[0][:-5], '%Y%m%d_%H%M%S')
    dt_start = dt_start.replace(tzinfo=utc)

    location = 'Turb'
    duration = len(files)
    signals = mergeTDMS(dt_start, duration, location, root=path, single_folder=True)

    return signals


def writeTDMS(Signals, path):
    # writes the list of DYNAwind.signal objects to a TDMS file
    pass


# %% Checks Strain Gauges (works for non opposing sensors)
def check_sg(Strains, treshold=0.1, plot_figure=True):
    """ Function to check strain gauges, that works for non opposing sensors

    For a practical guide : https://24seaa.atlassian.net/wiki/spaces/~wout.weijtjens/pages/449413176/Strain+gauge+check+functions
    """

    import numpy as np
    import itertools

    if not Strains:
        return
    Mtn = []
    Mtl = []
    N = []
    combi = []
    stat_dict = {"yaw/mean": 0}

    for strain in Strains:
        # Arbitrary values as this is not essential to the method
        strain.Ro = 5.000
        strain.Ri = 4.900
        stat_dict[strain.name + "/offset"] = strain.mean()
        stat_dict[strain.name + "/tc"] = 0

    for i in itertools.combinations(range(len(Strains)), 3):
        T = [Strains[j] for j in i]
        combi.append(i)
        tst1 = calcBendingMomentSignal(T, stat_dict)
        Mtn.append(tst1[-1])
        Mtl.append(tst1[-2])
        N.append(tst1[-3])

    Ic = np.pi / 4 * (Strains[0].Ro ** 4 - Strains[0].Ri ** 4)
    A = np.pi * (Strains[0].Ro ** 2 - Strains[0].Ri ** 2)
    Ri = Strains[0].Ri
    E = 210e9
    tst = np.zeros([len(Mtn), len(Strains)])
    i = 0
    heads = []
    for strain in Strains:
        j = 0
        head = strain.heading
        heads.append(head)
        head = (360 - head) / 180 * np.pi
        for n, l, normal, cmb in zip(Mtn, Mtl, N, combi):
            tst_strain = (
                    normal.data / A / E
                    + Ri / Ic / E * (l.data * np.sin(head) - n.data * np.cos(head)) * 1e6
            )
            ones = np.ones(len(Mtn[0].data))
            K = np.stack((tst_strain, ones), axis=1)
            theta = np.dot(np.linalg.pinv(K), strain.data - strain.mean())
            tst[j, i] = theta[0]
            j += 1
        i += 1
    error = np.mean(np.abs(1 + tst), axis=0)
    if plot_figure:
        plt.bar(range(len(Strains)), error, tick_label=heads)
        plt.ylim([0, 0.5])
        plt.hlines(treshold, -0.5, len(Strains) - 0.5, "r")
        plt.ylabel("Error")
        plt.xlabel("Strain gauge heading")
    return error


# %%  Plots results for opposing sensors


def checkOpposingSG(Strains, saveFig=False, path=".", tag=None, time_zoom=None):
    """ Function to check strain gauges, that works only for opposing sensors.

    If a function for non-opposing sensors is desired, consider :func:`dynawind.dynawind.check_sg`

    For a practical guide : https://24seaa.atlassian.net/wiki/spaces/~wout.weijtjens/pages/449413176/Strain+gauge+check+functions
    """
    from os.path import join
    import numpy as np

    f1 = plt.figure()
    Strains[0].plot()
    plt.plot(Strains[0].time(), -Strains[1].data)
    plt.grid(axis='both', which='both')
    plt.legend([Strains[0].name, Strains[1].name])
    if time_zoom is not None:
        plt.xlim(time_zoom)
    else:
        plt.xlim(100, 160)
    plt.ylabel("Strain (microstrain)")
    plt.xlabel('Time (s)')
    if saveFig:
        timestamp = Strains[0].timestamp.strftime("%Y%m%d_%H%M%S")
        file_name = "_".join(
            ["Timeseries", timestamp, Strains[0].location, 'Strain', 'LAT' + str(int(Strains[0].level)).zfill(3),
             str(int(Strains[0].heading))])
        if tag is not None:
            file_name += tag
        plt.savefig(join(
            path, file_name),
            dpi=300,
        )

    #
    f2 = plt.figure()
    plt.plot(Strains[0].data, Strains[1].data)
    plt.grid(axis='both', which='both')

    plt.plot(np.array([-15, 15]), np.array([15, -15]), "k-")
    plt.xlabel(Strains[0].name)
    plt.ylabel(Strains[1].name)

    # Determinate correction
    ones = np.ones(len(Strains[1].data))
    K = np.stack((Strains[1].data, ones), axis=1)
    theta = np.dot(np.linalg.pinv(K), Strains[0].data)
    plt.plot(np.array([-15 * theta[0], 15 * theta[0]]), np.array([-15, 15]), "r--")

    plt.text(5, 5, "Ratio: " + str(np.round(theta[0], 3)))

    if saveFig:
        file_name = "_".join(
            ["Comparisson", timestamp, Strains[0].location, 'Strain', 'LAT' + str(int(Strains[0].level)).zfill(3),
             str(int(Strains[0].heading))])
        if tag is not None:
            file_name += tag
        figPath = join(path, file_name)
        plt.savefig(figPath, dpi=300)

    f3 = plt.figure()
    Strains[0].plotPSD(xlim=(0, 5))
    Strains[1].plotPSD(xlim=(0, 5))
    plt.legend([Strains[0].name, Strains[0].name])
    if saveFig:
        file_name = "_".join(
            ["PSD", timestamp, Strains[0].location, 'Strain', 'LAT' + str(int(Strains[0].level)).zfill(3),
             str(int(Strains[0].heading))])
        if tag is not None:
            file_name += tag
        plt.savefig(join(
            path, file_name),
            dpi=300,
        )

    h = [f1, f2, f3]
    return h


# %%
def stat4signal(signal, parameter, exact_timestamp=True, db_con=None):
    """ Function to use the information in a signal object to establish a database connection and obtain corresponding parameter for the same location and timestamp.
    Typically used for retreing SCADA data associated with a particular signal. Returns numpy NaN when no record is found

    :param signal:
    :param parameter:
    :param exact_timestamp: enforce that the found parameter is exactly on the same timestamp as the parameter
    :type exact_timestamp: Bool.
    :param db_con: dynawind database class to an OPEN database connection to interact with. Using this speeds up the process as the database connection is not continuously opened and closed."""
    from numpy import nan
    from dynawind.db import Postgres

    if isinstance(signal, list):
        signal = signal[0]

    dt = signal.timestamp + datetime.timedelta(0, 600)  # Statistics are found in the next database entry

    if db_con is None:
        try:
            db_con = Postgres(signal.site)
            results = db_con.query(dt, [signal.location], [parameter])
            db_con.close()
        except:
            return nan
    else:
        try:
            results = db_con.query(dt, [signal.location], [parameter])
        except:
            return nan
    #
    if exact_timestamp:
        if results[0][1] == dt:
            stat = results[0][-1]
        else:
            stat = nan
    else:
        stat = results[0][-1]

    if stat is None:
        stat = nan

    return stat


# %%
def temperature_compensation(
        Signal=None,
        temperature=None,
        temperature_sensor=None,
        temp_coef=None,
        T_ref=None,
        group=None,
):
    """ Performs temperature compensation on Signal objects """

    def tempCompensation(T, coeficients, T_ref=0):
        # Default behavior is Coef[0]*(T-T_ref)^n+Coef[1]*(T-T_ref)^(n-1)+...+Coef[n-1]*(T-T_ref)+Coef[n]
        dT = T - T_ref
        n = len(coeficients) - 1
        tc = coeficients[0] * (dT ** n)
        for i in range(1, len(coeficients)):
            tc += coeficients[i] * (dT ** (n - i))
        return tc

    # Load default coeficients for temperature compensation
    if temp_coef is None:
        if group == "fiber":
            T_ref = 22.5
            # linear = S1/k+(alpha_t-alpha_f)
            linear = 6.37 / 0.772 + (12 - 0.552)
            # quadratic = S2/k
            quadratic = 0.00746 / 0.772
            temp_coef = [quadratic, linear, 0]
    # Temperature
    if temperature is None:
        # Should pull the temperature from the temperature_sensor from the JSON file
        raise ValueError(
            "temperature value not provided, automatic temperature retrieval not yet a feature"
        )

    tc = tempCompensation(temperature, temp_coef, T_ref=T_ref)
    if Signal is not None:
        Signal.data = Signal.data - tc
        Signal.temperature_compensation = tc
        return Signal
    else:
        return tc

# %% Allow for command line call
# if len(sys.argv) > 1:
# parseCLIarguments(sys.argv[1:])
