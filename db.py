"""
dynawind.db
===================

Overview
-------------------
The dynawind.db module handles all functionality with relation to interacting with databases.


Definitions
-------------------

"""
import datetime
import json
import os
import pandas as pd
import numpy as np
import progressbar
import psycopg2 as postgres
import pytz


class Postgres(object):

    """ Class that handles the main interactions with the postgres database. On init will use
     :func:`dynawind.db.connect2postgreSQL` to connect to the postgres database.
    
    This class is typically used within a with/as construction, to guarantee the database connection is closed
     after completion. Following example pulls data from Northwind database

    
     >>>   with Postgres('Northwind') as nw_db:
                 results = nw_db.query((dt_start, dt_stop), locations=['NW' + turb], parameters=param, as_dataframe=True)


    :param site: the site you connect to e.g. belwind
    :type site: str.
    :param ini_file: direct path to a specific .ini file that contains the database settings. By default the postprocessing ini-file of the site will be loaded.
    :type ini_file: str.
    """

    def __init__(self, site, ini_file=None):
        self.site = site
        self.ini_file = ini_file

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def delete(self, timestamp, location, parameter, table=None, commit=False):
        """
        Delete an entry from the database, based on timestamp, location and parameter

        :param timestamp: timestamp to remove from database
        :type timestamp: datetime.datetime
        :param location: location, e.g. BBC01
        :param parameter: pr
        :param table: Table of the database you wish to delete the data from
        :type table: str.
        :param commit: commit to the database
        :type commit: Boolean
        """
        if table is None:
            table = self.table

        # Read single entry
        result = self.query(dt=timestamp, locations=[location], table=table)[0]
        if result[1] == timestamp and parameter in result[2]:
            data = result[2]
            # Remove parameter from metrics
            data.pop(parameter)
            # Insert metrics again
            self.update(timestamp, self.site, location, data, table=table)
            if commit:
                self.commit()


    def insert(self, timestamp, site, location, data, table=None):
        if table is None:
            table = self.table
        sql = (
                """INSERT INTO """
                + table
                + """ (site,location,timestamp,metrics)
                VALUES (%(site)s,%(location)s,%(timestamp)s,%(jsonobj)s) ON CONFLICT DO NOTHING"""
        )
        self.cur.execute(
            sql,
            {
                "site": site,
                "location": location,
                "timestamp": timestamp,
                "jsonobj": json.dumps(data),
            },
        )

    def insert_json(self, path, table=None, action="insert", progress_bar=True):
        """ Process json files to Postgres database. The action parameter defines whether :func:`dynawind.db.Postgres.insert` or :func:`dynawind.db.Postgres.update` is used
        
        :param path: Either path to the json files you wish to process or a list of files you which to process
        :type path: str.
        :param table: table of the database you wish to insert the data into
        :type table: str.
        :param action: either 'insert', 'update' or 'upsert'. Insert will not overwrite existing database entries, while update will only overwrite existing entry. Upsert will do an update when an entry already exists and otherwise an insert..
        :type action: str.
        :param progress_bar: Boolean to indicate whether a progressbar should be plotted
        """
        if isinstance(path, list):
            filelist = path
        else:
            if os.path.isdir(path):
                # Make list of all files in filePath
                filelist = []
                for root, dirs, files in os.walk(path, topdown=False):
                    for name in files:
                        filelist.append(os.path.join(root, name))
            else:
                filelist = [path]
        if progress_bar:
            pbar = progressbar.ProgressBar(max_value=len(filelist))
            progress = 0
        for file in filelist:

            f = open(file)
            data = json.load(f)
            f.close()
            site = data[0].pop("site")
            timestamp = data[0].pop("timestamp")
            location = data[0].pop("location")
            if action == "insert":
                self.insert(timestamp, site, location, data[0], table=table)
            elif action == "update":
                self.update(timestamp, site, location, data[0], table=table)
            elif action == "upsert":
                self.upsert(timestamp, site, location, data[0], table=table)
            if progress_bar:
                progress += 1
                pbar.update(progress)

        self.commit()

    def insert_period(self, location, start_dt, stop_dt, root='.', action="insert"):
        """ Wrapper for insert_json to insert a specific period of data into the database """
        dt_list = make_dt_list(start_dt, stop_dt, interval=600)  # Single days
        filelist = []
        for dt in dt_list:
            path = returnJSONfilePath(dt, self.site, location, root=root, fileext=".json")
            if os.path.isfile(path):
                filelist.append(path)

        self.insert_json(filelist, action=action)

    def update(self, timestamp, site, location, data, table=None):
        if table is None:
            table = self.table

        sql = (
                """UPDATE """
                + table
                + """ SET metrics = %(jsonobj)s WHERE location=%(location)s AND timestamp=%(timestamp)s"""
        )
        self.cur.execute(
            sql,
            {
                "site": site,
                "location": location,
                "timestamp": timestamp,
                "jsonobj": json.dumps(data),
            },
        )

    def upsert(self, timestamp, site, location, data, table=None):
        if table is None:
            table = self.table
        sql = (
                """INSERT INTO """
                + table
                + """ (site,location,timestamp,metrics)
                VALUES (%(site)s,%(location)s,%(timestamp)s,%(jsonobj)s) ON CONFLICT (location,timestamp) DO UPDATE"""
                + """ SET metrics = %(jsonobj)s"""
        )
        self.cur.execute(
            sql,
            {
                "site": site,
                "location": location,
                "timestamp": timestamp,
                "jsonobj": json.dumps(data),
            },
        )

    def cluster(self, table=None, index=None):
        """ clusters the database table based on the default location_timestamp index 
        
        :param table: table to cluster
        :type table: str.
        :param index: by default a location and timestamp index is used. But some postgres tables might only have a location or timestamp index. So if that is the case either type 'location' or 'timestamp' as index.
        :type index: str.
 
        """
        if table is None:
            table = self.table
        if index is None:
            idx = """_location_timestamp_idx"""
        else:
            idx = """_""" + index + """_idx"""

        sql = (
                """CLUSTER """
                + table
                + """ USING """
                + table
                + idx
        )
        self.execute(sql)
        self.commit()

    def find_missing_entries(self, dt_range: tuple, location: str, parameter: str):
        """
        Returns a list of timestamps for which no entry of parameter could be found in the database.

        :param dt_range: tuple of timestamps between which the database is checked
        :param location: location, e.g. 'BBC01'
        :param parameter: parameter
        :return: list of datetime.datetime
        """

        results = self.query(dt_range, locations=[location], parameters=[parameter], as_dataframe=True)

        results = results.resample('10T').asfreq()  # fill in empty elements
        index = results[parameter].index[results[parameter].apply(np.isnan)]
        dt_list = [pd.to_datetime(ind).astimezone(pytz.utc) for ind in index]

        return dt_list

    def open(self):
        (self.conn, self.cur, self.table) = connect2postgreSQL(self.site, self.ini_file)

    def execute(self, sql):
        self.cur.execute(sql)

    def commit(self):
        self.conn.commit()

    def close(self):
        """ Close database connection """
        self.cur.close()
        self.conn.close()

    def query(self, dt=None, locations=None, parameters=None, conditions=None, table=None, as_dataframe=False):
        """ Primary function to query the postgres database, will generate and
        execute the sql commands
        
        :param dt: timestamp or tuple of timestamps. If a single timestamp is provided the closest database entry to the timestamp is returned. When a tuple of timestamps is entered all records between the two timestamps are returned
        :type dt: datetime.datetime
        :param locations: locations to query from table
        :type locations: list of str.
        :param parameters: parameters to query from table
        :type parameters: list of str.
        :param conditions: Additional conditions to be include after the WHERE in the SQL query
        :type conditions: list of SQL str.
        :param table: 
        :type table: str.
        :param as_dataframe: when true returns a pandas dataframe
        :type as_dataframe: bool.
        :returns: either list or dataframe (as_dataframe=True)
        """

        def add_where(sql, where_bool):
            if where_bool:
                sql = sql + """ AND"""
            else:
                sql = sql + """ WHERE"""
                where_bool = True
            return sql, where_bool

        if table is None:
            table = self.table
        # Parameters
        sql = """SELECT location, timestamp"""
        if parameters is None:
            sql = sql + ", metrics"
        else:
            for prm in parameters:
                if 'case' in prm:
                    sql = sql + """, metrics->>'""" + prm + """'"""
                elif 'regr_' in prm:
                    sql = sql + """, """ + prm.lower() + """(""" + table + """)"""
                else:
                    sql = sql + """, cast(metrics->>'""" + prm + """' as double precision)"""

        #
        sql = sql + " FROM " + table
        # Conditions
        where = False
        if locations is not None:
            sql, where = add_where(sql, where)
            sql = sql + """ location = '""" + locations[0] + """'"""
        if conditions is not None:
            for cond in conditions:
                sql, where = add_where(sql, where)
                sql = sql + ' ' + cond

        if dt is not None:
            if isinstance(dt, datetime.datetime):
                # when you query a single date
                sql = (
                        sql
                        + """ order by abs(extract(epoch from (timestamp - timestamp with time zone '"""
                        + str(dt)
                        + """'))) limit 1"""
                )
            elif isinstance(dt, tuple) and len(dt) == 2:
                # query a time range
                sql, where = add_where(sql, where)
                sql = (sql + """ timestamp BETWEEN '""" + str(dt[0]) + """' AND '""" + str(dt[1]) + """'""")
            else:
                raise TypeError('dt should be either datetime or tuple of length 2')

        self.cur.execute(sql)
        query_output = self.cur.fetchall()
        if as_dataframe:
            return query_output_to_dataframe(query_output, parameters)
        else:
            return query_output


def query_output_to_dataframe(query_output, parameters):
    """
    Converts the outputs of :func: `dynawind.db.Postgres.query` to a pandas dataframe
    
    :param parameters:
    :param query_output: ouput of the query
    :rtype: pandas.DataFrame
    """
    from pytz import utc
    cols = ['location', 'timestamp']
    if parameters is None:
        cols.append('metrics')
    else:
        cols.extend(parameters)

    df = pd.DataFrame(query_output, columns=cols)
    df.set_index('timestamp', inplace=True)

    return df


def roundedTimestamp(dt=None, delay=1):
    """ Rounds timestamp to nearest multiple of 10 minutes
    
    :param delay:
    :param dt: timestamp to round, if None this will use the current time
    """
    from math import floor
    import pytz

    """
   Produces a timeobject that represents the files to be processed when called
   """
    if dt is None:
        dt = datetime.datetime.utcnow()  # Time in UTC
    flooring = dt.minute - floor(dt.minute / 10) * 10
    dt = dt - datetime.timedelta(
        minutes=flooring, seconds=dt.second, microseconds=dt.microsecond
    )  # Rounding
    dt = dt - datetime.timedelta(minutes=delay * 10)  # Delay
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt


def make_dt_list(start_dt, stop_dt, interval=600):
    """ Generates a list of dt between a start_dt and a stop_dt
    
    :param start_dt:
    :param stop_dt:
    :param interval: Interval in seconds between each step, default is 10min
    """
    nr_steps = int((stop_dt - start_dt).total_seconds() / interval)
    dt_list = [start_dt + datetime.timedelta(0, interval * x) for x in range(nr_steps)]

    return dt_list


def clean_dict(my_dict):
    """ Processes dict to allow for storage into JSON
    
    :param my_dict: dictionary containing data for storage.
    :type my_dict: dict.
    :returns: cleaned dictionary
    :rtype: dict.
    """
    popkeys = []
    for key in my_dict:
        if my_dict[key] == "nan" or my_dict[key] == "NAN":
            popkeys.append(key)
        elif not isinstance(my_dict[key], str) and not isinstance(my_dict[key], dict) and np.isnan(my_dict[key]):
            popkeys.append(key)

    for key in popkeys:
        my_dict.pop(key)

    return my_dict


# %% return TDMS path
def returnTDMSpath(
        site, location, datasubtype, datatype="TDD", dt=None, root=r"\\192.168.119.14"
):
    """ Returns the path to the tdms file
    
    :param site: site of the file to be considered, e.g. belwind
    :type site: str.
    :param location: location of the file to be considered, e.g. BBC01
    :type location: str.
    :param datasubtype: datasubtype, e.g. 'strain','acc',...
    :param datatype: str
    :param datatype:
    :param dt: timestamp of the file to retrieve
    :type dt: datetime.datetime
    :param root: root of the filestructure containing the tdms files, by default NAS4
    :type root: str.
    :returns: path to a tdms file
    :rtype: str.
    """
    import os

    if dt is None:
        dt = roundedTimestamp(dt=None, delay=1)
    else:
        dt = dt.astimezone(pytz.utc)

    if "TDD_" not in datasubtype:
        datasubtype = "TDD_" + datasubtype

    timestr = dt.strftime("%Y%m%d_%H%M%S")  # ' 1:36PM EDT on Oct 18, 2010'
    if site is None:
        filePath = os.path.join(
            root,
            location,
            datatype,
            datasubtype,
            dt.strftime("%Y"),
            dt.strftime("%m"),
            dt.strftime("%d"),
            timestr + ".tdms",
        )
    else:
        filePath = os.path.join(
            root,
            "data_primary_" + site.lower(),
            location,
            datatype,
            datasubtype,
            dt.strftime("%Y"),
            dt.strftime("%m"),
            dt.strftime("%d"),
            timestr + ".tdms",
        )
    return filePath


# %% JSON HANDLING


def initJSON(jsonPath, dt, site, location):
    jsonFile = open(jsonPath, "w")
    data_dict = dict()
    data_dict["location"] = location
    data_dict["site"] = site
    data_dict["timestamp"] = dt.__str__()
    json.dump([data_dict], jsonFile, indent=2)
    jsonFile.close()


def returnJSONfilePath(dt, site, location, root=".", fileext=".json", init_json=True):
    """
    Returns the path to a JSON file
    
    :param fileext:
    :param dt: timestamp for which the JSON filepath has to be retreived
    :type dt: datetime.datetime
    :param site: Site, e.g. Belwind
    :param location: Location, e.g. BBC01
    :param root: JSON root folder from which the filepath has to bepr retreived
    :param init_json: Indicate whether the function should initiate an empty json file when the path does not exist (default behavior)
    :type init_json: Boolean
    :returns: path to a json file 
    """
    import os

    Folder = os.path.join(
        root, site, location, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    )
    if not os.path.isdir(Folder):
        os.makedirs(Folder)
    filePath = os.path.join(Folder, dt.strftime("%Y%m%d_%H%M%S") + fileext)
    if (not os.path.isfile(filePath)) & init_json:
        initJSON(filePath, dt, site, location)
    return filePath


def delete_from_json(dt, site, location, parameters, root='.'):
    """ 
    function to remove parameters from existing JSON files
     :param parameters:
     :param dt: timestamp for which the JSON filepath has to be retreived
    :type dt: datetime.datetime
    :param site: Site, e.g. Belwind
    :param location: Location, e.g. BBC01
    :param root: JSON root folder from which the filepath has to bepr retreived
   """
    path = returnJSONfilePath(dt, site, location, root=root, init_json=False)
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        for par in parameters:
            if par in data[0]:
                data[0].pop(par)
        jsonFile = open(path, "w")
        json.dump(data, jsonFile, indent=2)
        jsonFile.close()


# %% POSTGRESQl
def insertJSON2PostgreSQL(filePath, action="INSERT", cluster=False, table=None):
    # LEGACY!!!!, use dynawind.db.Postgres instead
    # table can be specified, if not will result in default
    import os

    if os.path.isdir(filePath):
        # Make list of all files in filePath
        filelist = []
        for root, dirs, files in os.walk(filePath, topdown=False):
            for name in files:
                filelist.append(os.path.join(root, name))
    else:
        filelist = [filePath]

    # %% Step 1 : Read config based on first file
    f = open(filelist[0])
    data = json.load(f)
    f.close()

    site = data[0].pop("site")

    # %% Open connection
    (conn, cur, table_config) = connect2postgreSQL(site)
    if table is None:
        table = table_config

    if action == "INSERT":
        # %% Write INSERT QUERY
        sql = (
                """INSERT INTO """
                + table
                + """ (site,location,timestamp,metrics)
        VALUES (%(site)s,%(location)s,%(timestamp)s,%(jsonobj)s) ON CONFLICT DO NOTHING"""
        )
    elif action == "UPDATE":
        sql = (
                """UPDATE """
                + table
                + """ SET metrics = %(jsonobj)s WHERE location=%(location)s AND timestamp=%(timestamp)s"""
        )

    ind = 0
    # Step 4 : Run through all files and execute SQL
    for file in filelist:
        f = open(file)
        ind += 1
        print(str(ind) + "/" + str(len(filelist)), end="\r")
        data = json.load(f)
        f.close()
        timestamp = data[0].pop("timestamp")
        location = data[0].pop("location")
        cur.execute(
            sql,
            {
                "site": site,
                "location": location,
                "timestamp": timestamp,
                "jsonobj": json.dumps(data[0]),
            },
        )

    # %% Step 5 : Commit and close connection to postgresSQL database
    if cluster:
        sql = (
                """CLUSTER """
                + table
                + """ USING """
                + table
                + """_location_timestamp_idx"""
        )
        cur.execute(sql)

    conn.commit()
    cur.close()
    conn.close()
    print(
        str(len(filelist))
        + " files transfered into "
        + table
        + " for location "
        + location,
        end="\r",
    )


def pullValuesfromJSON(site, location, dt, parameters=None, root=".", fileext=".json"):
    """
     Returns data in JSON as a dict
    """
    dt = dt.astimezone(pytz.utc)

    jsonPath = returnJSONfilePath(dt, site, location, root=root, fileext=fileext)
    f = open(jsonPath)
    data = json.load(f)
    f.close()
    data_expt = dict()
    if parameters is None:
        parameters = data[0].keys()
    for par in parameters:
        if par in data[0]:
            data_expt[par] = data[0][par]
        else:
            data_expt[par] = np.empty(1)
    return data_expt


# %%
def connect2postgreSQL(site, ini_file=None):
    """ Legacy, use class dynawind.db.Postgres instead"""
    import configparser
    import pkg_resources

    if ini_file is None:
        resource_package = __name__
        # Step 1 : load site config file
        resource_path = "/".join(
            ("config", site.lower(), site.lower() + "_postprocessing.ini")
        )
        ini_file = pkg_resources.resource_filename(resource_package, resource_path)

    config = configparser.ConfigParser()
    config.read(ini_file)
    if 'postgreSQL' in config:
        # %% Step 2 Make connection to postgres database
        conn = postgres.connect(
            host=config["postgreSQL"]["host"],
            port=config["postgreSQL"]["port"],
            database=config["postgreSQL"]["database"],
            user=config["postgreSQL"]["user"],
            password=config["postgreSQL"]["password"],
        )
        cur = conn.cursor()
        table = config["postgreSQL"]["table"]
        return conn, cur, table
    else:
        raise FileNotFoundError(
            'Postprocessing configuration for ' + site + ' not found or does not contain key on PostgreSQL')


def check_file_availability(locations, file_type, dt_start, dt_stop):
    """ Simple function to check the number of tdms files available. Returns a pandas DataFrame that has the number of files for each day.
    
    :param locations: List of locations
    :param file_type: File type to consider
    :param dt_start: start of period
    :param dt_stop: stop of period
    """
    from dynawind.dynawind import getTDMSpath

    date_list = make_dt_list(dt_start, dt_stop, interval=600 * 144)
    df = pd.DataFrame(index=date_list, columns=locations)
    df = df.fillna(0)  # with 0s rather than NaNs

    for location in locations:
        for dt in date_list:
            path = getTDMSpath(dt, 'lvdt', location)
            path = os.path.dirname(path)
            df[location][dt] = len(next(os.walk(path))[2])

    return df


def calc_availability(df, param, dt_start, dt_stop):
    """
    Calculates the availability for a given period, as defined by dt_start and dt_stop
    
    :param dt_start: Start timestamp
    :type dt_start: datetime.datetime
    :param dt_stop: Stop timestamp
    :type dt_stop: datetime.datetime
    :param df: pandas Dataframe
    :param param: parameter to base the availability on
    :return: Data Availability (%)
    """
    from numpy import isnan
    n_all = int((dt_stop - dt_start).total_seconds() / 600)
    avail = sum(~isnan(df[param][dt_start:dt_stop]))
    return avail / n_all * 100


def calc_monthly_availability(df, param, dt_start: datetime.datetime, dt_stop: datetime.datetime):
    """
    Calculates the availability for each month (in percent)
    :param df: DataFrame on which the data-availability will be calculated
    :param param: Parameter to be considered for the calculation of the availability
    :param dt_start: Start timestamp
    :param dt_stop: Stop timestamp
    :return:
    """
    import datetime
    from pytz import utc
    months = [datetime.datetime(m // 12, m % 12 + 1, 1) for m in
              range(dt_start.year * 12 + dt_start.month - 1, dt_stop.year * 12 + dt_stop.month)]
    avail = []
    for dt in months:
        dt = dt.replace(tzinfo=utc)
        end_of_month = dt.replace(day=28) + datetime.timedelta(days=4)
        end_of_month = end_of_month - datetime.timedelta(days=end_of_month.day - 1)
        avail.append(calc_availability(df, param, dt, end_of_month))

    return months, avail


def monthly_availability_plot(df, locations, parameters, target=90, colormap='GnBu'):
    """ Generates the monthly_availability plot 
    
    :param df: Pandas Dataframe to base results on
    :param locations: List of locations to consider (mainly for labeling)
    :param parameters: List of parameters to use to calculate the availability
    :param target: Target availability to display, default: 90%'
    :param colormap: str. defining a colormap
    
    :rtype: dynawind.report.figure handle
    """
    from matplotlib.cm import get_cmap
    from numpy import linspace, array
    from dynawind.report import Figure
    import matplotlib.pyplot as plt
    cmap = get_cmap(colormap)
    fig = Figure(figsize=(8, 8), caption='Availability plot', filename='DW_Monthly_availability_plot')
    ind = 0

    dt_start = df.index[0]
    dt_stop = df.index[-1]

    bar_height = 1 / len(locations) - 0.1
    for location, prm in zip(locations, parameters):
        months, m_avail = calc_monthly_availability(df, prm, dt_start, dt_stop)
        ms = [(m.year - dt_start.year) * 12 + m.month + (len(locations) - 1) * bar_height - ind * bar_height for m in
              months]
        ms_ticks = []
        m_labels = []
        for mms, m in zip(ms, months):
            if not (mms - 1) % 12:
                plt.axhline(mms - 0.2, color='black', linestyle='dotted')
                ms_ticks.append(mms - 0.2)
                m_labels.append(str(m.year) + '   ')
            ms_ticks.append(mms)
            m_labels.append(m.month)
        plt.barh(ms, array(m_avail), height=bar_height, align='center', label=location,
                 color=cmap(linspace(0, 1, len(locations) + 1)[ind + 1]))
        ind += 1
    plt.yticks(ms_ticks, m_labels)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
    plt.axvline(x=target, color='red', linestyle='dashed')
    plt.gca().set_axisbelow(True)
    plt.xlim([0, 100])
    plt.gca().grid(color='gray', linestyle='dashed')
    plt.xlabel('Availability (%)')
    return fig


def calendar_plot(df, param, dt_start, dt_stop, caption='Calender plot', target=90):
    """
    Allows to make a calendar plot. This is a plot where each day is represented by a colored square. The color of the
    square depends on the data availability of that day. Parameter target sets the value when a dataset is considered
    good (blue). A square is green when data availability = 100%

    :param df: DataFrame on which the results are based
    :param param: Parameter on which the calculation is made
    :type param: str
    :param dt_start: Start timestamp
    :param dt_stop: Stop timestamp
    :param caption: Figure Caption
    :param target: Target value (%)
    :return: Figure
    :rtype: dynawind.report.Figure
    """
    from dynawind.report import Figure
    import matplotlib.pyplot as plt

    df['avail'] = ~np.isnan(df[param])
    df.index = pd.to_datetime(df.index, utc=True)
    results_daily = df['avail'].resample('24h').sum()

    start_year = dt_start.year
    start_month = dt_start.month

    new_tick_locations = []
    no_months = round((dt_stop - dt_start).days / 30)
    Fig1 = Figure(figsize=(8, 6 / 15 * no_months), caption=caption,
                  filename='DW_Calender_availability_plot' + '_' + param)
    fig = plt.gcf()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    for i in range(0, 32):
        if i % 2 != 0:
            new_tick_locations.append(i)
    for index, row in results_daily.iteritems():
        y = index.year - start_year
        d = index.day
        m = index.month - start_month + y * 12
        if row == 144:
            ax1.plot(d, m, 's', markerfacecolor='mediumseagreen', markeredgecolor='k', markeredgewidth=0.5)
        elif row / 1.44 > target:
            ax1.plot(d, m, 's', markerfacecolor='DodgerBlue', markeredgecolor='k', markeredgewidth=0.5)
        elif row == 0:
            ax1.plot(d, m, 's', markerfacecolor='red', markeredgecolor='k', markeredgewidth=0.5)
        else:
            ax1.plot(d, m, 's', markerfacecolor='orange', markeredgecolor='k', markeredgewidth=0.5)
    y_labels = []
    dt = dt_start
    for i in range(m + 1):
        dt = dt + datetime.timedelta(days=33)
        dt = dt - datetime.timedelta(days=dt.day)
        month_str = dt.strftime('%b \'%y')
        y_labels.append(month_str)

    ax1.set_xlim([0.5, 31.5])
    ax2.set_xlim(ax1.get_xlim())
    ax1.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8)
    ax1.set_xticks(new_tick_locations)
    ax2.set_xticks(new_tick_locations)

    ax1.grid(which='both', axis='both', linestyle='dotted')
    ax1.set_yticks(range(m + 1))
    ax1.set_yticklabels(y_labels)
    return Fig1
