"""
dynawind.report
===================
Overview
-------------------
The DYNAwind report module handles the functions and classes associated with making reports.

Key classes in the DYNAwind Report module are:

    * :class:`dynawind.report.Report`
    * :class:`dynawind.report.Project`

Definitions
-------------------
"""
import itertools
import os
import matplotlib.pyplot as plt
import dynawind.dynawind as dw

class Report(object):
    """
    DYNAwind report class is the basic tool for reporting results
    
    :param path: Path to store the report
    :type path: str.
    :param project: Project associated with the report
    :type project: :class:`dynawind.report.Project`
    :param author: Author of the report
    :type author: str.
    :param location: Location to which the report applies, e.g. BBC01
    :type location: str.
    :param version: version code of the report
    :param document_nr: 
    :type document_nr: int
    :param document_type: Should refer to confluence page explaining document types
    :param title: title of the report, if None then :func:`dynawind.report.Report.make_title` is used
    :type title: str.
    :param titlepage: Add title page to report
    :type titlepage: bool.
    :param description: Report description (for titlepage)
    :type description: str.
    :parameter client_reference: client's reference to this report
    :type client_referece: str.
    :parameter template: specifies which template file is to be used
    :type template: str.
    :type document_type: str.
    :parameter add_header: Add the header to the beginning of the file
    """
    def __init__(
        self,
        path,
        project,
        author="Wout Weijtjens",
        location="nA",
        version=None,
        document_nr=1,
        document_type="AR",
        title=None,
        date = None,
        titlepage = False,
        description = "",
        client_reference=None,
        template="24SEA_Report_Template",
        add_header=True
    ):
        self.path = path
        self.author = author
        self.title = title
        self.document_type = document_type
        self.location = location
        self.project = project
        self.date = date
        self.document_nr = document_nr
        self.version = version
        self.client_reference = client_reference
        if title is None:
            self.title = self.make_title()
        if version is None:
            self.version = 1
        self.titlepage = titlepage
        self.description = description
        if titlepage and description == "":
            raise ValueError('Please provide a description of the report (field: discription)')
        self.body = []
        self.filename = self.title
        self.template = template
        if add_header:
            self.add_header()

    def make_title(self):
        """ Generates a structured title containing the parameters associated with the project. For monthly reports (MR) the file name is code_supplier_MR_MonthYear otherwise code_supplier_doc-type_doc-nr
        
        :returns: structured title in the form ``code_supplier_doc-type_doc-nr``
        :rtype: str.
        
        """
        
        if self.document_type == 'MR' and not self.date is None:
            # Monthly report
            title = "_".join(
                [
                    self.project.code,
                    self.project.supplier_reference(),
                    self.document_type,
                    self.date.strftime('%B%Y'),
                ]
            )
        else:
            title = "_".join(
                [
                    self.project.code,
                    self.project.supplier_reference(),
                    self.document_type,
                    str(self.document_nr).zfill(3),
                ]
            )
        return title

    def section(self, section_title):
        """ start new section, this will also impose a FloatBarrier in LaTeX
        
        :param section_title: title of the section
        :type section_title: str.
        """
        self.write("\\FloatBarrier{}\n")
        self.write("\\section{" + section_title + "}\n")
        self.write("\\FloatBarrier{}\n")

    def subsection(self, section_title):
        """ start new subsection 
        
        :param section_title: title of the section
        :type section_title: str.
        """
        self.write("\\subsection{" + section_title + "}\n")

    def subsubsection(self, section_title):
        """ start new subsubsection 
        
        :param section_title: title of the section
        :type section_title: str.
        """
        self.write("\\subsubsection{" + section_title + "}\n")

    def add_header(self):
        self.write("\\documentclass{" + self.template + "}\n")
        if self.client_reference is not None:
            self.write("\\ClientReference{"+self.client_reference+"}\n")
        self.write("\\definecolor{LightGray}{RGB}{180,180,180}\n")
        self.write("\\ProjectCode{" + self.project.code + "}\n")
        self.write("\\Project{" + self.project.name + "}\n")

        self.write(
            "\\FileReference{"
            + self.document_type
            + "}{"
            + str(self.document_nr).zfill(3)
            + "}\\par\n"
        )
        self.write("\\telephone{0032/2629 23 90}\n")
        self.write("\\newcommand{\\turb}{" + self.location + "}\n")
        self.write("\\Version{" + str(self.version) + "}\n")
        self.write("\\Client{" + self.project.client + "}\n")
        self.write("\\ReportDate{\\today}\n")
        self.write("\\author{" + self.author + "}\n")
        self.write("\\title{" + (self.title).replace('_','\\_') + "}\n")
        self.write("\\ReportDescription{" + self.description + "}\n")
        self.write("\\begin{document}\n")
        if self.titlepage:
            self.write("\\maketitle\n")


    def add_table(self, table, longtable=False, location = None, sideways=False):
        """ Add table to report 
        
        :param table: table to add to report
        :type table: :class:`dynawind.report.Table`
        :param location: LaTeX location specification, e.g. h!, t, b, ...
        :type location: str
        :param sideways: Indicates for a sideways table or not
        :type sideways: Boolean
        """
        if not longtable:
            if sideways:
                tab_str = 'sidewaystable'
            else:
                tab_str = 'table'
            head_str = "\\begin{"+tab_str+"}"
            if location is not None:
                head_str += '['+location+']'
            self.write(head_str+"\n")
            self.write("\\centering\n")
        self.write(table.write())
        self.add_caption(table.caption_str)
        if not longtable:
            self.write("\\end{"+tab_str+"}\n")

    def add_caption(self, caption_str):
        """ Add caption string to report
        
        :param caption_str: caption string to add to report
        :type caption_str: str.
        """
        if caption_str is not None:
            self.write("\caption{" + caption_str + "}\n")

    def add_figure(self, figure, sideways=False, width=0.8, fontsize=18, tick_fontsize=18, position='h!', caption=None, borders='tight', close=True):
        """ Adds a Figure object to the report. Takes several inputs associated with the LaTeX layout.
        
        :param figure: Figure to add to the report, if a list of Figures is parsed the items are plot as subfigures
        :type figure: :class:`dynawind.report.Figure` or List
        :param width: if <1, this is relative to textwidth, else in cm. Defaults to 0.8 for normal and 1.0 for sideways
        :param sideways: Boolean to specify that the figure has to be put in landscape
        :param fontsize: LaTeX fontsize to use for axes labels
        :param tick_fontsize: LaTeX fontsize to use for tick labels
        :param position: LaTeX position specifier, default 'h!'
        :param caption: Caption of figure, if None the  ´´caption´´ argument of the :class:`dynawind.report.Figure` object is used. For subplots this is the caption for the main figure, the caption of the subfigures is taken from the ´´caption´´ argument
        :param borders: Either 'tight' (default) or 'fixed'. Fixed will no interfere with the current borders. 
        :param close : Close figure after adding it to the report
        :type close: Boolean
        """
        if isinstance(figure, Figure):
            figures = [figure]
            single_figure = True
        elif isinstance(figure, list):
            figures = figure
            single_figure = False
        else:
            raise TypeError('input to add_figure should be dynawind.report.Figure or list')
        
        # Prepping the layout of the figures
        for figure in figures:
            figure_path = os.path.join(self.path, "Figures")
    
            if not os.path.exists(figure_path):
                os.makedirs(figure_path)
    
            figure_path = os.path.join(figure_path, figure.filename + ".png")
            
            plt.figure(figure.fig_handle.number)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.ylabel(plt.gca().get_ylabel(), fontsize=fontsize + 2)
            plt.xlabel(plt.gca().get_xlabel(), fontsize=fontsize + 2)
    
            figure.save(figure_path, borders=borders)
            if close:
                figure.close()
        # writing the LaTeX
        if sideways:
            figure_type_str = 'sidewaysfigure'
            if width == 0.8:
                width = 1
        else:
            figure_type_str = 'figure'
        self.write("\\begin{"+figure_type_str+"}["+position+"]\n")
        self.write("\\centering\n")
        for figure in figures:
                
                
            includegraphics_settings = []
            if width <= 1:
                includegraphics_settings.append("width=" + str(width) + "\\textwidth")
            else:
                includegraphics_settings.append("width=" + str(width) + "cm")
            figure_str = "\\includegraphics["
    
            for setting in includegraphics_settings:
                figure_str = figure_str + setting
            figure_str = figure_str + "]{./Figures/" + figure.filename + ".png" + "}"
            if not single_figure:
                figure_str = "\\subfigure["+figure.caption_str+"]{"+figure_str+"}"
            figure_str = figure_str + "\n"
            self.write(figure_str)
        
        if caption is None:
            self.add_caption(figure.caption_str)
        else:
            self.add_caption(caption)
        self.write("\\end{"+figure_type_str+"}\n")
        
    def add_pages(self,pages):
        """ adds pages to the report 
        
        :parameter pages: List of pages to add (i.e. other report objects)
        :type pages: List
        """
        for page in pages:
            self.write(page.body)

    def add_sensor_status_report(self, signals):
        """ Function that adds a sensor status table to the report based on the status and comment in the config files
        
        :param signals: list of signals for which the table will be included
        :type signals: list of :class:`dynawind.dynawind.Signal`
        """
        groups = set([x.group for x in signals])
        self.write("\\begin{longtable}{c|ll}\n")
        self.write("\\centering\n")
        self.write("&\\bf{Sensor}&\\bf{Comment}\\\\\n")  
        self.write("\\hline{}\n")
        self.write("\\endhead % all the lines above this will be repeated on every page\n")
        for group in groups:
            self.write("&&\\\\\n") 
            self.write("&\\bf{"+group.title()+"}&\\\\\n") 
            self.write("\\hline{}\n")
            for signal in [x for x in signals if x.group==group]:
                config = dw.get_config(signal)
                if signal.name+'/status' in config:
                    if config[signal.name+'/status']=='ok':
                        self.write("\\ok")
                    elif config[signal.name+'/status']=='notok':
                        self.write("\\notok")
                    elif config[signal.name+'/status']=='warn':
                        self.write("{\bf !}")
                    self.write("&"+signal.name.replace('_',' ')+"&")
                        
                if signal.name+'/comment' in config:
                    self.write(config[signal.name+'/status'])
                self.write("\\\\\n")
        self.write("\\caption{\\ok : Sensor approved, \\notok : Sensor not accepted, {\\bf !} sensor approved for continued use, but under scrutiny }\n")  
        self.write("\\end{longtable}\n")
        
    def add_sensor_properties_report(self, signals):
        """ Function that adds a sensor properties table to the report based on the config file (obtained through the signal attributes)
        :param signals: list of signals for which the table will be included
        :type signals: list of :class:`dynawind.dynawind.Signal`
        """
        groups = set([x.group for x in signals])
        self.write("\\begin{longtable}{l|cc|c|cc}\n")
        self.write("\\centering\n")
        self.write("\\bf{Sensor}&\\bf{Level}&\\bf{Heading}&\\bf{Offset}&\\bf{$R_i$ (m)}&\\bf{$R_o$ (m)}\\\\\n")  
        self.write("\\hline{}\n")
        self.write("\\endhead % all the lines above this will be repeated on every page\n")
        for group in groups:
            self.write("&&\\\\\n") 
            self.write("&\\bf{"+group.title()+"}&\\\\\n") 
            self.write("\\hline{}\n")
            for signal in [x for x in signals if x.group==group]:
                config = dw.get_config(signal)
                self.write(signal.name.replace('_',' ')+"&")

                self.write(str(signal.level)+"&")                
                self.write(str(signal.heading)+"&")
                if hasattr(signal, 'offset'):
                    self.write(str(signal.offset))
                self.write("&")
                if hasattr(signal, 'Ri'):
                    self.write(str(signal.Ri))
                self.write("&")
                if hasattr(signal, 'Ro'):
                    self.write(str(signal.Ro))
                    
                self.write("\\\\\n")
        self.write("\\caption{Overview of sensor configuration}")
        self.write("\\end{longtable}\n")

    def add_sensor_layout(self,signals, include_schematic=True, struct_type='monopile', tower_top=80,  tp_height=19):
        """ function that adds the sensor lay-out to the report based on the configuration of the signals
        
        :param signals: list of signals
        :param include_schematic: indicate whether or not the turbine overview needs to be included
        :type include_schematic: bool.
        :param struct_type: 'monopile' or 'jacket'
        :param tower_top: tower top level (LAT), together used with tp_height to draw the turbine overview
        :param tp_height: TP top (LAT)
        """

        lvl_list = [x.level for x in signals]
        lvls=sorted(list(set(lvl_list)))
        group_list=[x.group for x in signals]
        groups=sorted(list(set(group_list)))

        
        self.write('\\begin{figure}\n')
        self.write('\\centering\n')
        
        if include_schematic:
            self.write('\\TowerTopHeight{'+str(tower_top)+'}\n')
            self.write('\\TpHeight{'+str(tp_height)+'}\n')
            
            self.write('\\CurrentSensorLevel{'+','.join([str(x) for x in lvls])+'} % in m above LAT\n')
            if struct_type == 'monopile':
                self.write('\\subfigure[Turbine schematic with sensor levels]{\\DrawMonopileOWT}\n')
            elif struct_type == 'jacket':
                self.write('\\subfigure[Turbine schematic with sensor levels]{\\DrawJacketOWT}\n')
        
        caption_str=''

        for lvl in lvls:
            for group in groups:
                lvl_signals=[x for x in signals if x.level==lvl and x.group==group]
                if group == 'fiber':       
                    heads =[str(x.heading) for x in lvl_signals if '_FBG_' in x.name]
                    self.write('\\TypeTwoSensorPositions{')
                    self.write(','.join(heads))
                    self.write('}\n')
                    heads =[str(x.heading) for x in lvl_signals if '_TFBG_' in x.name]
                    self.write('\\TFBGSensorPositions{')
                    self.write(','.join(heads))
                    self.write('}\n')
                    if 'fiber strain' not in caption_str:
                        caption_str += '$$ optical fiber strain sensors, '
                elif group == 'acceleration':
                    for signal in lvl_signals:
                        if '_X' in signal.name:
                            self.write('\\RadialSensorPositions{'+str(signal.heading)+'}% Positive for outward pointing, Negative for inward pointing\n')
                        if '_Y' in signal.name:
                            self.write('\\TangentialSensorPositions{'+str(signal.heading)+'}% Positive for counterclockwise, negative for clockwise\n')
                    if 'accelerometers' not in caption_str:
                        caption_str += '($\\,\\to\\,$ ) position and orientation of the accelerometers, '
                elif group == 'lvdt':
                    heads =[str(x.heading) for x in lvl_signals]
                    self.write('\\TypeThreeSensorPositions{')
                    self.write(','.join(heads))
                    self.write('}\n')
                    if 'lvdt' not in caption_str:
                        caption_str += '(\\textcolor{orange}{+}) position of the lvdt displacement sensors, '
                elif group == 'temp':
                    heads =[str(x.heading) for x in lvl_signals]
                    self.write('\\TypeOneSensorPositions{')
                    self.write(','.join(heads))
                    self.write('}\n')
                    if 'temperature' not in caption_str:
                        caption_str += '($\\circ$) position of the temperature sensors, '
            self.write('\\subfigure['+str(lvl)+'m]{\\DrawTopSensorOverview}\n')
        self.write('\\caption{ Overview of the sensors installed on the turbine, (a) installation levels, (b) sensors installed at different levels with '+caption_str[:-2]+'}\n')
        self.write('\\end{figure}\n')
    
    def add_list_of_figures(self):
        self.write('\\listoffigures{}\n')
        self.write('\\clearpage{}\n')
    
    def compile_report(
        self,
        template_folder=r"\\192.168.119.12\Templates\Documents\24SEA\LaTeX_Report_Template",
        remove_aux=False,
        remove_figures=False,
    ):
        """ Compiles the report into a pdf document
        
        :param template_folder: root of the reporting templates
        :param remove_aux: remove aux files generated during compiling
        :type remove_aux: bool.
        :param remove_figures: remove figures after compiling is complete
        :type remove_figures: bool.
        
        """
        from shutil import copy2, rmtree
        from subprocess import call

        if not os.path.isdir(template_folder):
            raise ImportError("Template file not found/reachable")

        copy2(os.path.join(template_folder, self.template + ".cls"), self.path)
        if '24SEA' in self.template:
            copy2(os.path.join(template_folder, "24SEA_logo.png"), self.path)
        elif 'OWI' in self.template:
            copy2(os.path.join(template_folder, "OWI_logo.png"), self.path)

        self.export_tex()
        print("Compiling ...")
        call("pdflatex " + self.filename + ".tex", cwd=os.path.realpath(self.path))
        # Because pdflatex requires two runs to render references
        print("Rendering references ...")
        call("pdflatex " + self.filename + ".tex", cwd=os.path.realpath(self.path))
        if remove_aux:
            # Remove template files
            os.remove(os.path.join(self.path, self.template + ".cls"))
            os.remove(os.path.join(self.path, "24SEA_logo.png"))
            # Remove aux files generated by LaTeX
            auxs = [".aux", ".log", ".tex", ".out"]
            for aux in auxs:
                os.remove(os.path.join(self.path, self.filename + aux))

        if remove_figures:
            # Remove the figures Folder
            if os.path.exists(os.path.join(self.path, "Figures")):
                rmtree(os.path.join(self.path, "Figures"), ignore_errors=True)

    def export_tex(self):
        """ Finalizes report and exports .tex file"""
        self.write("\\end{document}\n")
        with open(os.path.join(self.path, self.filename + ".tex"), "w") as f:
            for line in self.body:
                f.write(line)
    def float_barrier(self):
        """ Add Float barrier to the report """
        self.write('\\FloatBarrier{}\n')
    def write(self, string):
        """ Add custom LaTeX string to report """
        self.body.extend(string)


class Project:
    """ DYNAwind project class that includes all information on the project, based on the project code the supplier will be determined.
    
    :param code: Project code as documented else
    :type code: str. or int.
    :param client: Project client
    :type client: str.
    :param name: Project name
    :
    """
    def __init__(self, code, client=None, name=None):
        if isinstance(code, int):
            code = str(code).zfill(4)
        self.code = code
        self.client = client
        self.name = name
        self.supplier = self.get_supplier()

    def get_supplier(self):
        if not self.code[0] == "0":
            supplier = "24SEA"
        else:
            if int(self.code) > 100:
                supplier = "24SEA"
            else:
                supplier = "OWI"
        return supplier

    def supplier_reference(self):
        if self.supplier == "24SEA":
            reference = "24SEA"
        else:
            reference = "OWI"
        return reference

    def __str__(self):
        return " ".join([self.code + ":", self.name, "(" + self.client + ")"])


class DVreport(Report):
    """ Report class specifically for design verification reports """
    def __init__(self, path, project, author="Wout Weijtjens", document_nr=1, title=None, location='nA', document_type='DV', add_header=False):
        Report.__init__(
            self,
            path,
            project,
            title=title,
            author=author,
            location=location,
            document_type=document_type,
            document_nr=document_nr,
            template="24SEA_Report_Template",
            add_header=add_header
        )

    def add_eoc(self, location, timestamp, yaw, tidal=None, heading=None, t0=0, te=600, rpm=None, windspeed=None, rms1p=None, technician=None, comment=None):
        """ Adds basic environmental and operational data to Design verification report
        
        :param location: measurement location
        :param timestamp: measurement timestamp
        :param yaw: yaw angle / wind direction of the turbine during the time of the measurements
        :param tidal: tidal level at the time of the measurement
        :param heading: heading of the sensors at the time of the measurement
        :param t0: considered start of the measurement
        :param te: considered end of the measurement
        :param rpm: rotational speed of the turbine at the time of the measurement
        :param technician: Person involved with performing the measurements
        :param comment: Any additional comment to the report
        """
        
        self.write("\\RadialSensorPositions{"+str(heading)+"}\n")
        self.write("\\TangentialSensorPositions{"+str(heading)+"}\n")
        self.write("\\ReferenceFeature{"+str(yaw)+"}\n") 
        self.write("\\begin{tabular}{cc|c}\n")
        self.write("\\raisebox{-2cm}{\\DrawTopSensorOverview}&\\begin{tabularx}{0.78\\textwidth}{l|X}\n")
        self.write("Turbine &{\\bf "+ location+ "}\\\\\n")
        self.write("Start of measurements &"+timestamp.strftime("%H:%M:%S")+"\\\\\n")
        self.write("$t_0 (s)$ &"+str(round(t0,2))+"\\\\\n")
        self.write("$t_e (s)$ &"+str(round(te,2))+"\\\\\n")

        self.write("Yaw angle (deg.) &"+ str(yaw) +"\\\\\n")
        if tidal is not None:
            self.write("Tidal level (cm) &"+ str(tidal) +"\\\\\n")
        if rpm is not None:
            self.write("RPM &" + str(rpm) + "\\\\\n")
            
        if windspeed is not None:
            self.write("Wind speed (m/s) &" + str(rpm) + "\\\\\n")
        if rms1p is not None:
            self.write("RMS1P (FA) &" + str(rms1p[0]) + "\\\\\n")
            self.write("RMS1P (SS) &" + str(rms1p[1]) + "\\\\\n")
        if technician is not None:
            self.write("Measurements conducted by &" + technician + "\\\\\n")
        self.write("&\\\\\n") #empty line as spacing
        if comment:
            self.write("{\\bf Comment:}&"+comment+"\\\\\n")
        else:
            self.write("{\\bf Comment:}& None\\\\\n")
 
        self.write("\\end{tabularx}\n")
        self.write("\\end{tabular}\n")
        self.write("\\FloatBarrier\n")

    
    def add_table(self, mpe_FA, mpe_SS):
        """ Adds a table stylelized specifically for Design verification reports """
        
        nrOfRows = max(len(mpe_FA.freq_median), len(mpe_SS.freq_median))
        self.write("{\\centering\n")
        self.write("\\begin{tabular}{l|cc||cc}\n")
        self.write("&\\multicolumn{2}{c||}{FA}&\\multicolumn{2}{c}{SS}\\\\\n")
        self.write("& Freq. (Hz) & Std. Freq (Hz) & Freq. (Hz) & Std. Freq (Hz) \\\\\n")
        self.write("\\hline{}\n")
        def write_freq_str(mpe,x):
            tr_freq_median = [x['freq_median'] for x in mpe.tracked]
            if mpe.freq_median[x] in tr_freq_median:
                tbl_str = "{\\bf "+str(round(mpe.freq_median[x], 4)) + "}&{\\bf " + str(round(mpe.freq_std[x], 5))+"}"
            else:
                tbl_str = str(round(mpe.freq_median[x], 4)) + "&" + str(round(mpe.freq_std[x], 5))
            return tbl_str

        for x in range(0, nrOfRows):
            if x > len(mpe_SS.freq_median) - 1:
                self.write(
                    "&"
                    + write_freq_str(mpe_FA,x)
                    + "&&\\\\\n"
                )
            elif x > len(mpe_FA.freq_median) - 1:
                self.write(
                    "&&&"
                    + write_freq_str(mpe_SS,x)
                    + "\\\\\n"
                )
            else:
                self.write(
                    "&"
                    + write_freq_str(mpe_FA,x)
                    + "&"
                    + write_freq_str(mpe_SS,x)
                    + "\\\\\n"
                )
        self.write("\\end{tabular}\\par}")

    def new_meas(self, location, timestamp):
        """ creates a new section based on a new measurement
        
        :param location: measurement location
        :param timestamp: measurement timestamp
        """
        self.write('\\clearpage\n')
        self.section(location+': '+ timestamp.strftime('%d/%m/%Y %H:%M'))
        
class Table(object):
    """ Table class object to add in reports
    
    :param data: data to add into the table, either a list of dicts with keys 'string' and 'value', a pandas dataframe, or a list of dicts with 'row' and other keys for columns
    :param no_columns: number of columns in the table
    :type no_columns: int.
    :param column_layout: typical latex formatting of columns e.g. l|cc|c (required for style columns)
    :type column_layout: str.
    """

    def __init__(self, data = None, no_columns=4, style='Stats', header=None, column_layout=None, first_column = False, first_row = False, long_table = False, round_val = None, caption = None):
        # data passed to the Table object is a list of dicts
        self.data = data
        self.caption_str = caption
        self.no_columns = no_columns
        self.style = style
        self.header = header
        self.column_layout = column_layout
        self.long_table = long_table
        self.round_val = round_val
        if long_table:
            self.body = ["\\begin{longtable}{" + self.column_layout + "}\n"]
        else:
            self.body = ["\\begin{tabular}{" + self.column_layout + "}\n"]
        self.first_row = first_row
        self.first_column = first_column
    def caption(self, caption):
        """Add caption to the table"""
        self.caption_str = caption
    def add_empty_row(self):
        """ Adds an empty row to the Table"""
        self.add_row('')
    def add_row(self,row):
        """ Adds a row to the table 
        
        :param row: Elements in the row
        :type row: list
        """
        
        if type(row) is str:
            # If a string is passed, the first cell is the string, rest is empty
            row = [row]
            for i in range(self.no_columns-1):
                row.append('')
        
        round_val = self.round_val
        
        if self.first_column or self.first_row:
            self.body.extend('{\\bf '+str(row[0])+'}')
        else:
            self.body.extend(str(row[0]))
        for cell in row[1:]:
            if round_val is not None:
                fmt = '{0:.'+str(round_val)+'f}'
                if type(cell) is not str:
                    cell = fmt.format(cell)
            if self.first_row:
                self.body.extend('&{\\bf '+str(cell)+'}')
            else:
                self.body.extend('&'+str(cell))
        if self.first_row:
            # At end of first row set first_row to False
            self.first_row = False
        self.end_of_row()
    def hline(self):
        """ Adds a horizontal line to the table """
        self.body.extend("\\hline{}\n")
    
    def end(self):
        """ Ends the entire table """
        
        if self.long_table:
            self.body.extend("\\end{longtable}\n")
        else:
            self.body.extend("\\end{tabular}\n")
    def end_of_row(self):
        self.body.extend('\\\\\n')
    def write(self, style=None):
        # write the table depending the style
        if style is None:
            style = self.style
        strings = []
        if style == "Stats":
            table_columns = "lc"
            for i in range(1, int(self.no_columns / 2)):
                table_columns += "|lc"
            strings.append("\\begin{tabular}{" + table_columns + "}\n")
            col_ind = 2
            for item in self.data:
                string = "\\bf{" + item["string"] + ":}& "

                if isinstance(item["value"], str):
                    string = string + item["value"]
                else:
                    string = string + "{:.2f}".format(item["value"])

                if col_ind == self.no_columns:
                    string = "&" + string + "\\\\\n"
                    col_ind = 0
                col_ind += 2
                strings.append(string)
            strings.append("\\end{tabular}\n")
        elif style == "df":
            table_columns = "l|c"
            for i in range(1, int(self.no_columns)):
                table_columns += "c"
            strings.append("\\begin{longtable}{" + table_columns + "}\n")
            string = "\\bf{Timestamp} "
            for head in self.header:
                string = string + "& "+head
            strings.append(string+ "\\\\\n")
            strings.append("\\hline{}\n")
            strings.append("\\endhead % all the lines above this will be repeated on every page\n")

            for index, row in self.data.iterrows():
                string =  index.strftime('%d-%m-%Y %H:%M:%S')
                for key in row.keys():
                    string = string + "& "+ "{:.2f}".format(row[key])
                strings.append(string+ "\\\\\n")
            strings.append("\\end{longtable}\n")
        elif style == "columns":
            # This style takes the dict keys as the header of columns, dict should include a key row
            
            strings.append("\\begin{tabular}{" + self.column_layout + "}\n")
            string = ""

            for key in self.data[0].keys():
                if not key == 'row':
                    string = string + "&{\\bf "+ key + "}"
            strings.append(string+ "\\\\\n")
            strings.append("\\hline{}\n")
            for item in self.data:
                string = "{\\bf "+ item['row'] +"}"
                for key in item.keys():
                    if not key == 'row':
                        if isinstance(item[key],str):
                            string = string + "&" + item[key]
                        else:
                            string = string + "&" + "{:.2f}".format(item[key])
                strings.append(string+ "\\\\\n")
            strings.append("\\end{tabular}\n")
        else:
            self.end()
            strings = self.body
            
            
            
        return strings

def unit_string(string):
    """ Default formating for unit """
    # This should better be captured by the LaTeX template.
    string = '('+it(string)+')'
    return string
def it(string):
    """ Make string in italics for LaTeX"""
    string = '{\\it '+string + '}'
    return string
def highlight(string):
    
    string = '\\tableHighlight{'+string+'}'
    return string
class Figure(object):
    """ Figure class object to add in report
    
    :param filename: Filename under which the figure will later be stored. When None defaults to structured filename ``DYNAwind_fig_00x``
    :type filename: str.
    :param dpi:dots per inch
    :type dpi: int.
    :param figsize:
    :type figsize: tuple
    :param caption: Caption to add to the figure
    :type caption: str.
    """

    newid = itertools.count()

    def __init__(self, filename=None, dpi=120, figsize=(20, 4), caption=None):
        self.caption_str = caption
        self.dpi = dpi
        self.fig_handle = plt.figure(figsize=figsize, dpi=dpi)
        if filename is None:
            self.filename = "DYNAwind_fig_" + str(next(Figure.newid)).zfill(3)
        else:
            self.filename = filename

    def caption(self, caption):
        """Add caption to figure"""
        self.caption_str = caption

    def close(self):
        plt.close(self.fig_handle)

    def save(self, path, borders='tight'):
        """ Saves figure to the report specified path. By default the borders are set to tight. However, when
        a fixed set of borders is preferred the borders setting can also be set to 'fixed'
        
        :param path: Final path of the saved figure.
        :param borders: 'fixed' or 'tight'
        :type borders: string
        """
        if borders=='tight':
            self.fig_handle.savefig(path, dpi=self.dpi, bbox_inches="tight")
        elif borders=='fixed':
            self.fig_handle.savefig(path, dpi=self.dpi)

def longterm_plot(df,   parameters, filename, ylabel, ylim =None, include_zoom=False, labels=None, alpha = 1, hlines=None):
    """ Make a long term plot for inclusion in a report 
    
    :param df: Data frame from a query to the database, serves as basis for the plot
    :type df: pd.DataFrame
    :param parameters: parameters to be considered for the plot
    :param filename: Filename to store the figure under
    :param ylabel: Graphs ylabel
    :param ylim: Fixed y-limits to use
    :param include_zoom: Include the zoom on the past month
    :param labels: labels to use for the legend
    :type labels: list
    :param alpha: alpha (transparancy) of the plot
    :param hlines: Horizontal lines to be added to the plot
    :type hlines: list
    """
    
    from dynawind.db import make_dt_list
    import matplotlib.dates as mdates
    from pytz import utc

    dt_start = df.index[0].astimezone(utc)
    dt_stop = df.index[-1].astimezone(utc)
    
    if labels is None:
        labels = parameters
    
    figs = []
    fig1=Figure(filename=filename, caption = 'Long term', figsize=(20,6))
    for prm,lbl in zip(parameters,labels):
        plt.plot(df[prm], label=lbl, alpha = alpha)
    plt.ylabel(ylabel)
    plt.legend()
    if not ylim is None:
        plt.ylim(ylim)
    
    plt.grid(which='both',axis='y', linestyle='dotted')
    if not hlines is None:
        for hline in hlines:
            plt.axhline(hline,color='red',linestyle='dashed')
    plt.xlim(dt_start,dt_stop)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b \'%y"))

    figs.append(fig1)
    if include_zoom:
        fig2=Figure(filename=filename+ '_zoom', caption = 'Zoom on '+dt_stop.strftime('%B %Y'), figsize=(20,6))
        for prm,lbl in zip(parameters,labels):
                plt.plot(df[prm],'-', label=lbl, alpha = alpha)
        plt.ylabel(ylabel)
        plt.legend()
        if not ylim is None:
            plt.ylim(ylim)
        else:
            y_min = df[prm][dt_stop.replace(day=1, hour=0, minute=0, second=0, microsecond=0):dt_stop].min()*0.9
            y_max = df[prm][dt_stop.replace(day=1, hour=0, minute=0):dt_stop].max()*1.1
            plt.ylim([y_min, y_max])
        if not hlines is None:
            for hline in hlines:
                plt.axhline(hline,color='red',linestyle='dashed') 
        plt.grid(which='both',axis='both', linestyle='dotted')
        plt.xlim(dt_stop.replace(day=1, hour=0, minute=0, second=0, microsecond=0).replace(day=1),dt_stop)
        ax = plt.gca()
        dt_list = make_dt_list(dt_stop.replace(day=1, hour=0, minute=0, second=0, microsecond=0),dt_stop, interval =600*144)  
        ax.set_xticks(dt_list)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

        figs.append(fig2)
    
    return figs

class LT_Export_Report(Report):
    """ An LT Export Report is a template report for sharing data from the Postgres database """
    
    def __init__(self, project, dt_start, dt_end, location= "nA",  author="Wout Weijtjens", document_nr=1, template = "24SEA_Report_Template", path='.', client_reference = '', version = 1):
        self.document_type = 'DD'
        self.author = author
        self.document_nr = document_nr
        self.dt_start = dt_start 
        self.dt_end = dt_end
        self.version = version
        self.template = template
        self.location = location
        self.path = path
        self.body = []
        self.description = 'Data description for data shared from ' + dt_start.strftime('%Y/%m/%d %H:%M') + ' to ' + dt_stop.strftime('%Y/%m/%d %H:%M')
        self.project = project
        self.client_reference = client_reference
        self.title = self.make_title()
        self.filename = self.title
        self.titlepage = True
        self.add_header()
        
        self.dt_start = dt_start
        self.dt_end = dt_end
        
        self.section('Introduction')
        self.write('This documents provides an overview of the data shared to ' + self.project.client +'.')
        self.write('All data shared ranges from ' + dt_start.strftime('%Y/%m/%d %H:%M') + ' to ' + dt_stop.strftime('%Y/%m/%d %H:%M')+'.')
        self.write('The data is shared in Comma seperated value (.csv) files per data type.')
        
        
    def data_section(self, data_type, database=None, sensors = None, result=None, stats=['mean'], add_figure=True, stat_4_plot='mean'):
        self.section(data_type)
        
        self.write('File shared :\n')
        self.write('{\\it '+self.export_file_name(data_type).replace('_','\\_')+'}')
        table_data=[]
        stat_list = ['min','max','mean','rms']
        for sensor in sensors:
            table_row = dict()
            table_row['row'] = sensor['sensor'].replace('_',' ')
            for stat in stat_list:
                table_row[stat]=''
                if stat in stats:
                    table_row[stat]='$\\bullet{}$'
            table_row['Unit']= "$"+sensor['unit']+"$"
            table_row['Descripion'] = sensor['description']
            table_data.append(table_row)
        
        Tab = Table(style='columns', column_layout='l|cccc|c|l', data=table_data, no_columns=7)
        
        self.add_table(Tab, location='h!')
        
        self.write('\\FloatBarrier{}\n')
        
        parameters = []
        for sensor in sensors:
            for stat in stats:
                parameters.append('_'.join([stat,sensor['sensor']]))
        
        if database is not None:
            result = database.query(dt=(self.dt_start,self.dt_end), as_dataframe=True, parameters=parameters)
    
        if add_figure:
            Fig1 = Figure(caption = 'Shared ('+str(stat_4_plot)+') data of ' + data_type.lower() + ' sensors.', filename='LT_'+data_type.lower())
            for sensor in sensors:
                if '_'.join([stat_4_plot,sensor['sensor']]) in result: 
                    plt.plot(result['_'.join([stat_4_plot,sensor['sensor']])], label = sensor['sensor'])
            
            plt.grid(which='both', axis='both')
            plt.legend()
            plt.xlim([self.dt_start,self.dt_end])
            self.add_figure(Fig1)
            
            Fig1.close()
        
        self.export_2_csv(result,self.export_file_name(data_type))
        
        
    def export_2_csv(self, df, file_name):
        df.to_csv(os.path.join(self.path,file_name))
        pass
        
    def export_file_name(self, data_type, extension ='.csv'):
        file_name = '_'.join([str(self.project.code).zfill(4),'24SEA',self.location.upper(),data_type.lower(),self.dt_start.strftime('%Y%m%d'), self.dt_end.strftime('%Y%m%d')])
        file_name += extension
        return file_name
