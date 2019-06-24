"""
dynawind.mdl
===================


"""
import json
import os
import dynawind.db as db
import numpy as np


class Model(object):
    """ dynawind model for regression analysis
    
    :param site: 
    :type site: str.
    :param location:
    :type location:
    :param output: output parameter of the model
    :type output:
    """
    def __init__(self, site, location, output, model_file=None):
        self.location = location
        self.site = site
        self.output = output
        self.importModel(model_file)

    def importModel(self, model_file):
        """ This function will import the model from the json files in the folder models"""
        import pkg_resources
        # Load the model 
        site = self.site
        mdl = None

        resource_package = __name__
        resource_path = "/".join(["config", site.lower(), 'models', '_'.join([self.location, self.output, "mdl.json"])])
        mdl_path = pkg_resources.resource_filename(resource_package, resource_path)
        
        configPath = ".\\config\\" + site.lower() + "\\models\\"
        if os.path.isfile(mdl_path):
            f = open(mdl_path,'r')
            mdls = json.load(f)
            f.close()
            mdl = dict()
            for md in mdls:
                newkey = {md["case"]: md}
                mdl.update(newkey)
        else:
            print('Model not found')
        self.mdl = mdl

    def export2SQL(self, case, table, commit=False):
        """ Translates the model to a sql expression to be used on the postgres database
        :param case: the case of which the model is to be exported
        :type case: str.
        :param table: database table to which the function applies
        :type table: str.
        :returns: sql expression
        :rtype: str.
        """
        sql = "CREATE OR REPLACE FUNCTION "
        sql = sql + 'regr_'+self.mdl[case]['output_parameter']+'_'+case +'('+table+')'
        sql = sql + " RETURNS double precision as $$"
        sql = sql + """SELECT cast($1.metrics->>'""" + self.mdl[case]['output_parameter'] + """' as double precision) -1*("""
        model_str = self.mdl[case]['model']
        for coef, value in self.mdl[case]['coeficients'][0].items():
            model_str=model_str.replace('$'+coef,str(value))
        for prm, value in self.mdl[case]['parameters'][0].items():
            sql_str = """cast($1.metrics->>'"""+value+"""' as double precision)"""
            model_str=model_str.replace('$'+prm,sql_str)

        sql = sql + model_str 
        sql = sql+ """) as result"""
        sql = sql + """ WHERE $1.metrics ->>'""" +self.mdl[case]['case_parameter'] +"""'='"""+case+"""'"""
        sql = sql + "$$ LANGUAGE SQL;"
        
        if commit:
            # commit the expression to the database
            db_conn = db.Postgres(self.mdl[case]['site'])
            db_conn.execute(sql)
            db_conn.commit()
            db_conn.close()            
        
        return sql
    
#    def evalModel(self, dt, root=".", fileext=".json"):
#        from db import pullValuesfromJSON
#
#        # %%
#        data = pullValuesfromJSON(
#            self.site, self.location, dt, root=root, fileext=fileext
#        )
#
#        output = dict()
#        output["residual"] = np.nan
#        output["prediction"] = np.nan
#        output["likelihood"] = np.nan
#        output["std"] = np.nan
#        # %% Select from mdl the mdl associated with the case at dt
#        if "all" in self.mdl.keys():  # Universal model for all cases
#            case = "all"
#        else:
#            if "case" in data:
#                case = data["case"]
#            else:
#                return output
#
#        if case in self.mdl:
#            mdl = self.mdl[case]
#            if "std" in mdl:
#                output["std"] = mdl["std"]
#            else:
#                output["std"] = 0
#
#            # %%
#            output["prediction"] = evaluateModel(
#                mdl["model"], data, mdl["parameters"], mdl["theta"]
#            )
#            # %%
#            if self.output in data:
#                output["residual"] = data[self.output] - output["prediction"]
#            elif "median_" + self.location + "_" + self.output + "_FREQ" in data:
#                output["residual"] = (
#                    data["median_" + self.location + "_" + self.output + "_FREQ"]
#                    - output["prediction"]
#                )
#
#            # %% Calculate likelihood
#
#            output["likelihood"] = (
#                output["residual"] * ((1 / output["std"]) ** 2) * output["residual"]
#            )
#            # %%
#        return output

#    def export2dict(self, output="prediction"):
#
#        if output == "prediction":
#            keystr = "MDL_PRE_"
#        elif output == "Residual":
#            keystr = "MDL_RES_"
#        elif output == "Likelihood":
#            keystr = "MDL_LH_"
#
#        export = {keystr + self.output: result}
#        return export

    def __repr__(self):
        repr_str = (
            "DYNAwind model object\n"
            + "---------------------\n"
            + "Parameter: "
            + self.output
        )
        return repr_str


#def evaluateModel(modeltype, data, parameters, theta):
#    """ Evaluates the model for the data provided """
#    if modeltype == "polynomial":
#        if len(theta) == 1:
#            result = theta
#        else:
#            # theta[0]*x**(N-1) + theta[1]*x**(N-2) + ... + theta[N-2]*x + theta[N-1]
#            result = np.polyval(theta, data[parameters[0]])
#    return result[0]


# %% Class definitions
#def readClassDefinitions(site):
#    import configparser
#
#    config = configparser.ConfigParser()
#    config.read("./config/" + site + "/" + site + "_casedefinitions.ini")
#    caseDefinitions = []
#    for section in config:
#        if section is not "DEFAULT":
#            caseDef = dict(config[section])
#            for key in caseDef.keys():
#                caseDef[key] = np.float64(np.array(caseDef[key].split(sep=",")))
#            caseDef["case"] = section
#            caseDefinitions.append(caseDef)
#    return caseDefinitions


#def caseClassifier(data, site, caseDefinitions):
#    for case in caseDefinitions:
#        isCase = True
#        for key in case.keys():
#            if key not in data:
#                return "no SCADA"
#            if key is "case":
#                continue
#            if data[key] >= case[key][0] and data[key] < case[key][1]:
#                inCase = True
#            else:
#                inCase = False
#            isCase = isCase and inCase
#        if isCase:
#            return case["case"]
#
#    else:
#        return "caseless"
#
#
#def getCaseforTimestamp(dt):
#
#    return case
