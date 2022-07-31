__metaclass__ = type


class Filter_bb:
    def __init__(self, struc=None, config=None, save=None, debug=None):
        self.myInfo = "Filter bb"
        self.struc = str(struc)
        self.save = str(save)
        self.config = str(config)
        self.debug = debug
        self.bbNum = -1
        self.bbList = []

        self.debug.line(4, " ")
        self.debug.line(4, "#")
        self.debug.debug("# " + self.myInfo, 1)
        self.debug.line(4, "#")

        self.reset_config_data()

    def reset(self, struc=None, config=None, save=None, debug=None):
        self.debug.debug("* " + self.myInfo + " -- reset", 5)
        if struc:
            self.struc = str(struc)
        if save:
            self.save = str(save)
        if config:
            self.config = str(config)
        if debug:
            self.debug = debug
        self.bbNum = -1
        self.bbList = []

        self.reset_config_data()

    def reset_config_data(self):
        self.debug.debug("* " + self.myInfo + " -- reset_config_data", 5)
        self.config_start = ';'
        self.config_sep = '\\n'
        self.config_equal = ': '
        self.config_data = []
        # self.config_data.append({'name_config':'date','name':'StartTime','value':''})

    def read_bb_struc(self):
        self.debug.debug("* " + self.myInfo + " -- read_bb_struc", 5)
        return

    def input_check(self, bbInfo):
        self.debug.debug("* " + self.myInfo + " -- input_check", 5)
        return

    def get_bb_num(self):
        self.debug.debug("* " + self.myInfo + " -- get_bb_num", 6)
        return self.bbNum

    def get_bb_data(self):
        self.debug.debug("* " + self.myInfo + " -- get_bb_data", 5)
        return self.bbList

    def output_bb_data(self):
        self.debug.debug("* " + self.myInfo + " -- output_bb_data", 5)
        if not self.save:
            print("Save file not set!")
            return
        return

    def output_bb_config(self):
        self.debug.debug("* " + self.myInfo + " -- output_bb_config", 5)
        if not self.config:
            print("Config file not set!")
            return
        return
