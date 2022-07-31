import re
import Filter.Filter_bb as filter_bb

__metaclass__ = type


class Filter_bb_SWF(filter_bb.Filter_bb):
    def reset_config_data(self):
        self.config_start = ';'
        self.config_sep = '\\n'
        self.config_equal = ': '
        self.config_data = []
        self.config_data.append({'name_config': 'Maxbb', 'name': 'Maxbb', 'value': ''})


    def read_bb_struc(self):
        nr_sign = ';'  # Not read sign. Mark the line not the job data
        sep_sign = ' '  # The sign seperate data in a line
        sep_sign2 = ':'  # The sign seperate data in a line
        nameList = []
        nameList.append(["Maxbb", "bb"])

        regex_rest = " *:([^\\n]+)\\n"
        regexList = []
        bb_info = {}

        for dataName in nameList:
            regexList.append([(dataName[0] + regex_rest), dataName[1]])
        # [['Maxbbs *:([^\\n]+)\\n', 'bb'], ['Maxbbs *:([^\\n]+)\\n', 'bb']]  regexList boyang

        bbFile = open(self.struc, 'r')  # test.swf boyang
        while (1):
            tempStr = bbFile.readline()
            if not tempStr:  # break when no more line
                break
            if tempStr[0] == nr_sign:  # The information line
                for dataRegex in regexList:
                    # logic is dateRegex是正则表达，看temptr是不是能match上
                    matchResult = re.findall(dataRegex[0], tempStr)
                    if (matchResult):
                        bb_info[dataRegex[1]] = int(matchResult[0].strip())
                        break

                for con_data in self.config_data:
                    con_ex = con_data['name'] + self.config_equal + "([^" + self.config_sep + "]*)" + self.config_sep

                    temp_con_List = re.findall(con_ex, tempStr)
                    if (len(temp_con_List) >= 1):
                        con_data['value'] = temp_con_List[0]
                        break
            else:
                break

        bbFile.close()
        print ("This is bb info {},",bb_info)
        self.bb_data_build(bb_info)  # print This is bb info,{} {'bb': 100, 'bb': 100} Boyang
        self.bbNum = len(self.bbList)

    def bb_data_build(self, bb_info):
        bb_num = bb_info['bb']
        self.bbList = []
        i = 0
        while (i < bb_num):
            self.bbList.append({"id": i + 1, \
                                  "location": [1], \
                                  "group": 1, \
                                  "state": -1, \
                                  "bb": 1, \
                                  "start": -1, \
                                  "end": -1, \
                                  "extend": None})
            i += 1
        return 1

    def output_bb_data(self):
        if not self.save:
            print("Save file not set!")
            return
        print ("This is self.save,{}",self.save)
        sep_sign = ";"
        f2 = open(self.save, "w")
        for bbResult_o in self.bbList:
            f2.write(str(bbResult_o['id']))
            f2.write(sep_sign)
            f2.write(str(bbResult_o['location']))
            f2.write(sep_sign)
            f2.write(str(bbResult_o['group']))
            f2.write(sep_sign)
            f2.write(str(bbResult_o['state']))
            f2.write(sep_sign)
            f2.write(str(bbResult_o['bb']))
            f2.write("\n")
        f2.close()

    def output_bb_config(self):
        if not self.config:
            print("Config file not set!")
            return

        format_equal = '='
        f2 = open(self.config, "w")

        for con_data in self.config_data:
            f2.write(str(con_data['name_config']))
            f2.write(format_equal)
            f2.write(str(con_data['value']))
            f2.write('\n')
        f2.close()