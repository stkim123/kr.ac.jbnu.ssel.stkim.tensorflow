import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import scale
import xlrd

class Visualizer:
    def __init__(self, fileName):
        workbook = xlrd.open_workbook(fileName)
        self.sheet = workbook.sheet_by_index(0)

        self.rows = self.sheet.nrows
        self.cols = self.sheet.ncols

        self.data = [[self.sheet.cell_value(r, c) for c in range(self.cols)] for r in range(self.rows)]
        type(self.data)

        self.stationNum = []
        self.stationName = []
        self.income_date = []
        self.stationData = pd.DataFrame(self.data,
                                        columns = ['station', 'stat_name', 'income_date', 'on_tot', 'on_05', 'on_06', 'on_07', 'on_08', 'on_09', 'on_10', 'on_11', 'on_12', 'on_13', 'on_14', 'on_15', 'on_16', 'on_17', 'on_18', 'on_19', 'on_20', 'on_21', 'on_22', 'on_23', 'on_24', 'off_tot', 'off_05', 'off_06', 'off_07', 'off_08', 'off_09', 'off_10', 'off_11', 'off_12', 'off_13', 'off_14', 'off_15', 'off_16', 'off_17', 'off_18', 'off_19', 'off_20', 'off_21', 'off_22', 'off_23', 'off_24'])

        for i in range(1, self.rows):
            self.stationNum.append(self.sheet.cell_value(i, 0))

        for i in range(1, self.rows):
            self.stationName.append(self.sheet.cell_value(i, 1))

        for i in range(1, self.rows):
            self.income_date.append(self.sheet.cell_value(i, 2))

        # for i in range(1, self.rows):
        #     real_distance_data.append(self.sheet.cell_value(i, 2))

    def aggregateSum(self):
        aggregate = self.stationData.groupby(by = ['station', 'income_date'])
        aggregateSum = aggregate['on_tot'].sum()

        print(*aggregateSum, sep=' ')
    def showStations(self):
        print(*self.stationNum, sep=' ')
        print(*self.stationName, sep=' ')


visualizer = Visualizer("subway.xlsx")
# visualizer.showStations()
visualizer.aggregateSum()
