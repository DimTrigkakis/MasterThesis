import csv

class CSVReader():
    
    target_file = None
    columns = None
    col_to_index = None

    def __init__(self, target_file):
        self.target_file = target_file
        self.extract_info()

    def extract_info(self):
        
        with open(self.target_file, 'rb') as csvfile:
            first_row = True
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
            columns = {}
            col_to_index = {}
            index = 0
            for row in spamreader:
                #print row[0]
                if first_row:
                    first_row = False
                    for r in row:
                        if r != "":
                            col_to_index[r] = index                  
                            columns[r] = []
                        index += 1
                else:
                    for r in columns.keys():
                        columns[r].append(row[col_to_index[r]])
                                       
            self.columns = columns
            self.col_to_index = col_to_index

    # get columns, a collection of all the data accessed through a dictionary
    def get_data(self):
        return self.columns

    # get a specific column, given the key that is the column name
    def get_column(self,key=None):
        if key != None:
            return self.columns[key]
    
    # constraints is a list of tuples, so that given a tuple (key, value), one obtains a dictionary of all the matching rows
    def get_rows(self,constraints=None):
        filtered_rows = {}

        for r in self.columns.keys():
            filtered_rows[r] = []

        for row_index in range(len(self.columns["GameId"])):
            next_row = False          

            for (key, value) in constraints:
                if str(self.columns[key][row_index]).strip("\"") != str(value).strip("\""):
                    next_row = True    
                    break
               
            if next_row == True:
                continue
            else:
                for key in self.columns.keys():
                    filtered_rows[key].append(self.columns[key][row_index])

        return filtered_rows

    def get_row(self, dictionary, index):
        datum = {}
        for key in dictionary.keys():
            datum[key] = dictionary[key][index]
        return datum
# Use cases
# csv_reader = CSVReader("./data/csv_files/pbp-2016.csv")
# print csv_reader.get_column(key="Description")[0:5]
# print csv_reader.get_rows([("GameId","2016090800"),("Description","END QUARTER 1")])
# print csv_reader.get_row(csv_reader.columns,5)
