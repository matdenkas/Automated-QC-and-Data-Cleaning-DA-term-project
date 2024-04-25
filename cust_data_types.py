class Batch_Info:
    def __init__(self, architecture, batch_size, input_shape, name):
        self.architecture = architecture
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.name = name


class Data_Set:
    def __init__(self, data_training, labels_training, data_validation, labels_validation, data_testing, labels_testing, input_shape):
        self.data_training = data_training
        self.labels_training = labels_training
        self.data_validation = data_validation
        self.labels_validation = labels_validation
        self.data_testing = data_testing
        self.labels_testing = labels_testing
        self.input_shape = input_shape

class Job:
    def __init__(self, batch_info: Batch_Info, data_set: Data_Set, epochs=1000, patience=20, binary= False):
        self.batch_info = batch_info
        self.data_set = data_set
        self.epochs = epochs
        self.patience = patience
        self.binary = binary