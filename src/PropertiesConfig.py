import os

from jproperties import Properties


# class definition
class PropertiesConfig:
    @staticmethod
    def is_running_in_colab():
        """
        Determines if the code is running in Google Colab.
        """
        return "COLAB_GPU" in os.environ or "COLAB_BACKEND_VERSION" in os.environ

    #test comment
    def __init__(self, properties_file_name="sensor-data.properties"):
        if PropertiesConfig.is_running_in_colab():
            properties_file_name = os.path.join(os.path.dirname(__file__), 'config_collab.properties')
            print(f"            LOADING PROPERTIES {properties_file_name} FROM COLLAB ENVIRONMENT")
        else:
            properties_file_name = os.path.join(os.path.dirname(__file__), 'config.properties')
            print(f"            LOADING PROPERTIES {properties_file_name} FROM LOCAL ENVIRONMENT")

        configs = Properties()
        with open(properties_file_name, 'rb') as read_prop:
            configs.load(read_prop)
        self.propertiesConfig = configs.items()

    def get_properties_config(self):
        properties_dict = {}
        for item in self.propertiesConfig:
            key, prop_tuple = item
            properties_dict[key] = prop_tuple.data
        return properties_dict

# Create an instance of PropertiesConfig
#propertiesConfig = PropertiesConfig("sensor-data.properties")
# Get properties as a dictionary
#properties = propertiesConfig.get_properties_config()

#st = properties['station_list_url']
#pt = properties['data_set_path']
#print(f'station_list_url:{st}')
#print(f'data_set_path:{pt}')
