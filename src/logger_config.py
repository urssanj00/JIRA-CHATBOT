import logging
from PropertiesConfig import PropertiesConfig as PC

properties_config = PC()
properties = properties_config.get_properties_config()
print(f"{properties['log_path']}")
logging.basicConfig(
    filename=f"{properties['log_path']}/app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime}-{levelname}-{message}",
    datefmt="%Y-%m-%d %H:%M",
    style="{",
    level = logging.INFO  # Set the logger level to DEBUG
)

logger = logging.getLogger()


