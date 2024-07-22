import logging

# ANSI escape codes for colors
RESET = "\033[0m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
GREEN = "\033[32m"
BOLD_RED = "\033[1;31m"

COLOR_LOG_FORMAT = (
    f"{CYAN}%(asctime)s{RESET} "
    f"%(levelcolor)s%(levelname)s{RESET} "
    f"{MAGENTA}%(filename)s{RESET}:{CYAN}%(lineno)d{RESET} "
    f"%(message)s"
)

LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"


class ColoredFormatter(logging.Formatter):
    level_colors = {
        "DEBUG": BLUE,
        "INFO": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": BOLD_RED,
    }

    def format(self, record):
        level_color = self.level_colors.get(record.levelname, "")
        record.levelcolor = level_color
        return super().format(record)


# Set up the colored formatter
# COLORED_FORMATTER = ColoredFormatter(COLOR_LOG_FORMAT, datefmt='%Y-%m-%d %H:%M:%S')

# Set up the console handler with the colored formatter


def set_log_level(log_level):
    logging.basicConfig(level=log_level, format=LOG_FORMAT)


def get_logger():
    logger = logging.getLogger()
    return logger
