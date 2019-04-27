"""Creates logging instances to log for tensorboard vis"""

import hypernn.config as config
import tensorboardX as tb
import logging
logger = logging.getLogger(__file__)

# create the directory in anycase.
# we use it to store training console logs
_dir = str(config.experiment_dir.absolute())
config.experiment_dir.mkdir(parents=True, exist_ok=True)

if config.cmd_args.tb_debug:
    logger.info("Debugging using tensorboardX ...")
    logger.info("Setting up tensorboard logging dir as {}".format(_dir))
    logger.info("Use the following command to start tensorboard\n"
                " $ tensorboard --logdir {}".format(_dir))
    logger.info("Use 0.0.0.0:6006 in browser to view tensorboard")

tb_logger = tb.SummaryWriter(_dir)


# TODO: Remove the default argument and
# and let the caller/main script provide the SummaryWriter instance
# and a logging config dictionary
class TensorboardLoggingMixin(object):
    def __init__(self, *args, tb_logger=tb_logger, **kwargs):
        self.tb_logger = tb_logger
        super().__init__(*args, **kwargs)
