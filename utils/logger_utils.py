from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


def setup_loggers(save_dir, run_name, version):
    return [
        setup_TensorBoard_logger(save_dir, run_name, version),
        setup_CSV_logger(save_dir, run_name, version),
    ]


def setup_TensorBoard_logger(save_dir, run_name, version):
    logger = TensorBoardLogger(save_dir, run_name, version)
    return logger


def setup_CSV_logger(save_dir, run_name, version):
    logger = CSVLogger(save_dir, run_name, version)
    return logger
