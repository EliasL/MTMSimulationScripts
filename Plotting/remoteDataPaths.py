from pathlib import Path

LOCAL_REMOTE_DATA_PATH = Path("~/Work/PhD/remoteData").expanduser()
EXTERNAL_REMOTE_DATA_PATH = Path("/Volumes/data/remoteData")


def remote_data_root():
    if EXTERNAL_REMOTE_DATA_PATH.parent.is_dir():
        return EXTERNAL_REMOTE_DATA_PATH
    return LOCAL_REMOTE_DATA_PATH


FOLDER_PATH = str(remote_data_root())
MACRO_PATH = str(Path(FOLDER_PATH) / "macro")
PLOTS_PATH = str(Path(FOLDER_PATH) / "plots")
RAW_DATA_PATH = str(Path(FOLDER_PATH) / "data")
REAL_SPACE_EVENT_PATH = str(Path(FOLDER_PATH) / "real_space_events")
