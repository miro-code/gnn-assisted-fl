import gdown
import os 
from pathlib import Path

path_cwd = os.getcwd()
print(path_cwd)

home_dir = Path(path_cwd)


if not (home_dir / "femnist_dataset.py").exists():
    id = "11xG4oIhdbVcDtXxbS2ZosDSJYAC0iL7q"
    gdown.download(
        f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
        str(home_dir / "femnist_dataset.py"),
    )
if not (home_dir / "client.py").exists():
    id = "11xRc__g3iMOBRiQsPr9mDor5Ile_Pude"
    gdown.download(
        f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
        str(home_dir / "client.py"),
    )
if not (home_dir / "client_utils.py").exists():
    id = "121UMOA7kg96rrZBe7vwt-2vgc-sTGu-X"
    gdown.download(
        f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
        str(home_dir / "client_utils.py"),
    )