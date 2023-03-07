import os
from pathlib import Path
import gdown

def main():
    home_dir = Path(os.getcwd())
    dataset_dir: Path = home_dir / "femnist"
    data_dir: Path = dataset_dir / "data"
    centralized_partition: Path = dataset_dir / 'client_data_mappings' / 'centralized'
    centralized_mapping: Path = dataset_dir / 'client_data_mappings' / 'centralized' / '0'
    federated_partition: Path = dataset_dir / 'client_data_mappings' / 'fed_natural'
    #  Download compressed dataset
    if not (home_dir / "femnist.tar.gz").exists() and not dataset_dir.exists():
        id = "1-CI6-QoEmGiInV23-n_l6Yd8QGWerw8-"
        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
            str(home_dir / "femnist.tar.gz"),
        )
        
    # Decompress dataset 
    if not dataset_dir.exists():
        os.system(f"tar -xf {str(home_dir)}/femnist.tar.gz -C {str(home_dir)} 2> /dev/null")
        print(f"Dataset extracted in {dataset_dir}")


if __name__ == "__main__":
    main()
