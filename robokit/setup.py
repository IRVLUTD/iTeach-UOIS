#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2024).
#----------------------------------------------------------------------------------------------------

import os
import requests
import subprocess
import setuptools
from tqdm import tqdm
from setuptools.command.install import install

import logging
from absl import app


class FileFetch(install):
    """
    Custom setuptools command to fetch required files from external sources.
    """
    def run(self):
        """
        Execute the command to fetch required files.
        """
        install.run(self)

        robokit_root_dir = os.getcwd()

        # Install the dependency from the Git repository
        subprocess.run([
            "pip", "install", "-U",
            'git+https://github.com/IDEA-Research/GroundingDINO.git@2b62f419c292ca9c518daae55512fabc3fead4a4',
            # 'git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588'
            'git+https://github.com/ChaoningZhang/MobileSAM@c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed',
        ])

        # Step SAMv2.1: Clone the repository
        samv2_dir = os.path.join(robokit_root_dir, "robokit", "sam2")
        os.makedirs(samv2_dir, exist_ok=True)
        try:
            subprocess.run(["git", "clone", "https://github.com/facebookresearch/sam2", samv2_dir], check=True)
        except:
            pass

        # Step SAMv2.2: cd to samv2 and checkout the desired commit branch
        os.chdir(samv2_dir)
        subprocess.run(["git", "checkout", "--branch", "c2ec8e14a185632b0a5d8b161928ceb50197eddc"])

        # Step SAMv2.3: Use sed to comment out line 171 (to get rid of py>=3.10)
        subprocess.run(["sed", "-i", "171s/^/#/", "setup.py"], check=True)

        # Step SAMv2.4: Install samv2
        subprocess.run(["python", "setup.py", "install"], check=True)

        # Step SAMv2.5: move to robokit root directory
        os.chdir(robokit_root_dir)        

        # subprocess.run([
        #     "conda", "install", "-y", "pytorch", "torchvision", "torchaudio", "pytorch-cuda=11.7", "-c", "pytorch", "-c", "nvidia"
        # ])

        subprocess.call


        # Download GroundingDINO checkpoint
        self.download_pytorch_checkpoint(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            os.path.join(os.getcwd(), "ckpts", "gdino"),
            "gdino.pth"
        )
        
        ##############################################################################################################

        # Download SAM checkpoint
        # self.download_pytorch_checkpoint(
        #     "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        #     os.path.join(os.getcwd(), "ckpts", "sam"),
        #     "vit_h.pth"
        # )

        # Download SAM checkpoint
        self.download_pytorch_checkpoint(
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
            os.path.join(os.getcwd(), "ckpts", "mobilesam"),
            "vit_t.pth"
        )
        
        ##############################################################################################################
        
        # Download SAM2 checkpoint
        self.download_pytorch_checkpoint(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            os.path.join(os.getcwd(), "ckpts", "samv2"),
            "sam2.1_hiera_large.pth"
        )

        # Download SAM2 checkpoint yaml (exploiting the download ckpt method's download nature)
        self.download_pytorch_checkpoint(
            "https://raw.githubusercontent.com/facebookresearch/sam2/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            os.path.join(os.getcwd(), "ckpts", "samv2"),
            "sam2.1_hiera_l.yaml"
        )
        
        ##############################################################################################################


    def download_pytorch_checkpoint(self, pth_url: str, save_path: str, renamed_file: str):
        """
        Download a PyTorch checkpoint from the given URL and save it to the specified path.

        Parameters:
        - pth_url (str): The URL of the PyTorch checkpoint file.
        - save_path (str): The path where the checkpoint will be saved.
        - renamed_file (str, optional): The desired name for the downloaded file.

        Raises:
        - FileNotFoundError: If the file cannot be downloaded or saved.
        - Exception: If an unexpected error occurs during the download process.
        """
        try:
            file_path = os.path.join(save_path, renamed_file)

            # Check if the file already exists
            if os.path.exists(file_path):
                logging.info(f"{file_path} already exists! Skipping download")
                return
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)

            # Log download attempt
            logging.info("Attempting to download PyTorch checkpoint from: %s", pth_url)


            response = requests.get(pth_url, stream=True)
            response.raise_for_status()  # Raise an HTTPError for bad responses

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            # Save the checkpoint to the specified path
            with open(file_path, 'wb') as file:
                for data in response.iter_content(chunk_size=block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()

            logging.info("Checkpoint downloaded and saved to: %s", file_path)

        except FileNotFoundError as e:
            logging.error("Error: Checkpoint file not found: %s", e)
            raise e


def run_setup(argv):
    del argv

    # Read requirements from requirements.txt
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()

    with open("README.md", "r", encoding = "utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        name = "RoboKit",
        version = "0.0.1",
        author = "Jishnu P",
        author_email = "jishnu.p@utdallas.edu",
        description = "A toolkit for robotic tasks",
        long_description = long_description,
        long_description_content_type = "text/markdown",
        url = "https://github.com/IRVLUTD/RoboKit",
        classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        package_dir = {"": "robokit"},
        packages = setuptools.find_packages(where="robokit"),
        python_requires = ">=3.0",
        install_requires=requirements,
        cmdclass={
            'install': FileFetch,
        },
        package_data={'gdino_cfg': ["robokit/cfg/gdino/GroundingDINO_SwinT_OGC.py"]}
    )


if __name__ == "__main__":
    app.run(run_setup)