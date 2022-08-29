import subprocess
import os
from src.utils.setup_s3cmd import setup_s3cmd
from src.config import settings as st


LIBRARIES = ["hub", "hub3", "webdataset"]
SERVERS = ["local", "remote_1", "remote_2"]  # "remote_3"]


def set_general():
    os.environ["DYNACONF_DATASET"] = "random"
    os.environ["DYNACONF_BATCH_SIZE"] = "64"
    os.environ["DYNACONF_NUM_WORKERS"] = "0"
    os.environ["DYNACONF_CUTOFF"] = "-1"
    os.environ["DYNACONF_NUM_EPOCHS"] = "2"
    os.environ["DYNACONF_NAME"] = "pure_remote_experiments"


def set_configs(num):

    st.S3_ENDPOINT = st[f"S3_ENDPOINT_{num}"]
    st.AWS_ACCESS_KEY_ID = st[f"AWS_ACCESS_KEY_ID_{num}"]
    st.AWS_SECRET_ACCESS_KEY = st[f"AWS_SECRET_ACCESS_KEY_{num}"]
    st.BUCKET_NAME = st[f"BUCKET_NAME_{num}"]
    st.IS_AWS = st[f"IS_AWS_{num}"]

    print(st[f"S3_ENDPOINT_{num}"])
    os.environ[f"DYNACONF_S3_ENDPOINT"] = str(st[f"S3_ENDPOINT_{num}"])
    os.environ[f"DYNACONF_AWS_ACCESS_KEY_ID"] = st[f"AWS_ACCESS_KEY_ID_{num}"]
    os.environ[f"DYNACONF_AWS_SECRET_ACCESS_KEY"] = st[f"AWS_SECRET_ACCESS_KEY_{num}"]
    os.environ[f"DYNACONF_BUCKET_NAME"] = st[f"BUCKET_NAME_{num}"]
    os.environ[f"DYNACONF_IS_AWS"] = st[f"IS_AWS_{num}"]


# %%
if __name__ == "__main__":

    set_general()

    for lib in LIBRARIES:
        for server in SERVERS:
            print(lib, server)

            os.environ["DYNACONF_LIBRARY"] = lib

            if server == "local":
                st.REMOTE = False
                os.environ["DYNACONF_REMOTE"] = "False"
            else:
                st.REMOTE = True
                os.environ["DYNACONF_REMOTE"] = "True"

            if server == "remote_1":
                set_configs(1)
            elif server == "remote_2":
                set_configs(2)
            elif server == "remote_3":
                set_configs(3)

            setup_s3cmd()

            ## Create dataset
            ARGS = [
                "python",
                "src/datasets/prepare.py",
                "--library",
                lib,
                "--dataset",
                "random",
            ]
            if server != "local":
                ARGS += ["--remote"]
            pid = subprocess.Popen(ARGS)
            pid.wait()

            print(f"Finished generating dataset: {lib} {server}")

            # Run the experiment
            # ARGS = [
            #     "python",
            #     "-Wignore",
            #     "src/run.py"
            #     # "-c",
            #     # "from src.config import settings as st; print(st.as_dict())",
            # ]
            # try:
            #     pid = subprocess.Popen(ARGS)
            # except Exception as e:
            #     print(e)
            #     pid.kill()

            # print(f"Finished running dataset: {lib} {server}")
