import os


def get_aml_master_ip(master_name=None):
    if 'AZ_BATCHAI_JOB_MASTER_NODE_IP' in os.environ:
        return os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
    else:
        return None

