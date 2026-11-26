"""OpenSSH argument provider when the historical MLIP-research helper is absent.

Archived TiAl launchers pass their host alias and remote command as the next
arguments. OpenSSH reads identity and connection settings from the user's own
SSH config. Known host keys must already be trusted on this computer.
"""


def ssh_args():
    return ["ssh", "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes"]
