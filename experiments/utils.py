import os
import subprocess
import json


def make_dir(dir, name):
    """Make directory if it doesn't exist, and return its path."""
    new_dir = os.path.join(dir, name)
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir


def save_json(dir, name, obj):
    """Save obj as json."""
    with open(os.path.join(dir, name), "w") as handle:
        json.dump(obj, handle, indent=4)


def create_and_save_class_map(classes, class_map_name, save_dir):
    """Create and save given class clap map to save_dir."""
    class_map = {label: idx for idx, label in enumerate(sorted(classes))}
    save_json(save_dir, f"class_map_{class_map_name}.json", class_map)
    return class_map


def get_git_info(repo_path):
    """Return info about git state of local repo."""
    git_remote_url = "git config --get remote.origin.url"
    git_branch_name = "git rev-parse --abbrev-ref HEAD"
    git_short_sha = "git describe --always --dirty"
    git_info_cmd = (
        f"cd {repo_path} && {git_remote_url} && {git_branch_name} && {git_short_sha}"
    )
    git_info = (
        subprocess.run(git_info_cmd, shell=True, stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .split("\n")
    )
    git_info_string = (
        f"repo / branch / commit: {git_info[0]} / {git_info[1]} / {git_info[2]}"
    )
    return git_info_string
