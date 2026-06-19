import os


class FileTool:
    def __init__(self, folder_path="data/docs"):
        self.folder_path = folder_path

    def run(self, _=None):
        return os.listdir(self.folder_path)

