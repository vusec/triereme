from pathlib import Path
from abc import ABC, abstractmethod
import shutil

from google.cloud import storage


class RemoteVisitor(ABC):
    @abstractmethod
    def list_subdirs(self, directory: Path) -> set[Path]:
        pass

    @abstractmethod
    def list_files(self, directory: Path) -> set[Path]:
        pass

    @abstractmethod
    def retrieve_file(self, file_path: Path, output_path: Path):
        pass


class GCSVisitor(RemoteVisitor):
    def __init__(self, storage_client: storage.Client, bucket_name: str):
        self.storage_client = storage_client
        self.bucket_name = bucket_name

    def list_subdirs(self, directory: Path) -> set[Path]:
        prefix = f"{directory}/"
        blobs = self.storage_client.list_blobs(
            self.bucket_name, prefix=prefix, delimiter="/"
        )
        # Consume iterator, otherwise prefixes is not initialized
        for _ in blobs:
            pass
        return {Path(path) for path in blobs.prefixes}

    def list_files(self, directory: Path) -> set[Path]:
        prefix = f"{directory}/"
        blobs = self.storage_client.list_blobs(
            self.bucket_name, prefix=prefix, delimiter="/"
        )
        return {Path(blob.name) for blob in blobs}

    def retrieve_file(self, file_path: Path, output_path: Path):
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(str(file_path))
        blob.download_to_filename(output_path)


class LocalVisitor(RemoteVisitor):
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path

    def list_subdirs(self, directory: Path) -> set[Path]:
        storage_directory = self.storage_path / directory
        return {
            dir.relative_to(self.storage_path)
            for dir in sorted(storage_directory.iterdir())
            if dir.is_dir()
        }

    def list_files(self, directory: Path) -> set[Path]:
        storage_directory = self.storage_path / directory
        return {
            file.relative_to(self.storage_path)
            for file in sorted(storage_directory.iterdir())
            if file.is_file()
        }

    def retrieve_file(self, file_path: Path, output_path: Path):
        storage_file = self.storage_path / file_path
        shutil.copyfile(storage_file, output_path)


class RemoteVisitorBuilder(ABC):
    @abstractmethod
    def build(self) -> RemoteVisitor:
        pass


class GCSVisitorBuilder(RemoteVisitorBuilder):
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    def build(self) -> RemoteVisitor:
        return GCSVisitor(storage.Client(), self.bucket_name)


class LocalVisitorBuilder(RemoteVisitorBuilder):
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path

    def build(self) -> RemoteVisitor:
        return LocalVisitor(self.storage_path)
