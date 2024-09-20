from __future__ import annotations

import typing
from abc import ABC, abstractmethod

import h5py
import numpy as np

from torchxai import *  # noqa


class HFDataset:
    """
    A class for saving a dataset in HDF5 format.
    """

    def __init__(self, filepath: str, mode="a"):
        self.filepath = filepath
        self.mode = mode
        self.hf = None

    def __enter__(self) -> HFIO:
        self.hf = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hf:
            self.hf.close()
            self.hf = None

    def save(self, key: str, data: np.ndarray) -> None:
        """
        Saves the dataset to the HDF5 file under the given key.

        Args:
            key (str): The key to identify the data.
            data: The data to be saved.

        Raises:
            ValueError: If the data cannot be saved.
        """
        if key in self.hf:
            raise ValueError(f"Key '{key}' already exists in the file.")

        # Create a dataset with max shape to allow future resizing
        self.hf.create_dataset(key, data=data.cpu().numpy())

    def load(self, key: str) -> np.ndarray:
        """
        Loads the dataset from the HDF5 file under the given key.

        Args:
            key (str): The key to identify the data.
            data: The data to be saved.
        """
        if key not in self.hf:
            raise KeyError(f"Key '{key}' does not exist under key '{key}'.")

        return self.hf[key][:]

    def key_exists(self, key: str) -> np.ndarray:
        return key in self.hf


class HFIO(ABC):
    """
    Abstract base class for handling HDF5 input/output operations.

    Attributes:
        filepath (str): The path to the HDF5 file.

    Methods:
        save(key, data, sample_key): Save data to the HDF5 file under the specified key.
        load(key, sample_key): Load data from the HDF5 file using the specified key.
    """

    def __init__(self, filepath: str, mode="a"):
        self.filepath = filepath
        self.mode = mode
        self.hf = None

    def __enter__(self) -> HFIO:
        self.hf = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hf:
            self.hf.close()
            self.hf = None

    def save_attribute(self, key: str, data, sample_key: str) -> None:
        if sample_key not in self.hf:
            self.hf.create_group(sample_key)
        if self.hf[sample_key].attrs.get(key) is None:
            self.hf[sample_key].attrs[key] = data

    @abstractmethod
    def save(self, key: str, data: typing.Any, sample_key: str):
        raise NotImplementedError()

    @abstractmethod
    def load(self, key: str, sample_key: str):
        raise NotImplementedError()


class HFIOSingleOutput(HFIO):
    """
    A class for handling single output HDF5 input/output operations.

    Attributes:
        filepath (str): Path to the HDF5 file.
        hf (h5py.File): The HDF5 file object.
    """

    def save(self, key: str, data, sample_key: str) -> None:
        """
        Saves the data to the HDF5 file under the given key and sample key.

        Args:
            key (str): The key to identify the data.
            data: The data to be saved.
            sample_key (str): The key to identify the sample.

        Raises:
            ValueError: If the data cannot be saved.
        """
        if sample_key not in self.hf:
            self.hf.create_group(sample_key)

        if key not in self.hf[sample_key]:
            # Create a dataset for the key
            if isinstance(data, np.ndarray):
                self.hf[sample_key].create_dataset(
                    key, data=[data], maxshape=(None, *data.shape)
                )
            else:
                self.hf[sample_key].create_dataset(key, data=data)
        else:
            # Create a dataset for the key
            if isinstance(data, np.ndarray):
                # Overwrite the existing data
                self.hf[sample_key][key][0] = data
            else:
                self.hf[sample_key][key][...] = data

    def load(self, key: str, sample_key: str):
        """
        Loads the data from the HDF5 file using the given key and sample key. If the key or sample key doesn't exist,

        Args:
            key (str): The key to identify the data.
            sample_key (str): The key to identify the sample.

        Returns:
            The loaded data if it exists, otherwise None.

        Raises:
            KeyError: If the sample_key or key doesn't exist in the HDF5 file.
        """
        if sample_key not in self.hf or key not in self.hf[sample_key]:
            return None

        return self.hf[sample_key][key][...]


class HFIOMultiOutput(HFIO):
    """
    A class for handling multiple output HDF5 input/output operations with batch processing.
    """

    def save(
        self, key: str, data, sample_key: str, batch_start_idx: int, batch_end_idx: int
    ) -> None:
        """
        Saves a batch of data to the HDF5 file under the given key and sample key.

        Args:
            key (str): The key to identify the data.
            data: The data to be saved.
            sample_key (str): The key to identify the sample.
            batch_start_idx (int): The starting index for the batch.
            batch_end_idx (int): The ending index for the batch.
        """
        if sample_key not in self.hf:
            self.hf.create_group(sample_key)

        if key not in self.hf[sample_key]:
            # Create a dataset with max shape to allow future resizing
            shape = (batch_end_idx,)
            self.hf[sample_key].create_dataset(
                key, shape=shape, maxshape=(None,), chunks=True
            )
        else:
            dataset = self.hf[sample_key][key]
            # Resize dataset if necessary to accommodate new data
            current_shape = dataset.shape[0]
            if batch_end_idx > current_shape:
                dataset.resize((batch_end_idx,))

        # Save data to the specified range
        self.hf[sample_key][key][batch_start_idx:batch_end_idx] = data

    def load(self, key: str, sample_key: str, batch_start_idx: int, batch_end_idx: int):
        """
        Loads a batch of data from the HDF5 file using the given key and sample key.

        Args:
            key (str): The key to identify the data.
            sample_key (str): The key to identify the sample.
            batch_start_idx (int): The starting index for the batch.
            batch_end_idx (int): The ending index for the batch.

        Returns:
            The loaded batch of data if it exists, otherwise None.

        Raises:
            KeyError: If the sample_key or key doesn't exist in the HDF5 file.
        """
        if sample_key not in self.hf:
            raise KeyError(f"Sample key '{sample_key}' does not exist in the file.")

        if key not in self.hf[sample_key]:
            raise KeyError(
                f"Key '{key}' does not exist under sample key '{sample_key}'."
            )

        return self.hf[sample_key][key][batch_start_idx:batch_end_idx]
