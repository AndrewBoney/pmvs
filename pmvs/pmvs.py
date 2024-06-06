import numpy as np

import os
import secrets
import safetensors
#import dataiku
#import torch

from functools import partial
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
#from tqdm import tqdm
#from torch.utils.data import DataLoader
from io import BytesIO

class PMVS(dict):
    """
    Poor Man's Vector Store (PMVS)

    A minimal vector store using Python dictionaries to store NumPy arrays and/or PyTorch tensors.

    Attributes:
        local_path (str): The local path where data will be stored.
        session (str): The current session identifier.

    Example:
        pmvs = PMVS()
        pmvs["data"] = torch.tensor([1, 2, 3])
        pmvs.write()
        pmvs.read()
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a PMVS instance.

        Args:
            local_path (str, optional): The local path where data will be stored. Defaults to "~/.PMVS".
            session (str, optional): The current session identifier. Defaults to a random hex string.

        Raises:
            AssertionError: If the specified local_path is not writable.

        Example:
            pmvs = PMVS(local_path="/path/to/store/data", session="my_session")
        """
        self.sections = ["inputs", "inputs_index", "processed", "embeddings"]
        
        # Set defaults if not specified
        if "local_path" not in kwargs.keys():
            self._set_default_path()
        if "session" not in kwargs.keys():
            self._set_default_session()

        # Handle specific kwargs not for dict
        for k, v in list(kwargs.items()):
            if k in ["local_path", "session"]:
                setattr(self, k, v)
                del kwargs[k]

        # Combine local_path and session
        self.full_path = os.path.join(self.local_path, self.session)
        for k in self.sections:
            setattr(self, f"{k}_path", os.path.join(self.full_path, k))

        self._make_path()

        # initialise dictionary
        super().__init__(*args, **kwargs)

        ks = self.keys()

        # parse inputs if stated on init
        if "inputs" in ks:
            self.parse_inputs(self["inputs"])
            self.inputs_parsed = True
        else:
            self.inputs_parsed = False
        
        # add keys if they're not defined on init
        for ka in self.sections:
            if ka not in ks:
                self[ka] = {}
        
        # create functions with defaults set
        # TODO: get working
        #for s in ["inputs", "inputs_index"]:
        #    setattr(self, f"index_{s}", partial(self.index, into_section = s))
        
    def __repr__(self):
        return f"PMVS: {self.session}"
    
    def _set_default_path(self):
        """
        Set the default local_path if not specified by the user.

        The default local_path is "~/.PMVS" if it's writable; otherwise, it uses "/tmp/.PMVS".
        """
        try_path = Path("~/.PMVS").expanduser().absolute()

        if os.access(try_path, os.W_OK):
            self.local_path = try_path
        else:
            tp = Path("/tmp/.PMVS").expanduser().absolute()
            os.makedirs(tp, exist_ok=True)
            self.local_path = tp

            assert os.access(tp, os.W_OK), "No save location found file. Please explicitly specify `local_path`"

    def _set_default_session(self):
        """
        Set the default session identifier if not specified by the user.

        The session identifier is a random 8-character hexadecimal string.
        """
        self.session = secrets.token_hex(8)

    def _make_path(self):
        """
        Create the directories.
        """
        os.makedirs(self.inputs_path, exist_ok = True)
        os.makedirs(self.processed_path, exist_ok = True)
        os.makedirs(self.embeddings_path, exist_ok = True)

    def prep_safetensor_write(self, section : str, key : str):
        """
        TODO:
            Ideally this could be simplified by reading / writing the whole PMVS as a single safetensor.
            The issue with that right now is
                - it would mean reading the whole PMVS into the same format, 
                    which wouldn't work with str inputs -> torch 
                    (... turns out this doesn't work anyway with safetensors)... probs should refactor
                - it would mean having to read *everything* if getting from remote (s3/dataiku folder). 
                    (although there might be a way to do this that I can't work out) 
        """        
        # if is numpy array and object instead convert to empty dict with strings as keys
        # TODO: major refactor in favour of each input being it's own 
        # {"inputs" : ..., "processed" : ..., "embeddings" : ...} dict
        # ... or just use an existing vector store if you can find one thats dataiku compatible 
        if section == "inputs":
            tensor_dict = {v : np.array([]) for v in self[section][key]}
        else:
            tensor_dict = {"arr" : self[section][key]}
        
        if isinstance(self[section][key], np.ndarray): 
            from safetensors.numpy import save_file
        elif isinstance(self[section][key], torch.Tensor):
            from safetensors.torch import save_file
        else:
            raise TypeError("tensor_dict should contain torch or numpy")

        return tensor_dict, save_file

    def get_format(self, section : str):
        return "torch" if section in ["processed", "embeddings"] else "numpy"
    
    def get_safetensor_load(self, section : str):
        format_ = self.get_format(section)
        if format_ == "numpy":
            from safetensors.numpy import load_file
        else:
            from safetensors.torch import load_file
        
        return load_file

    def get_full_path(self, section : str, key : str):
        return os.path.join(self.full_path, section, f"{key}.safetensors")
    
    def _write_array_to_disk(self, section : str, key : str):
        td, save_file = self.prep_safetensor_write(section, key)
        save_file(td, self.get_full_path(section, key))
    
    def _write_section_to_disk(self, section : str):
        """
        Write multiple NumPy arrays to disk.

        Args:
            keys (List[str]): A list of keys corresponding to the NumPy arrays to write.
        """
        keys = self[section].keys()
        
        for k in keys:
            self._write_array_to_disk(section, k)
    
    def _write_array_to_dataiku(self, section : str, key : str, folder_str : str, overwrite : bool = True):
        fn = self.get_full_path(section, key)

        exists = os.path.exists(fn)
        if overwrite or (not exists):
            self._write_array_to_disk(section, key)

        dku_outpath = os.path.join(self.session, section, f"{key}.safetensors")
        folder = dataiku.Folder(folder_str)
        folder.upload_file(dku_outpath, fn)
    
    def _write_section_to_dataiku(self, section : str, folder_str : str):
        keys = self[section].keys()

        for k in keys:
            self._write_array_to_dataiku(section, k, folder_str)    

    def write(self, to: str = "disk", *args, **kwargs):
        """
        Write data to disk or other destinations (e.g. cloud storage).

        Args:
            to (str): The destination to write data to (e.g. "disk" for local storage).

        Raises:
            ValueError: If the destination is not supported.

        Example:
            pmvs.write()  # Write data to disk
        """
        implemented = {m.split("_")[-1] : m for m in dir(self) if "_write_section_to" in m} 
        
        assert to in implemented.keys(), f"to = {to} is not currently supported."
        
        for s in self.sections:
            if s != "inputs_index": # gets recreated on read
                getattr(self, implemented[to])(s, *args, **kwargs)
    
    def _read_array_from_disk(self, section : str, key : str, device = 'cpu'):        
        load_file = self.get_safetensor_load(section)
        arr = load_file(self.get_full_path(section, key), device = device)["arr"]
        if section == "inputs":
            arr = np.array(list(arr.keys()))
        
    def _read_section_from_disk(self, section : str, device = 'cpu'):
        section_path = os.path.join(self.full_path, section)
        arrays = os.listdir(section_path)

        return {os.path.splitext(k)[0]: self._read_array_from_disk(section, k, device) for k in arrays}

    def _read_array_from_dataiku(self, section : str, key : str, folder_str : str):
        folder = dataiku.Folder(folder_str)

        fn = os.path.join(self.session, section, key)
        with folder.get_download_stream(fn) as stream:
            data = stream.read()

        lib_str = self.get_format(section)
        if lib_str == "numpy":
            from safetensors.numpy import load
        else:
            from safetensors.torch import load
        
        out = load(data)
        if section == "inputs":
            return np.array(list(out.keys()))
        else:
            return out["arr"]

    def _read_section_from_dataiku(self, section : str, folder_str : str):
        folder = dataiku.Folder(folder_str)
        paths = [s.split("/")[-1] for s in folder.list_paths_in_partition() 
                 if self.session in s and section in s]
        
        return {os.path.splitext(k)[0]: self._read_array_from_dataiku(section, k, folder_str) for k in paths}

    def read(self, frm: str = "disk", *args, **kwargs):
        """
        Read data from disk or other sources.

        Args:
            frm (str): The source from which to read data (e.g., "disk" for local storage).

        Raises:
            ValueError: If the source is not supported.

        Example:
            pmvs.read()  # Read data from disk
        """
        implemented = {m.split("_")[-1] : m for m in dir(self) if "_read_section_from" in m} 
        
        assert frm in implemented.keys(), f"from = {frm} is not currently supported."
        
        for s in self.sections:
            if s != "inputs_index":
                dicts = getattr(self, implemented[frm])(s, *args, **kwargs)
                for k, v in dicts.items():
                    self[s][k] = v
                    
        self.parse_inputs(self["inputs"])
                
    # soft deprecated. .read prefered
    def load_session(self, session: str):
        """
        Load a specific session's data from disk.

        Args:
            session (str): The session identifier for the session to load.

        Raises:
            AssertionError: If the specified session directory does not exist.

        Example:
            pmvs.load_session("my_session")
        """
        full_path = os.path.join(self.local_path, session)
        assert os.path.exists(full_path)

        arrays = self._read_arrays_from_disk(os.listdir(full_path))
        for k, v in arrays.items():
            self[k] = v

    def process_vectorized(self, inputs: str, processing_func: callable, *args, **kwargs):
        """
        Process inputs. Requires a `processing_func` that works on the whole dataset 
        (e.g. most text tokenizers) rather than individual examples (e.g. most image
        processing pipelines). 

        Args:
            inputs (str): The key for the input data to be processed.
            processing_func (callable): A function to process the input data.
            *args: Additional positional arguments to pass to the processing function.
            **kwargs: Additional keyword arguments to pass to the processing function.

        Example:
            pmvs.process_vectorized("input_data", my_processing_function)
        """
        if not self.inputs_parsed:
            self.parse_inputs(self["inputs"])
        
        # Assumes processing_func is vectorized
        self["processed"][inputs] = processing_func(self["inputs"][inputs], *args, **kwargs)

    def process(self, inputs: str, processing_func: callable, *args, **kwargs):
        """
        Process data element-wise and store the result.

        Args:
            inputs (str): The key for the input data to be processed.
            processing_func (callable): A function to process each element of the input data.
            *args: Additional positional arguments to pass to the processing function.
            **kwargs: Additional keyword arguments to pass to the processing function.

        Example:
            pmvs.process("input_data", my_processing_function)
        """
        if not self.inputs_parsed:
            self.parse_inputs(self["inputs"])
        
        # Assumes processing_func runs on individual examples and produces a single-element tensor
        self["processed"][inputs] = torch.stack([processing_func(i, *args, **kwargs)
                                                for i in tqdm(self["inputs"][inputs])], dim=0)

    def get_embeddings(self, processed: str, func: callable, *args, **kwargs):
        """
        Compute embeddings from processed data and store the result.

        Args:
            processed (str): The key for the processed data from which embeddings will be computed.
            func (callable): A function to compute embeddings from the processed data.
            *args: Additional positional arguments to pass to the embedding function.
            **kwargs: Additional keyword arguments to pass to the embedding function.

        Example:
            pmvs.get_embeddings("processed_data", my_embedding_function)
        """
        self["embeddings"][processed] = func(self["processed"][processed], *args, **kwargs) 
        
    def get_similarity(self, a_embed : str, b_embed : str, scale : float = None):        
        # if scale isn't specified use 100 (~cosine similarity I think?) 
        if scale is None:
            scale = 100.
            
        a = self["embeddings"][a_embed]
        b = self["embeddings"][b_embed]
        
        assert a.device == b.device
        
        a = a / a.norm(dim=-1, keepdim = True)
        b = b / b.norm(dim=-1, keepdim = True)

        return scale * a @ b.T

    def copy(self, new_name : str):
        new = self
        new.session = new_name
        return new
    
    def index(self, 
              query : Union[List[str], np.ndarray], 
              into_section : str, 
              into_key : str, 
              return_key : str = None
    ):
        if not self.inputs_parsed:
            self.parse_inputs(self["inputs"])
        
        if return_key is None:
            return_key = into_key
                
        query = self.parse_type(query)
        s = query.shape
        long = query.reshape(-1)
        ixs = np.array([self["inputs_index"][into_key][l] for l in long]).reshape(s)
        
        arr_out = self.parse_type
        md = len(self[into_section][return_key].shape) > 1
        if md:
            return self[into_section][return_key][ixs, :]
        else:
            return self[into_section][return_key][ixs]
    
    def index_embeddings(
        self, 
        query : Union[List[str], np.ndarray],  
        into_key : str, 
        return_key : str = None
    ):
        if return_key is None:
            return_key = into_key
        
        return self.index(query, "embeddings", into_key, return_key)
    
    def parse_type(self, arr : Union[np.ndarray, Union[List[str], List[int]]]):
        if isinstance(arr, list):
            arr = np.array(arr)
        elif isinstance(arr, np.ndarray):
            pass
        else:
            raise TypeError("inputs should be a np array or a list")
        
        return arr
        
    def add_index(self, arr : Union[np.ndarray, Union[List[str], List[int]]]):
        arr = self.parse_type(arr)
        return {el : i for i, el in enumerate(list(arr))}
        
    def parse_input(self, arr : Union[np.ndarray, Union[List[str], List[int]]]):
        # TODO: set inputs_parsed to a dict stating each input that is parsed
        arr = self.parse_type(arr)
        return np.unique(arr)

    def parse_inputs(self, inputs : Dict[str, Union[np.ndarray, Union[List[str], List[int]]]]):
        self["inputs"] = {k : self.parse_input(v) for k, v in inputs.items()}
        self["inputs_index"] = {k : self.add_index(v) for k, v in self["inputs"].items()}
        self.inputs_parsed = True