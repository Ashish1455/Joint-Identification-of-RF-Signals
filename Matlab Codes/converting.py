#!/usr/bin/env python3
"""
MATLAB v7.3 to NumPy Converter - Fixed variable naming
"""

import warnings
import numpy as np
import scipy.io
import sys
import os
import argparse

warnings.filterwarnings('ignore', category=UserWarning, module='h5py')

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

def get_variable_name_mapping():
    """Define mapping for common MATLAB variable patterns"""
    return {
        'signals_real': ['signals_real', 'real', 'real_part', 'real_data'],
        'signals_imag': ['signals_imag', 'imag', 'imag_part', 'imag_data'],
        'labels_class': ['labels_class', 'labels', 'class_labels', 'classes'],
        'signal_length': ['signal_length', 'length'],
        'num_signals': ['num_signals', 'total_signals'],
        'num_classes': ['num_classes', 'total_classes']
    }

def map_variable_name(h5_key, variable_mapping):
    """Map HDF5 keys to proper variable names"""
    # First check exact matches
    for proper_name, possible_names in variable_mapping.items():
        if h5_key in possible_names:
            return proper_name
    
    # Check partial matches
    h5_key_lower = h5_key.lower()
    for proper_name, possible_names in variable_mapping.items():
        for possible_name in possible_names:
            if possible_name.lower() in h5_key_lower or h5_key_lower in possible_name.lower():
                return proper_name
    
    # Keep original name if no match found
    return h5_key

def process_cell_array(h5file, dataset):
    """Process MATLAB cell arrays containing references"""
    data = np.array(dataset)
    processed_items = []
    
    for ref in data.flat:
        if isinstance(ref, h5py.Reference):
            try:
                ref_obj = h5file[ref]
                if isinstance(ref_obj, h5py.Dataset):
                    ref_data = np.array(ref_obj)
                    
                    # Handle string data
                    if ref_data.dtype.kind in ['U', 'S', 'u']:
                        if ref_data.size > 0:
                            try:
                                if ref_data.dtype.kind == 'u':
                                    chars = [chr(int(x)) for x in ref_data.flatten() 
                                           if 0 < x < 1114112 and x != 0]
                                    processed_items.append(''.join(chars))
                                else:
                                    processed_items.append(ref_data.tobytes().decode('utf-8', errors='ignore').strip())
                            except:
                                processed_items.append('')
                        else:
                            processed_items.append('')
                    else:
                        processed_items.append('')
                else:
                    processed_items.append('')
            except:
                processed_items.append('')
        else:
            processed_items.append(str(ref) if ref is not None else '')
    
    return np.array(processed_items, dtype=object).reshape(data.shape)

def load_mat_v73_with_proper_names(filename):
    """Load MATLAB v7.3 file with proper variable naming"""
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for v7.3 MAT files")
    
    data_dict = {}
    variable_mapping = get_variable_name_mapping()
    
    with h5py.File(filename, 'r') as h5file:
        print(f"HDF5 keys found: {list(h5file.keys())}")
        
        def process_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                key = name.split('/')[-1]
                if key.startswith('#'):
                    return
                
                print(f"Processing HDF5 key: {key}")
                
                try:
                    data = np.array(obj)
                    
                    # Map to proper variable name
                    proper_name = map_variable_name(key, variable_mapping)
                    print(f"  Mapped to: {proper_name}")
                    
                    # Handle object arrays (cell arrays with references)
                    if obj.dtype.kind == 'O' and data.size > 0:
                        data_dict[proper_name] = process_cell_array(h5file, obj)
                    
                    # Handle string data
                    elif data.dtype.kind in ['U', 'S', 'u']:
                        if data.size > 0:
                            try:
                                if data.dtype.kind == 'u':
                                    chars = [chr(int(x)) for x in data.flatten() 
                                           if 0 < x < 1114112 and x != 0]
                                    data_dict[proper_name] = ''.join(chars)
                                else:
                                    data_dict[proper_name] = data.tobytes().decode('utf-8', errors='ignore').strip()
                            except:
                                data_dict[proper_name] = str(data)
                        else:
                            data_dict[proper_name] = ''
                    
                    # Handle numeric data
                    else:
                        # Transpose for MATLAB compatibility if needed
                        if data.ndim == 2 and data.shape[0] != data.shape[1] and min(data.shape) > 1:
                            data = data.T
                        data_dict[proper_name] = data
                        
                    print(f"  Saved as: {proper_name} with shape {data_dict[proper_name].shape if hasattr(data_dict[proper_name], 'shape') else type(data_dict[proper_name])}")
                        
                except Exception as e:
                    print(f"  Error processing {key}: {e}")
        
        h5file.visititems(process_item)
    
    return data_dict

def convert_mat_to_npz_with_names(mat_filename, npz_filename=None):
    """Convert MATLAB file to NPZ with proper variable names"""
    
    if not os.path.exists(mat_filename):
        raise FileNotFoundError(f"File '{mat_filename}' not found!")
    
    if npz_filename is None:
        npz_filename = mat_filename.replace('.mat', '.npz')
    
    file_size_gb = os.path.getsize(mat_filename) / (1024**3)
    print(f"Converting: {mat_filename} ({file_size_gb:.2f} GB)")
    
    # Try regular scipy.io first for smaller files
    data_dict = None
    if file_size_gb < 2.0:
        try:
            data_dict = scipy.io.loadmat(mat_filename)
            data_dict = {k: v for k, v in data_dict.items() if not k.startswith('__')}
            print("Loaded with scipy.io.loadmat")
        except:
            pass
    
    # Use HDF5 method for v7.3 files
    if data_dict is None:
        data_dict = load_mat_v73_with_proper_names(mat_filename)
        print("Loaded with HDF5 method")
    
    if not data_dict:
        raise ValueError("No data loaded from MAT file")
    
    # Display final variable names and sizes
    print("\nFinal variables to save:")
    total_size_mb = 0
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            size_mb = value.nbytes / (1024**2)
            total_size_mb += size_mb
            print(f"  {key:20} : {str(value.shape):20} ({size_mb:.1f} MB)")
        else:
            print(f"  {key:20} : {type(value)}")
    
    print(f"Total data: {total_size_mb:.1f} MB")
    
    # Save as NPZ
    print(f"\nSaving to: {npz_filename}")
    np.savez_compressed(npz_filename, **data_dict)
    
    output_size_mb = os.path.getsize(npz_filename) / (1024**2)
    print(f"Output size: {output_size_mb:.1f} MB")
    
    return True

def verify_npz_with_names(npz_filename):
    """Verify NPZ file with proper variable names"""
    try:
        print(f"\nVerifying: {npz_filename}")
        with np.load(npz_filename, allow_pickle=True) as data:
            print("Variables in NPZ file:")
            for key in sorted(data.files):
                array = data[key]
                if hasattr(array, 'shape'):
                    print(f"  {key:20} : {str(array.shape):20} {str(array.dtype):10}")
                else:
                    print(f"  {key:20} : {type(array)}")
        print("✓ Verification successful")
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert MATLAB v7.3 files to NPZ with proper names')
    parser.add_argument('mat_file', nargs='?', default='signal_dataset_final.mat')
    parser.add_argument('npz_file', nargs='?', default=None)
    parser.add_argument('--verify', action='store_true', help='Verify output file')
    
    args = parser.parse_args()
    
    print("MATLAB v7.3 to NPZ Converter (Fixed Naming)")
    print("=" * 50)
    
    try:
        convert_mat_to_npz_with_names(args.mat_file, args.npz_file)
        
        output_file = args.npz_file or args.mat_file.replace('.mat', '.npz')
        
        if args.verify:
            verify_npz_with_names(output_file)
        
        print(f"\n{'='*50}")
        print("✓ Conversion completed successfully!")
        print(f"\nTo load in Python:")
        print(f"import numpy as np")
        print(f"data = np.load('{output_file}', allow_pickle=True)")
        print(f"signals_real = data['signals_real']")
        print(f"signals_imag = data['signals_imag']")
        print(f"labels_class = data['labels_class']")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
