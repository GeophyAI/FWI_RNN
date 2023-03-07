import numpy as np
import torch, struct
from scipy import signal
from sklearn.metrics import confusion_matrix

class CNN(torch.nn.Module):
    def __init__(self,):
        super(CNN, self).__init__()
        pass

        self.conv2d_1 = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.nonlinear = torch.nn.ReLU()
        self.conv2d_2 = torch.nn.Conv2d(4, 1, 3, padding=1)
        self.init_weights()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.nonlinear()
        x = self.conv2d_2(x)
        return x

    def init_weights(self,): #torch.nn.init.normal()
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight)


def to_tensor(x, dtype=None):
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    if type(x) is np.ndarray:
        return torch.from_numpy(x).type(dtype=dtype)
    else:
        return torch.as_tensor(x, dtype=dtype)


def set_dtype(dtype=None):
    if dtype == 'float32' or dtype is None:
        torch.set_default_dtype(torch.float32)
    elif dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError('Unsupported data type: %s; should be either float32 or float64' % dtype)


def window_data(X, window_length):
    """Window the sample, X, to a length of window_length centered at the middle of the original sample
    """
    return X[int(len(X) / 2 - window_length / 2):int(len(X) / 2 + window_length / 2)]

def accuracy_onehot(y_pred, y_label):
    """Compute the accuracy for a onehot
    """
    return (y_pred.argmax(dim=1) == y_label).float().mean().item()


def normalize_power(X):
    return X / torch.sum(X, dim=1, keepdim=True)
    
def ricker_wave(fm, dt, T, delay = 500, dtype='tensor'):
    """
        Ricker-like wave.
    """
    ricker = []
    delay = delay * dt 
    for i in range(T):
        c = np.pi * fm * (i * dt - delay) #  delay
        temp = (1-2*np.power(c, 2)) * np.exp(-np.power(c, 2))
        ricker.append(temp)
    if dtype == 'numpy':
        return np.array(ricker)
    else:
        return torch.from_numpy(np.array(ricker))

def cpu_fft(d, dt, N = 5, low = 5, if_plot = True, axis = -1, mode = 'lowpass'):
    """
        implementation of fft.
    """
    wn = 2*low/(1/dt)
    b, a = signal.butter(N, wn, mode)
    d_filter = signal.filtfilt(b, a, d, axis = axis)
    return d_filter.astype(np.float32)


def calc_cm(model, dataloader, verbose=True):
    """Calculate the confusion matrix
    """
    with torch.no_grad():
        list_yb_pred = []
        list_yb = []
        i = 1
        for xb, yb in dataloader:
            yb_pred = model(xb)
            list_yb_pred.append(yb_pred)
            list_yb.append(yb)
            if verbose: print("cm: processing batch %d" % i)
            i += 1

        y_pred = torch.cat(list_yb_pred, dim=0)
        y_truth = torch.cat(list_yb, dim=0)

    return confusion_matrix(y_truth.argmax(dim=1).numpy(), y_pred.argmax(dim=1).numpy())
    
def pad_by_value(d, pad, mode = 'double'):
    """pad the input by <pad>
    """
    if mode == 'double':
        return d + 2*pad
    else:
        return d + pad
        
def load_file_by_type(filepath, shape = None, pml_width = None):
    """load data files, differs by its type
    """
    fileType = filepath.split('/')[-1].split('.')[-1]
    if fileType == 'npy':
        return np.load(filepath)
    if fileType == 'dat':
        if shape is not None:
            Nx, Nz = shape
            Nz = Nz - 2*pml_width
            Nx = Nx - 2*pml_width
        else:
            raise ValueError('when the filetype of vel is .dat, the shape must be specified.')
        with open(filepath, "rb") as f:
            d = struct.unpack("f"*Nx*Nz, f.read(4*Nx*Nz))
            d = np.array(d)
            d = np.reshape(d, (Nx, Nz))
        return d
        
def update_cfg(cfg, geom = 'geom'):
    """update the cfg dict, mainly update the Nx and Ny paramters.
    """
    Nx, Ny = cfg[geom]['Nx'], cfg[geom]['Ny']

    if (Nx is None) and (Ny is None) and (cfg[geom]['cPath']):
        vel_path = cfg[geom]['cPath']
        vel = load_file_by_type(vel_path)
        Ny, Nx = vel.shape

    cfg[geom].update({'Nx':Nx + 2*cfg[geom]['pml']['N']})
    cfg[geom].update({'Ny':Ny + 2*cfg[geom]['pml']['N']}) 
    return cfg

    
