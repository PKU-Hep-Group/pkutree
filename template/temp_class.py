class ntuplize(object):
    """
    An example class
    """

    def __init__(self, fin=None, fout=None):
        self.input = fin
        self.output = fout


    def run(self):
        pass


class super_hist(object):
    """
    histogram along with some parameters
    """

    def __init__(self, **kwargs):
        # data=False, signal=False, title='bkg', color='red', scale=1.0, plot_shape=0, shape_norm=1.0
        self.data = kwargs.get('data', False)
        self.signal = kwargs.get('signal', False)
        self.hist = kwargs.get('hist', None)
        self.label = kwargs.get('label', 'bkg')
        self.color = kwargs.get('color', 'red')
        self.scale = kwargs.get('scale', 1.0)
        self.plot_shape = kwargs.get('plot_shape', 0) # plot_shape=0: stack only, plot_shape=1: shape only, plot_shape=2: shape and stack
        self.shape_norm = kwargs.get('shape_norm', 1.0)

    def set_data(self, data):
        self.data = data
    def set_signal(self, signal):
        self.signal = signal
    def set_hist(self, hist):
        self.hist = hist
    def set_label(self, label):
        self.label = label
    def set_color(self, color):
        self.color = color
    def set_scale(self, scale):
        self.scale = scale
    def set_plot_shape(self, plot_shape):
        self.plot_shape = plot_shape
    def set_shape_norm(self, shape_norm):
        self.shape_norm = shape_norm
